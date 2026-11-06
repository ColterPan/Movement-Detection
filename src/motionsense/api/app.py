"""FastAPI adapter: REST + WebSocket + MJPEG stream + dashboard.

The app only depends on the small ``ServiceLike`` protocol, so it can be tested with fakes
and reused in front of any other pipeline implementation.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from typing import Annotated, Any, Protocol
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from motionsense import __version__
from motionsense.config import Settings
from motionsense.events.bus import EventBus
from motionsense.events.models import Event, EventType
from motionsense.sinks.snapshot import SnapshotRecorder
from motionsense.types import SceneState

MJPEG_BOUNDARY = "frame"
WS_EVENT_BATCH = 50


class ServiceLike(Protocol):
    settings: Settings
    bus: EventBus
    snapshots: SnapshotRecorder | None

    @property
    def latest_state(self) -> SceneState: ...

    def latest_jpeg(self) -> bytes | None: ...

    def status(self) -> dict[str, Any]: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _token_ok(expected: str | None, provided: str | None) -> bool:
    if not expected:
        return True
    if provided is None:
        return False
    return secrets.compare_digest(expected.encode("utf-8"), provided.encode("utf-8"))


def _is_local_only(settings: Settings) -> bool:
    """Bound to loopback without a token: only this machine's own pages may talk to us."""
    return settings.api.host in LOOPBACK_HOSTS and not settings.api.token


def _origin_allowed(origin: str | None, local_only: bool) -> bool:
    """Browsers always send Origin on WebSockets; non-browser clients usually do not."""
    if not origin or not local_only:
        return True
    return (urlsplit(origin).hostname or "") in LOOPBACK_HOSTS


def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("token")


def create_app(service: ServiceLike, manage_lifecycle: bool = True) -> FastAPI:
    settings = service.settings

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if manage_lifecycle:
            service.start()
        try:
            yield
        finally:
            if manage_lifecycle:
                service.stop()

    app = FastAPI(
        title="MotionSense API",
        version=__version__,
        description="Real-time activity recognition and fall detection. Events follow a versioned "
        "schema (see /api/v1/schema/event). Subscribe to /ws for live state and events.",
        lifespan=lifespan,
    )
    local_only = _is_local_only(settings)
    if local_only:
        # Blocks DNS-rebinding attacks from web pages pretending to be "localhost".
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.api.token else ["http://localhost", "http://127.0.0.1"],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    def require_token(request: Request) -> None:
        if not _token_ok(settings.api.token, _extract_token(request)):
            raise HTTPException(status_code=401, detail="missing or invalid token")

    auth = [Depends(require_token)]
    static_dir = resources.files("motionsense.api") / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        return HTMLResponse((static_dir / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/v1/health", tags=["system"])
    def health() -> dict[str, Any]:
        return {"version": __version__, **service.status()}

    @app.get("/api/v1/state", tags=["scene"], dependencies=auth)
    def state(keypoints: bool = False) -> dict[str, Any]:
        """Current people (activities with probabilities, fall risk, reps) and objects."""
        return service.latest_state.to_dict(include_keypoints=keypoints)

    @app.get("/api/v1/events", tags=["events"], dependencies=auth)
    def events(
        since: Annotated[
            int | None,
            Query(ge=0, description="Page forward: events with seq > since. Omit for the latest."),
        ] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        type: Annotated[list[EventType] | None, Query()] = None,
    ) -> dict[str, Any]:
        items = service.bus.history(since_seq=since, limit=0)  # filter by type before limiting
        if type:
            wanted = set(type)
            items = [e for e in items if e.type in wanted]
        # With `since` the *next* events are returned (paging forward); without it, the latest.
        items = items[-limit:] if since is None else items[:limit]
        return {"last_seq": service.bus.last_seq, "events": [e.to_dict() for e in items]}

    @app.get("/api/v1/events/{event_id}", tags=["events"], dependencies=auth)
    def event_detail(event_id: str) -> dict[str, Any]:
        event = service.bus.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        return event.to_dict()

    @app.get("/api/v1/events/{event_id}/snapshots", tags=["events"], dependencies=auth)
    def snapshots(event_id: str) -> dict[str, Any]:
        """Incident frames in time order. ``url`` is annotated (skeletons, ids, risk);
        ``raw_url`` is the untouched camera frame. ``offset_s`` is relative to the event."""
        recorder = service.snapshots
        frames = recorder.frames(event_id) if recorder else []
        base = f"/api/v1/events/{event_id}/snapshots"
        stored = service.bus.get(event_id)
        return {
            "event_id": event_id,
            "event": stored.to_dict()
            if stored
            else (recorder.event_info(event_id) if recorder else None),
            "frames": [
                {
                    "phase": f["phase"],
                    "offset_s": f["offset_s"],
                    "url": f"{base}/{f['name']}",
                    "raw_url": f"{base}/{f['raw']}",
                }
                for f in frames
            ],
            "snapshots": [f"{base}/{f['name']}" for f in frames],
        }

    @app.get("/api/v1/incidents", tags=["events"], dependencies=auth)
    def incidents(limit: Annotated[int, Query(ge=1, le=100)] = 20) -> dict[str, Any]:
        """Saved incidents (falls, inactivity) with snapshots, newest first; kept across restarts."""
        recorder = service.snapshots
        return {"incidents": recorder.incidents(limit) if recorder else []}

    @app.get("/api/v1/events/{event_id}/snapshots/{name}", tags=["events"], dependencies=auth)
    def snapshot_file(event_id: str, name: str) -> FileResponse:
        recorder = service.snapshots
        path = recorder.snapshot_path(event_id, name) if recorder else None
        if path is None:
            raise HTTPException(status_code=404, detail="snapshot not found")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/v1/schema/event", tags=["events"])
    def event_schema() -> dict[str, Any]:
        """JSON Schema of the event payload (for webhook / WebSocket consumers)."""
        return Event.model_json_schema()

    @app.get("/api/v1/frame.jpg", tags=["video"], dependencies=auth)
    def frame() -> Response:
        jpeg = service.latest_jpeg()
        if jpeg is None:
            raise HTTPException(status_code=503, detail="no frame yet")
        return Response(
            content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    @app.get("/stream.mjpg", tags=["video"], dependencies=auth)
    async def stream() -> StreamingResponse:
        interval = 1.0 / settings.api.stream_fps

        async def gen() -> AsyncIterator[bytes]:
            last: bytes | None = None
            while True:
                jpeg = service.latest_jpeg()
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield (
                        (
                            f"--{MJPEG_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n"
                        ).encode()
                        + jpeg
                        + b"\r\n"
                    )
                await asyncio.sleep(interval)

        return StreamingResponse(
            gen(), media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}"
        )

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        """Pushes ``{"type": "state", ...}`` at api.state_hz and ``{"type": "event", ...}``."""
        if not _origin_allowed(websocket.headers.get("origin"), local_only) or not _token_ok(
            settings.api.token, websocket.query_params.get("token")
        ):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        interval = 1.0 / settings.api.state_hz
        since = max(0, service.bus.last_seq - 20)  # send a little recent history on connect
        include_kp = websocket.query_params.get("keypoints", "false").lower() == "true"
        try:
            while True:
                for event in service.bus.history(since_seq=since, limit=WS_EVENT_BATCH):
                    await websocket.send_json({"type": "event", "event": event.to_dict()})
                    since = event.seq
                await websocket.send_json(
                    {
                        "type": "state",
                        "status": service.status(),
                        "state": service.latest_state.to_dict(include_keypoints=include_kp),
                    }
                )
                await asyncio.sleep(interval)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            with contextlib.suppress(RuntimeError):
                await websocket.close()

    return app
