"""Command-line interface: ``motionsense serve | bench | record | eval | models | scenarios``."""

from __future__ import annotations

import logging
import webbrowser
from pathlib import Path
from typing import Annotated, Any

import typer

from motionsense.config import Settings, load_settings

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Real-time activity recognition and fall detection.",
)

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="YAML config file")]
SourceOpt = Annotated[
    str | None,
    typer.Option(
        "--source",
        "-s",
        help="Webcam index (0), video file, rtsp://, http:// or synthetic[:scenario]",
    ),
]
DeviceOpt = Annotated[str | None, typer.Option(help="Inference device: cpu | intel:gpu | cuda:0")]
ImgszOpt = Annotated[int | None, typer.Option(help="Model input size, e.g. 480 for more FPS")]


def _settings(config: Path | None, **overrides: Any) -> Settings:
    nested: dict[str, Any] = {}
    mapping = {
        "source": ("camera", "source"),
        "device": ("models", "device"),
        "imgsz": ("models", "imgsz"),
        "runtime": ("models", "runtime"),
        "host": ("api", "host"),
        "port": ("api", "port"),
        "token": ("api", "token"),
        "objects": ("objects", "enabled"),
    }
    for key, value in overrides.items():
        if value is None:
            continue
        section, field = mapping[key]
        nested.setdefault(section, {})[field] = value
    if config is None and Path("config/default.yaml").is_file():
        config = Path("config/default.yaml")
    return load_settings(config, nested)


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def serve(
    config: ConfigOpt = None,
    source: SourceOpt = None,
    host: Annotated[str | None, typer.Option(help="Bind address (0.0.0.0 = whole LAN)")] = None,
    port: Annotated[int | None, typer.Option()] = None,
    token: Annotated[
        str | None, typer.Option(envvar="MOTIONSENSE_TOKEN", help="Require this API token")
    ] = None,
    device: DeviceOpt = None,
    imgsz: ImgszOpt = None,
    objects: Annotated[bool | None, typer.Option("--objects/--no-objects")] = None,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run detection and serve the dashboard + API."""
    import uvicorn

    from motionsense.api.app import create_app
    from motionsense.service import MotionService

    _logging(verbose)
    settings = _settings(
        config,
        source=source,
        host=host,
        port=port,
        token=token,
        device=device,
        imgsz=imgsz,
        objects=objects,
    )
    if settings.api.host not in LOCAL_HOSTS and not settings.api.token:
        typer.secho(
            "Refusing to serve the camera on the network without a token: anyone on your "
            "network could watch it.\nAdd --token <a long random secret> "
            '(e.g. python -c "import secrets; print(secrets.token_urlsafe(24))").',
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    url = f"http://{'localhost' if settings.api.host in ('0.0.0.0', '127.0.0.1') else settings.api.host}:{settings.api.port}"  # nosec B104 - string comparison for the display URL, not a bind
    typer.secho(f"Dashboard: {url}   API docs: {url}/docs", fg=typer.colors.GREEN)
    if open_browser:
        webbrowser.open(url + (f"/?token={settings.api.token}" if settings.api.token else ""))
    service = MotionService(settings)
    uvicorn.run(
        create_app(service), host=settings.api.host, port=settings.api.port, log_level="warning"
    )


@app.command()
def bench(
    config: ConfigOpt = None,
    source: SourceOpt = None,
    seconds: Annotated[float, typer.Option(help="How long to measure")] = 15.0,
    device: DeviceOpt = None,
    imgsz: ImgszOpt = None,
    runtime: Annotated[str | None, typer.Option(help="openvino | torch")] = None,
) -> None:
    """Measure model speed (FPS) on this machine."""
    from motionsense.tools.bench import run_bench

    _logging(False)
    settings = _settings(config, source=source, device=device, imgsz=imgsz, runtime=runtime)
    r = run_bench(settings, seconds)
    typer.echo(f"device={r.device} runtime={settings.models.runtime} imgsz={settings.models.imgsz}")
    typer.echo(
        f"pose: {r.pose_ms} ms/frame -> {r.pose_fps} FPS over {r.frames} frames "
        f"(max people seen: {r.people_seen})"
    )
    if r.objects_ms is not None:
        typer.echo(f"objects: {r.objects_ms} ms/run (runs every {settings.objects.interval_s}s)")


@app.command()
def record(
    out: Annotated[Path, typer.Argument(help="Output .mp4 path")],
    seconds: Annotated[float, typer.Option()] = 15.0,
    source: SourceOpt = None,
    config: ConfigOpt = None,
) -> None:
    """Record a raw clip from the camera (to build labelled test sets)."""
    from motionsense.tools.record import record_clip

    s = _settings(config, source=source)
    typer.echo(
        f"Recording {seconds:.0f}s from {s.camera.source} -> {out} (starts after camera warm-up)"
    )
    n = record_clip(
        s.camera.source, out, seconds, s.camera.width, s.camera.height, min(s.camera.fps, 30)
    )
    typer.secho(f"Saved {n} frames to {out}", fg=typer.colors.GREEN)


@app.command("eval")
def evaluate(
    labels: Annotated[Path | None, typer.Option(help="CSV: video,has_fall,fall_time_s")] = None,
    synthetic: Annotated[
        bool, typer.Option(help="Evaluate on the built-in simulated scenarios")
    ] = False,
    noise: Annotated[float, typer.Option(help="Keypoint noise (px) for synthetic runs")] = 2.0,
    suspected: Annotated[bool, typer.Option(help="Count FALL_SUSPECTED as a detection")] = False,
    out: Annotated[Path, typer.Option(help="Report directory")] = Path("reports/latest"),
    config: ConfigOpt = None,
) -> None:
    """Measure fall-detection precision / recall / latency on labelled clips."""
    from motionsense.tools.evaluate import (
        ClipLabel,
        any_detection_metrics,
        read_labels,
        run_evaluation,
        write_report,
    )

    if (labels is None) == (not synthetic):
        raise typer.BadParameter("use exactly one of --labels or --synthetic")
    settings = _settings(config)
    if synthetic:
        from motionsense.simulation.backend import (
            ScriptedObjectBackend,
            ScriptedPoseBackend,
            SyntheticSource,
        )
        from motionsense.simulation.scenarios import SCENARIOS

        items = [ClipLabel(n, s.has_fall, s.fall_time) for n, s in SCENARIOS.items() if n != "demo"]

        def make(label: ClipLabel) -> tuple[Any, Any, Any]:
            sc = SCENARIOS[label.name]
            return (
                SyntheticSource(sc, fps=15.0),
                ScriptedPoseBackend(sc, noise_px=noise, seed=7),
                ScriptedObjectBackend(sc),
            )

        title = f"Synthetic scenarios (noise {noise}px)"
    else:
        from motionsense.perception.yolo_objects import YoloObjectBackend
        from motionsense.perception.yolo_pose import YoloPoseBackend
        from motionsense.sources.opencv_source import VideoFileSource

        if labels is None:  # pragma: no cover - guarded by the check above
            raise typer.BadParameter("--labels is required")
        items = read_labels(labels)
        # Load (and compile) the models once; only the tracker is reset between clips.
        pose_backend = YoloPoseBackend(settings.models, settings.models_dir)
        obj_backend = (
            YoloObjectBackend(settings.models, settings.models_dir)
            if settings.objects.enabled
            else None
        )

        def make(label: ClipLabel) -> tuple[Any, Any, Any]:
            pose_backend.reset()
            return VideoFileSource(label.name), pose_backend, obj_backend

        title = f"Clips from {labels.parent.name}/{labels.name}"

    def progress(r: Any) -> None:
        mark = "OK " if r.detected == r.has_fall else "XX "
        typer.echo(
            f"{mark} {Path(r.name).name:28s} fall={r.has_fall!s:5s} detected={r.detected!s:5s} "
            f"latency={r.latency_s}"
        )

    metrics, results = run_evaluation(items, make, settings, suspected, progress)
    report = write_report(out, title, metrics, results)
    early = any_detection_metrics(results)
    typer.secho(
        f"confirmed: precision={metrics.precision} recall={metrics.recall} f1={metrics.f1} "
        f"mean_latency={metrics.mean_latency_s}s",
        fg=typer.colors.GREEN,
    )
    typer.secho(
        f"any detection: precision={early.precision} recall={early.recall} f1={early.f1}"
        f"  -> {report}",
        fg=typer.colors.GREEN,
    )


@app.command()
def models(
    config: ConfigOpt = None,
    device: DeviceOpt = None,
    imgsz: ImgszOpt = None,
) -> None:
    """Download and export the configured models (one-time; also happens on first serve)."""
    from motionsense.perception.yolo_objects import YoloObjectBackend
    from motionsense.perception.yolo_pose import YoloPoseBackend

    _logging(False)
    s = _settings(config, device=device, imgsz=imgsz)
    YoloPoseBackend(s.models, s.models_dir)
    YoloObjectBackend(s.models, s.models_dir)
    typer.secho(f"Models ready in {s.models_dir}", fg=typer.colors.GREEN)


@app.command()
def scenarios() -> None:
    """List built-in simulation scenarios (use with --source synthetic:<name>)."""
    from motionsense.simulation.scenarios import SCENARIOS

    for name, sc in SCENARIOS.items():
        typer.echo(f"{name:18s} {sc.duration:5.1f}s  fall={'yes' if sc.has_fall else 'no '}")


if __name__ == "__main__":  # pragma: no cover
    app()
