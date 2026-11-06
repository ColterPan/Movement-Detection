"""POST events as JSON to external URLs (Home Assistant, n8n, your own AI service...).

Delivery happens on a background thread so a slow receiver never stalls detection.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Sequence

import httpx

from motionsense.events.models import SEVERITY_RANK, Event, Severity

logger = logging.getLogger(__name__)

MAX_QUEUE = 500
RETRIES = 2
TIMEOUT_S = 5.0


class WebhookSink:
    def __init__(
        self,
        urls: Sequence[str],
        min_severity: Severity = Severity.WARNING,
        client: httpx.Client | None = None,
    ) -> None:
        for url in urls:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"webhook URL must start with http:// or https://: {url!r}")
        self._urls = list(urls)
        self._min_rank = SEVERITY_RANK[min_severity]
        self._client = client or httpx.Client(timeout=TIMEOUT_S)
        self._queue: queue.Queue[Event | None] = queue.Queue(maxsize=MAX_QUEUE)
        self.delivered = 0
        self.failed = 0
        self._thread = threading.Thread(target=self._worker, name="webhook-sink", daemon=True)
        self._thread.start()

    def __call__(self, event: Event) -> None:
        if SEVERITY_RANK[event.severity] < self._min_rank or not self._urls:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.failed += 1
            logger.warning("webhook queue full; dropping event %s", event.id)

    def _worker(self) -> None:
        while True:
            event = self._queue.get()
            try:
                if event is None:
                    return
                payload = event.to_dict()
                for url in self._urls:
                    self._deliver(url, payload)
            except Exception:  # never let one bad event kill the delivery thread
                self.failed += 1
                logger.exception("webhook delivery crashed")
            finally:
                self._queue.task_done()

    def _deliver(self, url: str, payload: dict) -> None:
        for attempt in range(RETRIES + 1):
            try:
                resp = self._client.post(url, json=payload)
                resp.raise_for_status()
                self.delivered += 1
                return
            except Exception as exc:  # httpx.HTTPError, InvalidURL, ...
                if attempt == RETRIES:
                    self.failed += 1
                    logger.warning("webhook %s failed after %d attempts: %s", url, attempt + 1, exc)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until queued events are delivered (used by tests and shutdown)."""
        done = threading.Event()

        def wait() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=wait, daemon=True).start()
        done.wait(timeout)

    def close(self) -> None:
        self.flush()
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            logger.warning("webhook queue still full at shutdown; abandoning pending events")
        self._thread.join(timeout=2.0)
        self._client.close()
