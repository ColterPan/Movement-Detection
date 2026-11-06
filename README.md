# MotionSense

**Real-time activity recognition and fall detection from any camera, with an open event API so other apps and AI agents can plug in.**

MotionSense watches a webcam, phone camera, IP camera or video file and, for every person in view, reports live percentages for **standing, sitting, lying, walking, running, squatting and push-ups** (with rep counting), whether they are **looking around**, which **objects** are around them, and a continuous **fall-risk score**. When someone falls it raises a `fall_confirmed` event within about 2–3 seconds, saves before/after snapshots, and pushes the event to the dashboard, a WebSocket, a JSON log and any webhook you configure.

> ⚠️ **Prototype, not a medical or safety device.** A single camera has blind spots and the detector can miss falls or raise false alarms. Do not rely on it as the only protection for a vulnerable person.

![Dashboard](docs/dashboard.png)

---

## Contents
- [Quick start](#quick-start)
- [Using a phone or IP camera](#using-a-phone-or-ip-camera)
- [How it works](#how-it-works)
- [Integrating with other systems (API, WebSocket, webhooks, Python)](#integrating-with-other-systems)
- [Configuration](#configuration)
- [Testing and measured accuracy](#testing-and-measured-accuracy)
- [Roadmap](#roadmap)
- [License and credits](#license-and-credits)

---

## Quick start

Requirements: Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). Works on Windows, macOS and Linux. On Intel laptops the models run on the integrated GPU through OpenVINO automatically.

```bash
git clone https://github.com/<you>/motionsense.git
cd motionsense

# Windows tip: if the folder is inside OneDrive, keep the virtual environment outside it
#   PowerShell:  $env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\motionsense"
uv sync --extra yolo

# Try it without a camera: a simulated person walks, sits, squats, does push-ups and falls
uv run motionsense serve --source synthetic:demo --open

# Use your webcam (models download and convert automatically on first run, ~1 minute)
uv run motionsense serve --source 0 --open
```

Open **http://localhost:8000** for the dashboard and **http://localhost:8000/docs** for the interactive API docs.

Other commands:

| Command | What it does |
|---|---|
| `motionsense serve` | Run detection and serve the dashboard and API |
| `motionsense bench --source 0` | Measure model speed (FPS) on your machine |
| `motionsense record clip.mp4 --seconds 20` | Record a raw clip to build your own test set |
| `motionsense eval --labels labels.csv` | Measure precision, recall and latency on labelled clips |
| `motionsense eval --synthetic` | Run the built-in simulated regression scenarios |
| `motionsense scenarios` | List simulation scenarios (`--source synthetic:<name>`) |
| `motionsense models` | Pre-download and export the models |

## Using a phone or IP camera

Any stream OpenCV can read works as `--source`:

- **Android**: install *IP Webcam*, start the server, then `--source http://<phone-ip>:8080/video`
- **iPhone / Android**: *DroidCam* or *iVCam* (exposes the phone as a webcam index, or as an HTTP URL)
- **IP / CCTV cameras**: `--source rtsp://user:pass@192.168.1.20:554/stream1`
- **Video file**: `--source path/to/video.mp4`

To open the dashboard from your phone, bind to your LAN **with a token**:
```bash
uv run motionsense serve --source 0 --host 0.0.0.0 --token "choose-a-long-secret"
# then on the phone: http://<pc-ip>:8000/?token=choose-a-long-secret
```

## How it works

```mermaid
flowchart LR
  S[Frame source<br/>webcam · RTSP · HTTP · file · simulator] --> P[Pose backend<br/>YOLO26-pose + ByteTrack]
  S --> O[Object backend<br/>YOLO26, ~2 Hz]
  P --> F[Features<br/>angles · torso tilt · speeds<br/>normalised by torso length]
  F --> A[Activity classifier<br/>fuzzy rules → %]
  F --> D[Fall detector<br/>state machine]
  O --> C[Context<br/>bed / couch / chair under hips]
  C --> A & D
  A & D --> B[(Event bus)]
  B --> W[WebSocket / dashboard]
  B --> J[JSONL log]
  B --> N[Snapshots]
  B --> H[Webhooks]
```

The design is **hexagonal**: the core pipeline is plain Python with no web dependency, and every stage sits behind a small interface (`PoseBackend`, `ObjectBackend`, `ActivityClassifier`, `FrameSource`). To add MediaPipe, RTMPose or a trained model, write one class; nothing else changes.

### 1. Perception
[Ultralytics YOLO26s-pose](https://docs.ultralytics.com/) finds every person and their 17 COCO keypoints, and ByteTrack gives each person a stable ID. Models are exported to **OpenVINO** once, and `device: auto` picks the Intel iGPU when one is present (on an Iris Xe laptop: `yolo26s-pose` about 27 ms per frame, roughly 37 FPS; `yolo26n-pose` about 20 ms, against about 235 ms on the CPU). On machines without a supported GPU, set `models.pose_model: yolo26n-pose`. A YOLO26n object detector runs twice a second for context (bed, couch, chair, TV and so on); furniture is remembered while it is hidden behind a person.

### 2. Features (camera-distance invariant)
For each person: torso tilt from vertical, hip, knee and elbow angles, hip height, hip speed and drop, **leg alternation** (walking and running move the legs in anti-phase, squats move them together), head yaw, and the range of knee and elbow angles over 2 s. Every distance is measured in **torso lengths**, so features mean the same thing whether the person is 1 m or 6 m from the camera.

### 3. Activity percentages
A transparent fuzzy-logic classifier ([rule_classifier.py](src/motionsense/activity/rule_classifier.py)) scores every activity with smooth thresholds, normalises the scores into percentages and smooths them over time. Scene context matters: hips over a chair raise the sitting score, and squatting in front of a chair reads as sitting down. When only the upper body is visible (a laptop webcam at a desk) the view is marked `partial` and the classifier is deliberately modest. Push-ups and squats are counted with hysteresis on the elbow and knee angles.

### 4. Fall detection
A per-person state machine ([fall/detector.py](src/motionsense/fall/detector.py)):

```
MONITORING ──impact──▶ IMPACT ──still down after 2 s──▶ FALLEN ──down 30 s──▶ INACTIVE
     ▲                    │                               │                    │
     └──── got back up ───┴──────────── got back up ──────┴──── got back up ───┘
```

- **Impact** requires all of: the hips drop by more than 0.7 torso lengths within 1 s, the torso rotates by more than 40° within 1.5 s, and the peak downward speed exceeds 1.3 torso lengths per second, starting from an upright posture. *Speed* is what separates a fall from lying down on purpose.
- **Confirmation**: the person is still down 2 s after the impact. Getting back up first produces `fall_recovered`, not an alarm.
- **Context**: landing on a bed or couch multiplies confidence by 0.35, so flopping onto the bed is ignored.
- **Unwitnessed falls**: someone lying still on the floor (not on furniture) for 30 s raises `inactive_on_floor`, even if the fall happened out of view.
- **Occlusion**: a person who vanishes mid-fall (behind a sofa, or out of the frame) still triggers a confirmation. If someone on the floor flickers out of detection and reappears under a new ID, the incident continues on the new track.
- **Recovery**: standing up, or climbing into a chair (the hips rise well above their lowest point), closes the incident with `fall_recovered`.
- **Health**: if the camera stops delivering frames or the model keeps failing, a critical `system_error` event is raised, so silence never means "all fine".

Every threshold lives in [config/default.yaml](config/default.yaml).

## Integrating with other systems

Everything the dashboard shows comes from the public API, so any other program can use it the same way.

### REST
| Endpoint | Description |
|---|---|
| `GET /api/v1/health` | Status, FPS, source, last event sequence number |
| `GET /api/v1/state?keypoints=true` | Current people (activity %, fall state and risk, reps, attributes) and objects |
| `GET /api/v1/events?since=<seq>&type=fall_confirmed` | Event history (poll with `since`) |
| `GET /api/v1/events/{id}` | One event |
| `GET /api/v1/events/{id}/snapshots` | Incident snapshot URLs (before and after frames) |
| `GET /api/v1/frame.jpg` | Latest annotated frame |
| `GET /stream.mjpg` | Live annotated MJPEG stream |
| `GET /api/v1/schema/event` | JSON Schema of the event payload |
| `WS /ws` | Push stream: `{"type":"state",...}` at 10 Hz and `{"type":"event",...}` |

If a token is configured, send `Authorization: Bearer <token>` or `?token=<token>`. Without a token the server only accepts requests addressed to `localhost`, and WebSocket connections from other websites are refused, so a web page you visit cannot spy on your camera. Stream passwords in `rtsp://user:pass@…` URLs are never shown in the API.

### Event schema (v1.0)
```json
{
  "schema_version": "1.0",
  "id": "4f0c…",
  "seq": 42,
  "type": "fall_confirmed",
  "severity": "critical",
  "ts": 1762400000.12,
  "iso_time": "2026-11-06T03:33:20.120000+00:00",
  "camera_id": "cam0",
  "track_id": 3,
  "confidence": 0.93,
  "message": "FALL DETECTED: person #3 is down (93%)",
  "details": {"reason": "stayed_down_after_impact", "down_s": 2.0, "impact_peak": 0.97, "support": null},
  "snapshots_url": "/api/v1/events/4f0c…/snapshots"
}
```
Event types: `person_entered`, `person_left`, `activity_changed`, `rep_completed`, `fall_suspected`, `fall_confirmed`, `fall_recovered`, `inactive_on_floor`, `system_error`.

Polling: `GET /api/v1/events` without `since` returns the latest events; pass `since=<last seq you saw>` (starting at `0`) to page forward without ever skipping an event.

### Webhooks
```yaml
sinks:
  webhooks: ["http://localhost:5678/webhook/fall"]   # n8n, Home Assistant, Zapier, your own service
  webhook_min_severity: warning
```

### As a Python library
```python
from motionsense import Pipeline, load_settings, EventType
from motionsense.perception.yolo_pose import YoloPoseBackend
from motionsense.sources.opencv_source import CameraSource

settings = load_settings("config/default.yaml")
pipeline = Pipeline(settings, YoloPoseBackend(settings.models, settings.models_dir))
pipeline.bus.subscribe(lambda e: print(e.type, e.message) if e.type == EventType.FALL_CONFIRMED else None)

for frame in CameraSource(0).frames():
    scene = pipeline.process(frame)   # SceneState: people, activities, fall risk, objects
```

## Configuration
Copy [config/default.yaml](config/default.yaml) and pass it with `--config`, or override any key with an environment variable such as `MOTIONSENSE_CAMERA__SOURCE=1` or `MOTIONSENSE_API__TOKEN=…`. Precedence: command-line flags > environment variables > YAML file > built-in defaults. Models, events and snapshots are stored in `%LOCALAPPDATA%\motionsense` on Windows and `~/.local/share/motionsense` elsewhere, never inside the repo.

## Testing and measured accuracy

```bash
uv run pytest --cov             # unit and integration tests (no camera or model download needed)
uv run motionsense eval --synthetic
```

The test suite drives the whole pipeline with a **procedural skeleton simulator** ([simulation/](src/motionsense/simulation/)). Scripted scenarios (falls, falling and getting up, falling out of view, stumbles, sitting, lying on a bed, lying down slowly, push-ups, squats, walking, running and looking around) run with random keypoint noise, so every behaviour is reproducible and checked in CI.

### Real-video benchmark: UR Fall Detection Dataset
70 real RGB clips (30 falls, 40 everyday activities such as sitting down, crouching, picking things up and getting down to the floor), 320×240, processed end to end with the default configuration (`yolo26s-pose` on OpenVINO):

| Metric | Confirmed falls | Any detection (early `fall_suspected` or confirmed) |
|---|---|---|
| Precision | **0.78** | 0.76 |
| Recall | **0.70** | **0.87** |
| F1 | **0.74** | 0.81 |
| Mean latency after fall onset | 1.5 s | — |

Full per-clip report: [reports/urfall/report.md](reports/urfall/report.md).

Honest caveats:
- The three impact thresholds (hip drop 0.7, speed 1.3, torso rotation 40°) and the choice of `yolo26s-pose` over `yolo26n-pose` were **tuned on this same dataset** (a 36-point grid with a broad, stable optimum; the first untuned run scored F1 0.50). Treat these numbers as optimistic until they are confirmed on your own recordings.
- Many UR Fall clips end only 1.2–1.7 s after the fall begins, which is shorter than the 2 s "still down" confirmation window. That is why *any detection* recall is higher than confirmed recall.
- Most false alarms are people quickly getting down to the floor near the camera (crawling, looking under the bed). At 320×240 these are hard to tell apart from falls without a learned temporal model (roadmap v2).

Reproduce:
```bash
uv run python scripts/prepare_urfall.py --out datasets/urfall
uv run motionsense eval --labels datasets/urfall/labels.csv --out reports/urfall
```

### Live checklist (in front of your own camera)
Stand, sit on a chair, walk, jog on the spot, squat (rep count), do push-ups (rep count), look around, lie on the couch (must **not** alert), then make a controlled "fall" onto a mattress (must alert within about 3 s).

## Roadmap
- **Phase 2: AI response layer.** On `fall_suspected`, send the snapshot sequence to a vision LLM (Claude) for a second opinion, speak "Are you okay?" (text-to-speech) and listen for an answer (local Whisper), then escalate to a phone notification (ntfy, Telegram or Pushover).
- **Activity model v2**: a small temporal network trained on your own recordings (`motionsense record`) plus public datasets.
- **Phase 3: live AI connectivity.** An MCP server (`get_scene`, `get_events`, `get_snapshot`) so AI agents can "see" the room, multi-camera support, and Docker images for a Raspberry Pi or home server.

## License and credits
AGPL-3.0, because the default perception backend is [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0). If you need a permissive license, implement the `PoseBackend` interface with MediaPipe or RTMPose.

The fall benchmark uses the **UR Fall Detection Dataset**: B. Kwolek and M. Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine, 2014. The dataset is not redistributed here.
