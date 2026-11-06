# MotionSense

**Real-time activity recognition and fall detection from any camera, with an open event API so other apps and AI agents can plug in.**

MotionSense watches a webcam, phone camera, IP camera or video file and, for every person in view, reports live percentages for **standing, sitting, lying, walking, running, squatting and push-ups** (with rep counting), whether they are **looking around**, which **objects** are around them, and a continuous **fall-risk score**. When someone falls it raises a `fall_confirmed` event within about 2–3 seconds, saves before/after snapshots, and pushes the event to the dashboard, a WebSocket, a JSON log and any webhook you configure.

> ⚠️ **Prototype, not a medical or safety device.** A single camera has blind spots and the detector can miss falls or raise false alarms. Do not rely on it as the only protection for a vulnerable person.

![Dashboard](docs/dashboard.png)

---

## Contents

- [Run it on Windows (easiest way)](#run-it-on-windows-easiest-way)
- [Set up the camera (the biggest factor for accuracy)](#set-up-the-camera-the-biggest-factor-for-accuracy)
- [Test it yourself: the live checklist](#test-it-yourself-the-live-checklist)
- [Record your own clips and measure accuracy](#record-your-own-clips-and-measure-accuracy)
- [Troubleshooting](#troubleshooting)
- [Command reference (Windows, macOS, Linux)](#command-reference-windows-macos-linux)
- [Using a phone or IP camera](#using-a-phone-or-ip-camera)
- [How it works](#how-it-works)
- [Integrating with other systems (API, WebSocket, webhooks, Python)](#integrating-with-other-systems)
- [Configuration](#configuration)
- [Testing and measured accuracy](#testing-and-measured-accuracy)
- [Roadmap](#roadmap)
- [Privacy and security](#privacy-and-security)
- [License and credits](#license-and-credits)

---

## Run it on Windows (easiest way)

**You need:** Windows 10/11, [Python 3.11–3.13](https://www.python.org/downloads/) (tick *"Add python.exe to PATH"* when installing) and a webcam. An Intel laptop with Iris Xe graphics runs at 30+ FPS. Other PCs work too, just slower.

**1. Open the project folder in a terminal.** In File Explorer, open the `Movement Detection` (or cloned `motionsense`) folder, click the address bar, type `cmd` and press Enter.

**2. Start the demo first (no camera needed).** This checks that everything installed correctly:

```bat
run.bat demo
```

The first run installs everything and downloads the AI models (about 2–5 minutes, once). Your browser then opens **http://localhost:8000**, where a simulated stick figure walks, sits, squats, does push-ups and falls. You should see the activity percentages change and a red **FALL DETECTED** alert near the end. Press **Ctrl+C** in the terminal to stop.

**3. Start it on your webcam:**

```bat
run.bat
```

Step back so the camera sees your **whole body**, head to feet. A spot 2–4 m away, with the camera at chest height or higher, works best.

**4. Everything `run.bat` can do:**

| Command | What it does |
|---|---|
| `run.bat` | Live webcam (camera 0) and the dashboard |
| `run.bat camera 1` | Use a different webcam (1, 2, …) |
| `run.bat camera http://192.168.1.50:8080/video` | Use a phone or IP camera ([how](#using-a-phone-or-ip-camera)) |
| `run.bat demo` | Simulated person, no camera needed |
| `run.bat video recordings\myclip.mp4` | Run on a recorded video |
| `run.bat record myclip` | Record 20 s from the webcam to `recordings\myclip.mp4` |
| `run.bat bench` | Measure speed (FPS) on your PC |
| `run.bat test` | Run all automated tests (about 30 s) |
| `run.bat eval` | Run the simulated accuracy check |
| `run.bat setup` | Reinstall or update everything |

You can also double-click `run.bat` in File Explorer to start the webcam.

> The Python environment goes to `%USERPROFILE%\.venvs\motion-detection`, and models, events and snapshots go to `%LOCALAPPDATA%\motionsense`. Neither is inside the project folder, so OneDrive does not try to sync gigabytes of files.

## Set up the camera (the biggest factor for accuracy)

Fall detection needs to **see the floor and the whole body**. A laptop webcam on a desk only sees people from the waist up, so a person who falls simply drops out of view.

```
   GOOD                                   POOR
   camera high in a corner,               laptop on a desk,
   tilted down at the room                at face height
        📷                                       
         ╲                                 🧍  📷  (sees head + shoulders only)
          ╲   🧍                          ─────────
   ────────╲─────── floor visible          floor not visible
```

- **Height:** about 1.5–2.5 m (a shelf, the top of a wardrobe, or a phone on a tripod or wall mount), tilted down so the floor in the middle of the room is in view.
- **Distance:** people should take up roughly ⅓–⅔ of the image height while standing. Too close is the most common problem.
- **Angle:** from the side or a corner, not straight down from the ceiling.
- **Light:** a lit room. In the dark, webcams drop to 5–10 FPS and the pose model misses people.
- An old **Android phone running IP Webcam** on a shelf makes an excellent fall camera: `run.bat camera http://<phone-ip>:8080/video`.

The dashboard tells you when the view is limited: the person card shows **partial view** whenever the legs or hips are cut off. Sitting at the laptop is fine for trying it out, but not for fall monitoring.
## Test it yourself: the live checklist

Start `run.bat`, put the dashboard next to you (or on your phone, see below), and work through this list. Each line says what you should see.

| # | Do this | Expected on the dashboard |
|---|---|---|
| 1 | Stand still, full body in view | **Standing** above 70%, fall risk near 0% |
| 2 | Walk across the room | **Walking** |
| 3 | Jog on the spot or run across | **Running** |
| 4 | Sit on a chair | **Sitting**; the card says "on chair" once the chair is detected |
| 5 | Do 5 squats | **Squatting**, with the squat counter going up |
| 6 | Do push-ups side-on to the camera | **Push up**, with the push-up counter going up |
| 7 | Turn your head left and right a few times | "looking around" tag |
| 8 | Lie down on a couch or bed | **Lying**, with **no** fall alert |
| 9 | Lie down slowly on the floor and stay still for 30 s | **Lying**, then a "not getting up" alert after 30 s (this is intended) |
| 10 | Fall **safely** onto a mattress or thick cushions on the floor, then stay down | Orange "Possible fall" within about 1 s, then a red **FALL DETECTED** alert about 2 s later |
| 11 | After step 10, stand up | "Person got back up"; the alert clears |

**Safety first for step 10:** only fall onto something soft, start from kneeling if you are unsure, and never do it alone if you have any health concerns.

Tips:
- Click **Acknowledge** to silence an alert. Tick **Voice check-in** to have the browser say *"Are you okay?"* on a confirmed fall.
- Tick **show activity changes** in the Events panel to see every activity change.
- Each alert has a **snapshots** link. It opens a viewer with the frames from 2 s before to 3 s after the alert, with the skeletons and IDs exactly as the detector saw them. Use ‹ › or the arrow keys to step through, and tick *raw camera frames* for the untouched images. The **Incidents** panel keeps past falls, even after a restart.
- Write down anything that looks wrong, such as the wrong activity, a missed fall or a false alarm. Then record a clip of it (next section) so it can be fixed and turned into a test.

## Record your own clips and measure accuracy

This is how you find out how well MotionSense works in **your** room, with **your** camera angle.

1. **Record clips** of 10–20 s each:
   ```bat
   run.bat record fall-01
   run.bat record fall-02
   run.bat record normal-sit-01
   run.bat record normal-lie-couch-01
   ```
   Recording starts after the camera warms up (about 2 s). Aim for at least 5 safe falls (onto a mattress) and 10 normal activities, especially ones that *look* like falls: sitting down hard, lying on the couch, picking something up off the floor.

2. **Label them.** Create `recordings\labels.csv`, with the second (from the start of the clip) at which each fall begins:
   ```csv
   video,has_fall,fall_time_s
   fall-01.mp4,1,6.5
   fall-02.mp4,1,8.0
   normal-sit-01.mp4,0,
   normal-lie-couch-01.mp4,0,
   ```
   Leave `fall_time_s` empty if you don't know it; you just won't get a latency number.

3. **Score them:**
   ```bat
   set UV_PROJECT_ENVIRONMENT=%USERPROFILE%\.venvs\motion-detection
   python -m uv run --extra yolo motionsense eval --labels recordings\labels.csv --out reports\my-room
   ```
   Open `reports\my-room\report.md` to see precision, recall, latency and the result for every clip.

4. **Watch a clip with the overlays** to see what went wrong: `run.bat video recordings\fall-01.mp4`.

Recordings stay on your PC; `recordings/` is in `.gitignore`.

## Troubleshooting

| Problem | Fix |
|---|---|
| The video is black or very dark | Turn on the lights. Open the laptop's camera privacy shutter (many laptops have a slider above the screen or a camera key on the keyboard). Check that the Windows **Camera** app shows a picture. |
| "cannot open camera" | Close other apps using the camera (Teams, Zoom, the Camera app). Check *Settings → Privacy & security → Camera* and allow desktop apps. Try `run.bat camera 1`. |
| Low FPS (under 10) | Webcams slow down in dim light, so add light. Run `run.bat bench`: if `device=cpu`, edit `config/default.yaml` and set `pose_model: yolo26n-pose` and `imgsz: 480`. |
| "Nobody in view" while you are there | The whole body (or at least head to hips) must be visible. Step back from the camera. |
| Always "sitting (partial view)" | The camera only sees your upper body. Raise it and step back (see [camera setup](#set-up-the-camera-the-biggest-factor-for-accuracy)). |
| Person number keeps changing (#3, #9, #14…) | The detector keeps losing you: improve the light and make sure you are fully in view. Short dropouts are already bridged. |
| Activities look wrong | Side or high-corner camera views work best. If only your upper body is visible (sitting at a laptop), the person card says **partial view** and the guess is deliberately cautious. |
| False fall alerts | Note what you were doing and record a clip. You can make detection stricter in `config/default.yaml` (raise `fall.drop_threshold` or `fall.velocity_threshold`). |
| Port 8000 already in use | `python -m uv run --extra yolo motionsense serve --source 0 --port 8010 --open` |
| `python` not found | Install Python from python.org with *"Add python.exe to PATH"* ticked, then open a new terminal. |

## Command reference (Windows, macOS, Linux)

`run.bat` is a shortcut for these commands. On macOS or Linux, or if you prefer the terminal, use [uv](https://docs.astral.sh/uv/) directly:

```bash
git clone https://github.com/<you>/motionsense.git
cd motionsense
uv sync --extra yolo                                    # install (once)
uv run --extra yolo motionsense serve --source synthetic:demo --open   # demo, no camera
uv run --extra yolo motionsense serve --source 0 --open                # webcam
uv run pytest                                           # tests
```

Open **http://localhost:8000** for the dashboard and **http://localhost:8000/docs** for the interactive API docs.

| Command | What it does |
|---|---|
| `motionsense serve` | Run detection and serve the dashboard and API |
| `motionsense bench --source 0` | Measure model speed (FPS) on your machine |
| `motionsense record clip.mp4 --seconds 20` | Record a raw clip to build your own test set |
| `motionsense eval --labels labels.csv` | Measure precision, recall and latency on labelled clips |
| `motionsense eval --synthetic` | Run the built-in simulated regression scenarios |
| `motionsense scenarios` | List simulation scenarios (`--source synthetic:<name>`) |
| `motionsense models` | Pre-download and export the models |

Add `--help` to any command to see all its options.

## Using a phone or IP camera

Any stream OpenCV can read works as `--source`:

- **Android**: install *IP Webcam*, tap *Start server*, then run `run.bat camera http://<phone-ip>:8080/video` (the app shows the address on screen)
- **iPhone / Android**: *DroidCam* or *iVCam* (exposes the phone as a webcam index, or as an HTTP URL)
- **IP / CCTV cameras**: `--source rtsp://user:pass@192.168.1.20:554/stream1`
- **Video file**: `--source path/to/video.mp4`

To open the dashboard from your phone, bind to your LAN **with a token**. The server refuses to listen on the network without one:

```bat
python -m uv run --extra yolo motionsense serve --source 0 --host 0.0.0.0 --token "choose-a-long-secret"
rem then on the phone (same Wi-Fi): http://<pc-ip>:8000/?token=choose-a-long-secret
rem find <pc-ip> with the "ipconfig" command (IPv4 Address)
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

| Endpoint                                               | Description                                                                    |
| ------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `GET /api/v1/health`                                 | Status, FPS, source, last event sequence number                                |
| `GET /api/v1/state?keypoints=true`                   | Current people (activity %, fall state and risk, reps, attributes) and objects |
| `GET /api/v1/events?since=<seq>&type=fall_confirmed` | Event history (poll with `since`)                                            |
| `GET /api/v1/events/{id}`                            | One event                                                                      |
| `GET /api/v1/events/{id}/snapshots`                  | Incident snapshot URLs (before and after frames)                               |
| `GET /api/v1/frame.jpg`                              | Latest annotated frame                                                         |
| `GET /stream.mjpg`                                   | Live annotated MJPEG stream                                                    |
| `GET /api/v1/schema/event`                           | JSON Schema of the event payload                                               |
| `WS /ws`                                             | Push stream: `{"type":"state",...}` at 10 Hz and `{"type":"event",...}`    |

If a token is configured (12+ characters), send `Authorization: Bearer <token>` or `?token=<token>`; without it `/api/v1/health` only reports version and status. Without a token the server only accepts requests addressed to `localhost`, and WebSocket connections from other websites are refused, so a web page you visit cannot spy on your camera. Stream passwords in `rtsp://user:pass@…` URLs are never shown in the API.

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

| Metric                        | Confirmed falls | Any detection (early `fall_suspected` or confirmed) |
| ----------------------------- | --------------- | ----------------------------------------------------- |
| Precision | **0.79** | 0.80 |
| Recall | **0.73** | **0.80** |
| F1 | **0.76** | 0.80 |
| Mean latency after fall onset | 1.6 s | — |

Full per-clip report: [reports/urfall/report.md](reports/urfall/report.md).

Honest caveats:

- The three impact thresholds (hip drop 0.7, speed 1.3, torso rotation 40°) and the choice of `yolo26s-pose` over `yolo26n-pose` were **tuned on this same dataset** (a 36-point grid with a broad, stable optimum; the first untuned run scored F1 0.50). Treat these numbers as optimistic until they are confirmed on your own recordings.
- Many UR Fall clips end only 1.2–1.7 s after the fall begins, which is shorter than the 2 s "still down" confirmation window. That is why *any detection* recall is higher than confirmed recall.
- Most false alarms are people quickly getting down to the floor near the camera (crawling, looking under the bed). At 320×240 these are hard to tell apart from falls without a learned temporal model (roadmap v2).

Reproduce:

```bash
uv run python scripts/prepare_urfall.py --out datasets/urfall
uv run --extra yolo motionsense eval --labels datasets/urfall/labels.csv --out reports/urfall
```

### Live checklist (in front of your own camera)

Stand, sit on a chair, walk, jog on the spot, squat (rep count), do push-ups (rep count), look around, lie on the couch (must **not** alert), then make a controlled "fall" onto a mattress (must alert within about 3 s).

## Roadmap

- **Phase 2: AI response layer.** On `fall_suspected`, send the snapshot sequence to a vision LLM (Claude) for a second opinion, speak "Are you okay?" (text-to-speech) and listen for an answer (local Whisper), then escalate to a phone notification (ntfy, Telegram or Pushover).
- **Activity model v2**: a small temporal network trained on your own recordings (`motionsense record`) plus public datasets.
- **Phase 3: live AI connectivity.** An MCP server (`get_scene`, `get_events`, `get_snapshot`) so AI agents can "see" the room, multi-camera support, and Docker images for a Raspberry Pi or home server.

## Privacy and security

Video is processed in memory and never uploaded. Only short incident snapshots and an event log are kept, on your own computer, and the server is reachable from this PC only unless you deliberately enable network access with a token. See [SECURITY.md](SECURITY.md) for exactly what is stored, the network protections, and how to report a vulnerability.

## License and credits

AGPL-3.0, because the default perception backend is [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0). If you need a permissive license, implement the `PoseBackend` interface with MediaPipe or RTMPose.

The fall benchmark uses the **UR Fall Detection Dataset**: B. Kwolek and M. Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine, 2014. The dataset is not redistributed here.
