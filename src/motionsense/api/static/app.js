"use strict";
// MotionSense dashboard: consumes the same public API any other integration would use.

const ACTIVITIES = ["standing", "sitting", "lying", "walking", "running", "squatting", "push_up"];
const ALARM_TYPES = new Set(["fall_confirmed", "inactive_on_floor"]);
const INFO_TYPES = new Set(["activity_changed", "rep_completed", "person_entered", "person_left"]);
const MAX_EVENTS = 200;

const params = new URLSearchParams(location.search);
const token = params.get("token");
const withToken = (url) => (token ? `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}` : url);

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

const seenEvents = new Set();
let audioCtx = null;
let alarmTimer = null;
let reconnectDelay = 500;

// ------------------------------------------------------------------ video
function startVideo() {
  const img = $("video");
  img.onload = () => $("video-empty").classList.add("hidden");
  img.onerror = () => {
    $("video-empty").classList.remove("hidden");
    setTimeout(() => { img.src = withToken(`/stream.mjpg?t=${Date.now()}`); }, 2000);
  };
  img.src = withToken("/stream.mjpg");
}

// ------------------------------------------------------------------ people
function barRow(name, value, isTop) {
  const row = el("div", `bar-row${isTop ? " top" : ""}`);
  row.append(el("span", "bar-label", name.replace("_", " ")));
  const bar = el("div", "bar");
  const fill = el("span");
  fill.style.width = `${Math.round(value * 100)}%`;
  bar.append(fill);
  row.append(bar, el("span", "bar-val", `${Math.round(value * 100)}%`));
  return row;
}

function riskColor(r) {
  if (r >= 0.6) return "var(--bad)";
  if (r >= 0.3) return "var(--warn)";
  return "var(--ok)";
}

function renderPeople(state) {
  const box = $("people");
  $("people-count").textContent = state.person_count;
  box.replaceChildren();
  if (!state.persons.length) {
    box.append(el("p", "muted", "Nobody in view."));
    return;
  }
  for (const p of state.persons) {
    const alarm = p.fall_state === "fallen" || p.fall_state === "inactive";
    const card = el("div", `person${alarm ? " alarm" : ""}`);
    const head = el("div", "person-head");
    head.append(
      el("span", "person-title", `#${p.track_id} · ${p.top_activity.replace("_", " ")}`),
      el("span", "person-meta", `${p.quality} view${p.support ? " · on " + p.support : ""}`),
    );
    card.append(head);
    const sorted = ACTIVITIES.map((a) => [a, p.activities[a] ?? 0]).sort((a, b) => b[1] - a[1]);
    for (const [name, v] of sorted) card.append(barRow(name, v, name === p.top_activity));

    const risk = el("div", "risk");
    const label = el("div", "risk-label");
    label.append(
      el("span", "", `Fall risk${p.fall_state !== "monitoring" ? " · " + p.fall_state.toUpperCase() : ""}`),
      el("strong", "", `${Math.round(p.fall_risk * 100)}%`),
    );
    const bar = el("div", "bar");
    const fill = el("span");
    fill.style.width = `${Math.round(p.fall_risk * 100)}%`;
    fill.style.background = riskColor(p.fall_risk);
    bar.append(fill);
    risk.append(label, bar);
    card.append(risk);

    const tags = el("div", "tags");
    if (p.attributes.looking_around) tags.append(el("span", "tag", "looking around"));
    if (p.attributes.head_direction && p.attributes.head_direction !== "unknown")
      tags.append(el("span", "tag", `head: ${p.attributes.head_direction}`));
    for (const [k, v] of Object.entries(p.reps || {})) if (v) tags.append(el("span", "tag", `${k.replace("_", " ")} ×${v}`));
    if (tags.childElementCount) card.append(tags);
    box.append(card);
  }
}

function renderObjects(state) {
  const box = $("objects");
  box.replaceChildren();
  const counts = {};
  for (const o of state.objects) counts[o.label] = (counts[o.label] || 0) + 1;
  const labels = Object.keys(counts).sort();
  if (!labels.length) { box.append(el("span", "muted", "None detected.")); return; }
  for (const l of labels) box.append(el("span", "chip", counts[l] > 1 ? `${l} ×${counts[l]}` : l));
}

function renderStatus(status, connected) {
  const conn = $("conn");
  conn.textContent = connected ? (status && status.error ? "degraded" : "live") : "disconnected";
  conn.className = `pill ${connected && !(status && status.error) ? "pill-on" : "pill-off"}`;
  if (status) {
    $("fps").textContent = `${status.fps.toFixed(1)} FPS`;
    $("source").textContent = status.source;
    $("source").title = status.error || "";
  }
}

// ------------------------------------------------------------------ events
function addEvent(e, live) {
  if (seenEvents.has(e.id)) return;
  seenEvents.add(e.id);
  const li = el("li");
  li.dataset.info = INFO_TYPES.has(e.type) ? "1" : "0";
  if (li.dataset.info === "1" && !$("opt-info").checked) li.classList.add("hidden");
  const t = new Date(e.ts * 1000);
  const time = el("time", "", isNaN(t) || e.ts < 1e9 ? `${e.ts.toFixed(1)}s` : t.toLocaleTimeString());
  const msg = el("span", `sev-${e.severity}`, e.message || e.type);
  if (e.snapshots_url) {
    const a = el("a", "", "snapshots");
    a.href = "#";
    a.addEventListener("click", (ev) => { ev.preventDefault(); openViewer(e.id); });
    msg.append(a);
  }
  li.append(time, msg);
  const list = $("events");
  list.prepend(li);
  while (list.childElementCount > MAX_EVENTS) list.lastElementChild.remove();
  // Events replayed on (re)connect are history; current alarms are rebuilt from live state.
  if (live && ALARM_TYPES.has(e.type)) raiseAlarm(e.track_id, e.type, e.message, true);
  if (live && e.type === "system_error") raiseAlarm("system", e.type, e.message, false);
  if (live && e.type === "fall_recovered") clearTrackAlarm(e.track_id);
  if (live && e.snapshots_url) setTimeout(loadIncidents, 500);
}

// ------------------------------------------------------------------ alarm
function beep() {
  try {
    audioCtx = audioCtx || new AudioContext();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.frequency.value = 880;
    gain.gain.value = 0.15;
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.35);
  } catch (_) { /* audio blocked until user interaction */ }
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  speechSynthesis.speak(u);
}

// One alarm per person (track id), so one person getting up never hides another's fall.
const alarms = new Map(); // key -> {type, message}
const acknowledged = new Set();

function raiseAlarm(key, type, message, isNew) {
  alarms.set(key, { type, message });
  if (isNew) {
    acknowledged.delete(key); // a new incident (or an escalation) always re-alerts
    if ($("opt-voice").checked && type !== "system_error")
      speak("Are you okay? A fall was detected. Please say something or stand up.");
  }
  renderAlarm();
}

function clearTrackAlarm(key) {
  alarms.delete(key);
  acknowledged.delete(key);
  renderAlarm();
}

function syncAlarmsWithState(state) {
  const present = new Map(state.persons.map((p) => [p.track_id, p]));
  for (const p of state.persons) {
    const down = p.fall_state === "fallen" || p.fall_state === "inactive";
    if (down && !alarms.has(p.track_id)) {
      const type = p.fall_state === "inactive" ? "inactive_on_floor" : "fall_confirmed";
      raiseAlarm(p.track_id, type, `Person #${p.track_id} is down`, false);
    }
  }
  for (const key of [...alarms.keys()]) {
    const p = present.get(key);
    if (p && p.fall_state === "monitoring") clearTrackAlarm(key);
  }
}

function renderAlarm() {
  const active = [...alarms.entries()].filter(([key]) => !acknowledged.has(key));
  if (!active.length) {
    $("alert").classList.add("hidden");
    clearInterval(alarmTimer);
    alarmTimer = null;
    return;
  }
  const [, latest] = active[active.length - 1];
  const titles = { inactive_on_floor: "PERSON NOT GETTING UP", system_error: "MONITORING PROBLEM" };
  $("alert-title").textContent = titles[latest.type] || "FALL DETECTED";
  $("alert-msg").textContent = active.length > 1 ? `${latest.message} (+${active.length - 1} more)` : latest.message;
  $("alert").classList.remove("hidden");
  if ($("opt-sound").checked && alarmTimer === null) {
    beep();
    alarmTimer = setInterval(beep, 1500);
  }
}

function acknowledgeAll() {
  for (const key of alarms.keys()) acknowledged.add(key);
  renderAlarm();
}
// ------------------------------------------------------------------ websocket
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(withToken(`${proto}://${location.host}/ws`));
  let caughtUp = false; // the server sends recent history first, then the first state message
  ws.onopen = () => { reconnectDelay = 500; renderStatus(null, true); };
  ws.onmessage = (msg) => {
    const data = JSON.parse(msg.data);
    if (data.type === "event") addEvent(data.event, caughtUp);
    else if (data.type === "state") {
      caughtUp = true;
      renderStatus(data.status, true);
      renderPeople(data.state);
      renderObjects(data.state);
      syncAlarmsWithState(data.state);
    }
  };
  ws.onclose = () => {
    renderStatus(null, false);
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 8000);
  };
}

$("alert-ack").addEventListener("click", acknowledgeAll);
$("opt-info").addEventListener("change", (ev) => {
  for (const li of $("events").children) if (li.dataset.info === "1") li.classList.toggle("hidden", !ev.target.checked);
});
// ------------------------------------------------------------------ incidents & snapshot viewer
const viewer = { frames: [], index: 0 };

async function api(path) {
  const res = await fetch(withToken(path));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

const TYPE_TITLES = {
  fall_confirmed: "Fall detected",
  fall_suspected: "Possible fall",
  inactive_on_floor: "Not getting up",
};

function timeLabel(ts) {
  const t = new Date(ts * 1000);
  return ts > 1e9 ? t.toLocaleString() : `${ts.toFixed(1)} s (video time)`;
}

async function loadIncidents() {
  const box = $("incidents");
  try {
    const data = await api("/api/v1/incidents?limit=20");
    box.replaceChildren();
    if (!data.incidents.length) {
      box.append(el("li", "muted", "No falls recorded yet."));
      return;
    }
    for (const inc of data.incidents) {
      const li = el("li");
      const text = el("div");
      const title = el("div", `inc-title sev-${inc.severity}-text`,
        `${TYPE_TITLES[inc.type] || inc.type} · person #${inc.track_id ?? "?"}`);
      text.append(title, el("div", "inc-time", timeLabel(inc.ts)));
      const btn = el("button", "", "View");
      btn.type = "button";
      btn.addEventListener("click", () => openViewer(inc.id));
      li.append(text, btn);
      box.append(li);
    }
  } catch (err) {
    box.replaceChildren(el("li", "muted", `Could not load incidents (${err.message}).`));
  }
}

async function openViewer(eventId) {
  $("viewer").classList.remove("hidden");
  $("viewer-title").textContent = "Loading…";
  $("viewer-sub").textContent = "";
  $("viewer-strip").replaceChildren();
  $("viewer-img").removeAttribute("src");
  $("viewer-caption").textContent = "";
  try {
    const data = await api(`/api/v1/events/${encodeURIComponent(eventId)}/snapshots`);
    const ev = data.event || {};
    $("viewer-title").textContent = ev.message || TYPE_TITLES[ev.type] || "Incident";
    const conf = ev.confidence != null ? ` · confidence ${Math.round(ev.confidence * 100)}%` : "";
    $("viewer-sub").textContent = ev.ts ? `${timeLabel(ev.ts)}${conf}` : "";
    viewer.frames = data.frames;
    $("viewer-empty").classList.toggle("hidden", data.frames.length > 0);
    const strip = $("viewer-strip");
    data.frames.forEach((f, i) => {
      const b = el("button", f.phase);
      b.type = "button";
      b.title = `${f.offset_s >= 0 ? "+" : ""}${f.offset_s.toFixed(1)} s`;
      const img = el("img");
      img.loading = "lazy";
      img.alt = `${f.phase} frame ${b.title}`;
      img.src = withToken(f.url);
      b.append(img);
      b.addEventListener("click", () => showFrame(i));
      strip.append(b);
    });
    // Start at the moment of the event (first "after" frame), or the last frame we have.
    const first = data.frames.findIndex((f) => f.phase === "after");
    if (data.frames.length) showFrame(first >= 0 ? first : data.frames.length - 1);
  } catch (err) {
    $("viewer-title").textContent = `Could not load snapshots (${err.message})`;
  }
}

function showFrame(i) {
  if (!viewer.frames.length) return;
  viewer.index = (i + viewer.frames.length) % viewer.frames.length;
  const f = viewer.frames[viewer.index];
  $("viewer-img").src = withToken($("viewer-raw").checked ? f.raw_url : f.url);
  const when = f.offset_s >= 0 ? `${f.offset_s.toFixed(1)} s after` : `${(-f.offset_s).toFixed(1)} s before`;
  $("viewer-caption").textContent = `Frame ${viewer.index + 1} of ${viewer.frames.length} · ${when} the alert`;
  [...$("viewer-strip").children].forEach((b, j) => b.classList.toggle("active", j === viewer.index));
}

function closeViewer() { $("viewer").classList.add("hidden"); }

$("viewer-close").addEventListener("click", closeViewer);
$("viewer").addEventListener("click", (ev) => { if (ev.target === $("viewer")) closeViewer(); });
$("viewer-prev").addEventListener("click", () => showFrame(viewer.index - 1));
$("viewer-next").addEventListener("click", () => showFrame(viewer.index + 1));
$("viewer-raw").addEventListener("change", () => showFrame(viewer.index));
$("incidents-refresh").addEventListener("click", loadIncidents);
document.addEventListener("keydown", (ev) => {
  if ($("viewer").classList.contains("hidden")) return;
  if (ev.key === "Escape") closeViewer();
  if (ev.key === "ArrowLeft") showFrame(viewer.index - 1);
  if (ev.key === "ArrowRight") showFrame(viewer.index + 1);
});

startVideo();
connect();
loadIncidents();
// Deep link: /?incident=<event id> opens that incident directly (e.g. from a notification).
if (params.get("incident")) openViewer(params.get("incident"));
