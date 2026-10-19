"use strict";

// Execute the shipped monitor script with a minimal DOM and deterministic clock.
// This verifies UI behavior, not native browser rendering or camera acquisition.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "static/index.html"), "utf8");
const source = fs.readFileSync(path.join(root, "static/app.js"), "utf8");
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
assert.equal(new Set(ids).size, ids.length, "DOM IDs must be unique");
const literalReferences = [...source.matchAll(/\$\("([^"$]+)"\)/g)].map((match) => match[1]);
assert.deepEqual([...new Set(literalReferences)].filter((id) => !ids.includes(id)), []);

class Element {
  constructor(id = "") {
    Object.assign(this, { id, textContent: "", dataset: {}, attributes: {}, children: [],
      hidden: false, disabled: false, value: "", checked: false, listeners: {}, naturalWidth: 0 });
    this.classList = { add() {}, toggle() {} };
  }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name]; }
  removeAttribute(name) { delete this.attributes[name]; }
  hasAttribute(name) { return name in this.attributes; }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); }
  querySelector(selector) {
    this.queries ??= {};
    return this.queries[selector] ??= new Element();
  }
}

const elements = Object.fromEntries(ids.map((id) => [id, new Element(id)]));
const cards = Object.fromEntries(["rgb", "depth", "rgbd", "pose"].map((name) => [name, new Element(name)]));
let now = 1000;
const calls = [];
let imuResponse = { imu: { state: "waiting", capture_epoch: 1 }, samples: [], latest_sequence: 0 };
const status = {
  camera: { state: "streaming", active: true, fps: 30, frame_id: 4, frame_age_ms: 10, device: {} },
  pose: { state: "ready", model_id: "mediapipe", fps: 10, frame_age_ms: 50,
    visible_keypoints: 21, valid_depth_keypoints: 17 },
  settings: { pose_model: "mediapipe", pose_enabled: true, confidence: 0.3, depth_min: 0.3, depth_max: 4 },
  recording: { state: "idle" },
};
const context = {
  console, URL, AbortController, performance: { now: () => now },
  setTimeout: () => 1, clearTimeout() {}, setInterval() {},
  document: {
    hidden: false, activeElement: null,
    getElementById(id) { assert(elements[id], `Missing DOM node ${id}`); return elements[id]; },
    querySelector(selector) {
      const match = selector.match(/data-stream="(.+)"/);
      return match ? cards[match[1]] : new Element();
    },
    querySelectorAll: () => [], createElement: () => new Element(),
    createTextNode: (text) => ({ textContent: text }), addEventListener() {},
  },
  window: { location: { origin: "http://127.0.0.1:8766" }, addEventListener() {} },
  async fetch(url, options) {
    calls.push({ url, options });
    return { ok: true, json: async () => url.startsWith("/api/imu") ? typeof imuResponse === "function" ? imuResponse(url) : imuResponse : status };
  },
};
context.globalThis = context;
vm.createContext(context);
const exported = source.replace(/\}\)\(\);\s*$/, "globalThis.uiTest = {renderStatus, syncMotionStatus, acceptMotionSamples, renderSensorData, renderRecording, syncRecording, pollMotion, state, motion};})();");
assert.notEqual(exported, source, "Test hook must be injected into the actual script");
vm.runInContext(exported, context);

async function main() {
  await new Promise((resolve) => setImmediate(resolve));
  const ui = context.uiTest;
  assert.equal(elements["sensor-rgb-state"].textContent, "수신 중");
  assert.equal(elements["sensor-depth-hz"].textContent, "30.0 Hz");
  assert.equal(elements["sensor-pose-detail"].textContent, "21점 · 유효 깊이 17점");
  now += 1200;
  ui.renderSensorData();
  assert.equal(elements["sensor-rgb-state"].textContent, "수신 중", "Normal 1.2-second status polling must not make RGB stale");
  assert.equal(elements["sensor-depth-state"].textContent, "수신 중", "Normal status polling must not make Depth stale");
  now -= 1200;
  const imu = { state: "streaming", available: true, capture_epoch: 1,
    accel: { x: 1, y: 2, z: 9.8, hz: 63, age_ms: 2, count: 2 },
    gyro: { x: 0.1, y: 0.2, z: 0.3, hz: 200, age_ms: 1, count: 2 } };
  const samples = [
    { sequence: 1, stream: "accel", x: 0, y: 2, z: 9.8, received_monotonic: 50, capture_epoch: 1 },
    { sequence: 2, stream: "gyro", x: 0, y: 0.2, z: 0.3, received_monotonic: 50, capture_epoch: 1 },
    { sequence: 3, stream: "accel", x: 1, y: 2, z: 9.8, received_monotonic: 50.1, capture_epoch: 1 },
    { sequence: 4, stream: "gyro", x: 0.1, y: 0.2, z: 0.3, received_monotonic: 50.1, capture_epoch: 1 },
  ];
  ui.acceptMotionSamples({ imu, samples, latest_sequence: 4 }, now);
  ui.renderSensorData();
  assert.equal(elements["imu-accel-x"].textContent, "1.000");
  const graphPath = elements["imu-accel-x-path"].attributes.d;
  assert(graphPath.includes("L"));
  ui.acceptMotionSamples({ imu, samples, latest_sequence: 4 }, now);
  ui.renderSensorData();
  assert.equal(ui.motion.histories.accel.length, 2, "Duplicate sequences cannot create graph samples");
  assert.equal(elements["imu-accel-x-path"].attributes.d, graphPath);
  now += 2600;
  ui.renderSensorData();
  assert.equal(elements["sensor-gyro-state"].textContent, "지연");
  assert.equal(elements["sensor-rgb-state"].textContent, "지연");
  ui.syncMotionStatus({ ...imu, stream_stale: { gyro: true } }, now);
  ui.renderSensorData();
  assert.equal(elements["sensor-gyro-state"].textContent, "지연", "Server stale flag is authoritative");
  ui.syncMotionStatus({ state: "unavailable", available: false, reason: "Motion Module을 찾지 못했습니다", capture_epoch: 1 }, now);
  ui.renderSensorData();
  assert.equal(elements["sensor-gyro-state"].textContent, "미수신");
  assert.equal(elements["imu-gyro-x"].textContent, "—");
  assert.equal(elements["imu-reason"].textContent, "Motion Module을 찾지 못했습니다");
  now += 1;
  ui.acceptMotionSamples({ imu: { ...imu, capture_epoch: 2 },
    samples: [{ ...samples[0], sequence: 1, capture_epoch: 2, received_monotonic: 60 }],
    latest_sequence: 1, truncated: true }, now);
  ui.renderSensorData();
  assert.equal(ui.motion.histories.accel.length, 1);
  assert.equal(ui.motion.histories.gyro.length, 0);
  assert.equal(ui.motion.cursor, 1);
  assert(elements["imu-history-note"].textContent.includes("일부 생략"));
  ui.acceptMotionSamples({ imu, samples, latest_sequence: 4 }, now - 1);
  assert.equal(ui.motion.epoch, 2, "Old response cannot restore an old capture");
  now += 1;
  ui.motion.cursor = 200000;
  imuResponse = (url) => ({ imu: { ...imu, capture_epoch: 2 }, latest_id: 1,
    latest_sequence: url.endsWith("after=0") ? 1 : 200000,
    samples: url.endsWith("after=0") ? [{ ...samples[0], sequence: 1, capture_epoch: 2, received_monotonic: 1 }] : [] });
  const beforeRestart = calls.length;
  await ui.pollMotion();
  assert.equal(calls[beforeRestart].url, "/api/imu?after=200000");
  assert.equal(calls[beforeRestart + 1].url, "/api/imu?after=0", "Server restart must retry immediately from zero");
  assert.equal(ui.motion.cursor, 1, "Reused capture epoch cannot preserve a dead server's cursor");
  assert.equal(ui.motion.histories.accel.length, 1);
  assert.equal(ui.motion.histories.accel[0].time, 1);
  ui.syncRecording({ state: "countdown", remaining_s: 5, dataset: { state: "countdown", rows_total: 0, rows_written: {} } }, now);
  now += 6000;
  ui.renderRecording();
  assert.equal(elements["recording-badge-label"].textContent, "준비", "Countdown completion needs server confirmation");
  ui.syncRecording({ id: "test", state: "completed", duration_s: 5, frames: { combined: 75 },
    dataset: { state: "completed", rows_total: 3000, rows_written: { camera: 100, imu_accel: 700, imu_gyro: 2000, pose_frame: 50, pose_joint: 150 }, dropped_total: 2 },
    files: { combined: "/recordings/test/combined.mp4", imu: "/recordings/test/imu.mp4", dataset_csv: "/recordings/test/dataset.csv", bad: "https://bad.example/file.csv" } }, now);
  const links = elements["recording-file-links"].children;
  assert.equal(links.length, 3);
  assert(links[0].textContent.startsWith("데이터셋 CSV"));
  assert(links[1].textContent.startsWith("4분할 + IMU"));
  assert(links[2].textContent.startsWith("IMU 그래프 영상"));
  assert(elements["recording-dataset"].textContent.includes("3,000행"));
  assert(elements["recording-dataset"].title.includes("imu_gyro: 2,000행"));
  assert(elements["recording-dropped"].textContent.includes("CSV 누락 2샘플"));
  ui.syncRecording({ id: "restored", state: "completed", imu_visualization: { enabled: false },
    files: { combined: "/recordings/restored/combined.mp4", dataset_csv: "/recordings/restored/dataset.csv" } }, now);
  const restoredLinks = elements["recording-file-links"].children;
  assert.equal(restoredLinks.length, 2);
  assert(restoredLinks[0].textContent.startsWith("데이터셋 CSV"));
  assert(restoredLinks[1].textContent.startsWith("4분할 영상"));
  assert(!restoredLinks.some((link) => link.textContent.includes("IMU 그래프")));
  ui.syncRecording({ state: "recording", dataset: { rows_written: { camera: 3, imu_accel: 4 }, dropped_samples: 1 } }, now);
  assert(elements["recording-dataset"].textContent.includes("7행"), "Dictionary row counts have a fallback sum");
  context.document.hidden = true;
  const requestCount = calls.length;
  await ui.pollMotion();
  assert.equal(calls.length, requestCount, "Hidden tabs must pause IMU polling");
  console.log("IMU UI: DOM, reception states, actual sample plots, deduplication, capture/server restart reset, stale/unavailable sensors, CSV row dictionaries/downloads, countdown, and hidden-tab polling passed.");
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
