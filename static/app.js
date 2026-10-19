"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const streams = ["rgb", "depth", "rgbd", "pose"];
  const state = { status: null, layout: "all", poseBackground: "rgb", busy: false, pendingCameraAction: null, online: false, requestInFlight: false, settingsReady: false, streamSerial: 0, toastTimer: null, streamRetry: {}, modelRequestPending: false, requestedModel: null, activeModel: null, roiDirty: false, roiSaving: false, recordingAnchor: null, recordingSupported: false, recordingRequestPending: false, recordingConfirmedAt: 0, recordingFilesKey: null };
  const cards = Object.fromEntries(streams.map((name) => [name, document.querySelector(`[data-stream="${name}"]`)]));
  const images = Object.fromEntries(streams.map((name) => [name, $(`stream-${name}`)]));
  const labels = { disconnected: "카메라 대기", connecting: "카메라 연결 중", streaming: "카메라 연결됨", error: "카메라 연결 오류", offline: "서버 연결 끊김" };
  const format = (value, digits = 1) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
  const safeError = (error) => String(error?.message || error || "알 수 없는 오류");
  const models = {
    yolo: { name: "YOLOv8n-pose", task: "몸 17점", feed: "YOLO POSE", count: "감지된 사람", unit: "명", note: "사람 영역 검출", help: "사람을 검출한 뒤 몸 17점을 추정합니다. 전신이나 상체가 보이는 영상에 적합합니다.", confidence: "사람 검출 기준", confidenceHelp: "사람 영역을 검출할 때 적용하는 기준입니다." },
    mediapipe: { name: "MediaPipe Hand", task: "손 · 손가락 21점", feed: "MEDIAPIPE HAND", count: "감지된 손", unit: "개", note: "손 전용 추정 결과", help: "손가락과 손목 21점을 추정합니다. 팔꿈치와 어깨는 출력하지 않습니다.", confidence: "관절 표시", confidenceHelp: "관절별 확률을 제공하지 않아 모든 추정점을 표시합니다. 좌우 손 분류 점수는 관절 정확도가 아닙니다." },
    rtmpose: { name: "RTMPose-m", task: "몸 17점 · ROI 추정", feed: "RTMPOSE", count: "ROI 추정", unit: "개", note: "검출기 없이 몸 관절 추정", help: "사람 검출 없이 지정한 영역에서 몸 17점을 추정합니다. 손가락 관절은 출력하지 않습니다.", confidence: "관절 표시 기준", confidenceHelp: "관절별 점수가 이 기준 이상인 점을 표시합니다. 추정 개수는 사람 검출 개수가 아닙니다." },
    egoforce: { name: "EgoForce HALO", task: "손 + 전완 · 2D", feed: "EGOFORCE HALO", count: "처리한 손 영역", unit: "개", note: "손 · 전완의 2D 관절", help: "손과 전완의 2D 관절을 추정합니다. 3D 메시나 힘 추정은 포함하지 않습니다.", confidence: "관절 표시", confidenceHelp: "관절별 확률을 제공하지 않아 모든 추정점을 표시합니다. 내부 가중치는 정확도 확률이 아닙니다." },
  };
  const roiIds = ["pose-roi-x1", "pose-roi-y1", "pose-roi-x2", "pose-roi-y2"];
  const motion = { cursor: 0, epoch: null, busy: false, nextPollAt: 0, anchor: null, requestStartedAt: 0, histories: { accel: [], gyro: [] }, lastSequence: { accel: -1, gyro: -1 }, lastSampleAt: 0, lastSampleObservedAt: 0, truncated: false, unavailable: false, error: "" };
  let sensorStatusObservedAt = 0;

  function selectedModel() { return state.requestedModel || state.status?.settings?.pose_model || "yolo"; }
  function poseIsCurrent() {
    const pose = state.status?.pose || {};
    const selected = selectedModel();
    return !state.modelRequestPending && !state.roiSaving && (!pose.model_id || pose.model_id === selected) && pose.state === "ready";
  }

  function syncModelControls() {
    const selected = selectedModel();
    const model = models[selected] || models.yolo;
    const pose = state.status?.pose || {};
    const enabled = state.status?.settings?.pose_enabled !== false;
    $("pose-model-select").value = selected;
    $("pose-model-select").disabled = !state.online || state.modelRequestPending;
    $("pose-model-help").textContent = model.help;
    $("pose-count-label").textContent = model.count;
    $("pose-count-unit").textContent = model.unit;
    $("pose-count-note").textContent = model.note;
    $("pose-view-description").textContent = model.task;
    $("pose-feed-label").textContent = model.feed;
    $("confidence-label").textContent = model.confidence;
    $("confidence-help").textContent = model.confidenceHelp;
    const usesConfidence = selected === "yolo" || selected === "rtmpose";
    $("confidence").disabled = !usesConfidence || !state.online || state.modelRequestPending;
    $("confidence-value").textContent = usesConfidence ? Number($("confidence").value).toFixed(2) : "모든 점";
    $("confidence-ends").hidden = !usesConfidence;
    $("confidence").classList.toggle("is-inapplicable", !usesConfidence);
    $("pose-roi-controls").hidden = selected !== "rtmpose" && selected !== "egoforce";
    $("pose-roi-help").textContent = selected === "egoforce" ? "손이 자동 검출되면 전완 영역을 지정하세요. 손을 찾지 못하면 손을 포함한 영역을 지정해 다시 시도할 수 있습니다." : "손목·팔꿈치 등 실제로 보이는 팔이 포함되도록 지정하세요. 선택하지 않으면 전체 영상을 사용합니다.";
    $("pose-roi-note").textContent = state.roiDirty ? "변경한 좌표를 적용하면 실시간 추론에 반영됩니다." : "원본 RGB 픽셀 좌표 · 화면 밖 관절은 실제 관측값이 아닙니다.";
    for (const id of [...roiIds, "pose-roi-apply", "pose-roi-clear"]) $(id).disabled = !state.online || state.roiSaving || state.modelRequestPending;
    if (!enabled) $("pose-card-note").textContent = "Pose 추론 꺼짐";
    else if (!poseIsCurrent()) $("pose-card-note").textContent = `${model.name} · ${pose.state === "error" ? "오류" : "준비 중"}`;
    else if (Number.isFinite(pose.visible_keypoints)) $("pose-card-note").textContent = `${pose.visible_keypoints}점 · 유효 깊이 ${Number.isFinite(pose.valid_depth_keypoints) ? pose.valid_depth_keypoints : "—"}점`;
    else $("pose-card-note").textContent = `${model.name} · 관절 깊이`;
    $("pose-card-note").title = pose.model_note || model.help;
  }

  function syncROI(roi) {
    if (state.roiDirty || state.roiSaving || roiIds.includes(document.activeElement?.id)) return;
    roiIds.forEach((id, index) => { $(id).value = Array.isArray(roi) ? roi[index] : ""; });
    $("pose-roi-summary").textContent = Array.isArray(roi) ? `${Math.round(roi[2] - roi[0])} × ${Math.round(roi[3] - roi[1])} px` : "전체 영상";
  }

  function recordingSeconds(value) {
    return typeof value === "number" && Number.isFinite(value) ? Math.max(0, value) : 0;
  }

  function recordingTime(value) {
    const seconds = Math.floor(recordingSeconds(value));
    const minutes = Math.floor(seconds / 60);
    const tail = String(seconds % 60).padStart(2, "0");
    return minutes >= 60 ? `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}:${tail}` : `${String(minutes).padStart(2, "0")}:${tail}`;
  }

  function recordingText(id, text) {
    if ($(id).textContent !== text) $(id).textContent = text;
  }

  function recordingFileLinks(recording) {
    const finished = ["completed", "cancelled", "error"].includes(recording.state);
    const files = recording.files || {};
    const key = JSON.stringify([recording.id, recording.state, files]);
    if (state.recordingFilesKey === key) return;
    state.recordingFilesKey = key;
    $("recording-result").hidden = !finished || !recording.id;
    if (!finished || !recording.id) return;
    const links = [];
    const priority = (name) => name === "dataset_csv" ? 0 : name === "combined" ? 1 : name === "imu" ? 2 : 3;
    const entries = Object.entries(files).sort(([a], [b]) => priority(a) - priority(b));
    for (const [name, item] of entries) {
      const value = typeof item === "string" ? item : item?.url;
      if (!value) continue;
      let url;
      try { url = new URL(value, window.location.origin); } catch { continue; }
      if (url.origin !== window.location.origin || !/^https?:$/.test(url.protocol)) continue;
      const link = document.createElement("a");
      link.href = url.href;
      link.setAttribute("download", "");
      link.setAttribute("rel", "noopener");
      let filename = url.pathname.split("/").pop() || name;
      try { filename = decodeURIComponent(filename); } catch { /* Keep the encoded filename. */ }
      link.textContent = typeof item === "object" && item?.filename ? item.filename : filename;
      if (name === "combined") {
        link.textContent = (files.imu ? "4분할 + IMU · " : "4분할 영상 · ") + filename;
        link.classList.add("recording-combined-link");
      }
      if (name === "imu") link.textContent = "IMU 그래프 영상 · " + filename;
      if (name === "dataset_csv") {
        link.textContent = "데이터셋 CSV · " + filename;
        link.classList.add("recording-dataset-link");
      }
      link.title = `${name} 파일 저장`;
      links.push(link);
    }
    $("recording-file-links").replaceChildren(...links);
    if (!links.length) {
      const empty = document.createElement("span");
      empty.textContent = recording.state === "cancelled" ? "촬영 전에 취소되어 저장된 영상이 없습니다." : "저장된 파일이 없습니다.";
      $("recording-file-links").append(empty);
    }
    recordingText("recording-result-title", recording.state === "completed" && links.length ? "녹화 저장 완료" : recording.state === "cancelled" ? "녹화 준비 취소" : "녹화 종료");
    recordingText("recording-result-id", String(recording.id));
  }

  function syncRecording(recording, requestStartedAt = performance.now()) {
    if (requestStartedAt < state.recordingConfirmedAt) return;
    state.recordingSupported = !!recording && typeof recording.state === "string";
    const normalized = recording ? { ...recording, state: recording.state === "canceled" ? "cancelled" : recording.state, remaining_seconds: recording.remaining_seconds ?? recording.remaining_s, elapsed_seconds: recording.elapsed_seconds ?? recording.duration_s, frames: recording.frames ?? recording.frames_written } : { state: "idle" };
    state.recordingAnchor = { data: normalized, observedAt: performance.now() };
    renderRecording();
  }

  function renderRecording() {
    const anchor = state.recordingAnchor;
    const recording = anchor?.data || { state: "idle" };
    const phase = recording.state || "idle";
    const sinceUpdate = anchor ? Math.max(0, performance.now() - anchor.observedAt) / 1000 : 0;
    const remaining = Math.max(0, recordingSeconds(recording.remaining_seconds) - sinceUpdate);
    const elapsed = recordingSeconds(recording.elapsed_seconds) + (phase === "recording" ? sinceUpdate : 0);
    const active = ["countdown", "recording", "stopping"].includes(phase);
    $("recording-panel").dataset.state = phase;
    const labels = { idle: "녹화 대기", countdown: remaining > 0 ? `${Math.ceil(remaining)}초 후 녹화 시작` : "서버의 촬영 시작을 확인하고 있습니다", recording: "녹화 중", stopping: "녹화를 마무리하고 있습니다", completed: "녹화를 종료했습니다", cancelled: "녹화 준비를 취소했습니다", error: "녹화 상태를 확인해 주세요" };
    recordingText("recording-status", !state.recordingSupported ? "녹화 기능 준비 중" : !state.online && active ? "서버 연결 확인 중 · 녹화 상태 재확인 필요" : labels[phase] || phase);
    recordingText("recording-badge-label", !state.online && active ? "연결 확인" : phase === "recording" ? "REC" : phase === "countdown" ? "준비" : phase === "stopping" ? "저장 중" : "VIDEO + CSV");
    recordingText("recording-timer", !state.online && active ? "—" : phase === "countdown" ? String(Math.ceil(remaining)) : recordingTime(elapsed));
    recordingText("recording-timer-unit", !state.online && active ? "상태 확인 필요" : phase === "countdown" ? remaining > 0 ? "초 후 시작" : "시작 확인 중" : phase === "recording" || elapsed > 0 ? "녹화 시간" : "경과 시간");
    const buttonLabel = state.recordingRequestPending ? active ? "중지 요청 중…" : "시작 요청 중…" : phase === "countdown" ? "녹화 준비 취소" : phase === "recording" ? "녹화 정지" : phase === "stopping" ? "파일 저장 중…" : "녹화 시작";
    recordingText("recording-button-label", buttonLabel);
    $("recording-button").disabled = !state.recordingSupported || !state.online || state.recordingRequestPending || phase === "stopping" || (!active && state.status?.camera?.state !== "streaming");
    const helpers = { countdown: "카메라 앞에서 준비하세요. 5초 뒤 영상과 데이터셋 CSV 기록을 시작합니다. IMU 원본은 CSV에 저장됩니다.", recording: "4분할 영상과 데이터셋 CSV를 저장하고 있습니다. IMU 원본은 CSV로 기록하며 그래프 영상은 만들지 않습니다.", stopping: "저장 중입니다. 완료되면 영상과 CSV를 내려받을 수 있습니다.", completed: "데이터셋 CSV와 녹화 영상을 아래에서 내려받으세요.", cancelled: "다시 시작을 누르면 새로 5초를 기다린 뒤 기록합니다.", error: "오류를 확인한 뒤 카메라가 연결되어 있으면 녹화를 다시 시작할 수 있습니다." };
    recordingText("recording-help", !state.online && active ? "서버와 다시 연결되면 실제 녹화 상태를 확인합니다." : helpers[phase] || (state.status?.camera?.state === "streaming" ? "5초 뒤 녹화를 시작합니다. RGB·Depth·RGB-D·Pose 영상과 센서·IMU·Pose 데이터셋 CSV를 함께 저장합니다." : "카메라를 시작한 뒤 녹화를 시작할 수 있습니다. 기록은 시작 버튼을 누른 5초 뒤에 시작합니다."));
    const frameNames = { combined: recording.imu_visualization?.enabled ? "4분할+IMU" : "4분할", imu: "IMU", rgb: "RGB", depth: "Depth", rgbd: "RGB-D", pose: "Pose", pose_rgbd: "Pose RGB-D" };
    const frames = Object.entries(recording.frames || {}).filter(([, count]) => typeof count === "number" && Number.isFinite(count));
    const dataset = recording.dataset || {};
    const datasetRowsByType = dataset.rows_written && typeof dataset.rows_written === "object" ? Object.entries(dataset.rows_written).filter(([, count]) => Number.isFinite(count)) : [];
    const datasetRows = Number.isFinite(dataset.rows_total) ? Math.max(0, dataset.rows_total) : Number.isFinite(dataset.rows_written) ? Math.max(0, dataset.rows_written) : datasetRowsByType.length ? datasetRowsByType.reduce((sum, [, count]) => sum + Math.max(0, count), 0) : null;
    const datasetDrops = recordingSeconds(dataset.dropped_total ?? dataset.dropped_samples);
    $("recording-progress").hidden = (!frames.length && datasetRows === null) || (!active && !recording.id);
    recordingText("recording-frames", frames.map(([name, count]) => `${frameNames[name] || name} ${count}프레임`).join(" · "));
    const datasetState = ({ idle: "대기", countdown: "준비", recording: "기록 중", writing: "기록 중", stopping: "저장 중", completed: "저장 완료", cancelled: "취소", error: "오류" })[dataset.state] || "";
    recordingText("recording-dataset", datasetRows === null ? "" : `CSV ${Math.floor(datasetRows).toLocaleString("ko-KR")}행${datasetState ? ` · ${datasetState}` : ""}${Number.isFinite(dataset.queue_depth) && dataset.queue_depth > 0 ? ` · 대기 ${dataset.queue_depth}` : ""}`);
    $("recording-dataset").dataset.state = dataset.state || "";
    $("recording-dataset").title = datasetRowsByType.map(([type, count]) => `${type}: ${Math.floor(count).toLocaleString("ko-KR")}행`).join(" · ");
    const dropped = typeof recording.dropped_frames === "object" && recording.dropped_frames !== null ? Object.values(recording.dropped_frames).reduce((total, count) => total + recordingSeconds(count), 0) : recordingSeconds(recording.dropped_frames);
    recordingText("recording-dropped", [dropped > 0 ? `영상 제외 ${Math.floor(dropped)}프레임` : "", datasetDrops > 0 ? `CSV 누락 ${Math.floor(datasetDrops)}샘플` : ""].filter(Boolean).join(" · "));
    const error = [recording.error, dataset.error].filter(Boolean).map((item) => typeof item === "string" ? item : JSON.stringify(item)).filter((item, index, list) => list.indexOf(item) === index).join("\n");
    $("recording-error").hidden = !error;
    recordingText("recording-error", error);
    recordingFileLinks(recording);
  }

  function sensorAge(value, observedAt) {
    return Number.isFinite(value) ? Math.max(0, value + performance.now() - observedAt) : null;
  }

  function ageLabel(age) {
    return age === null ? "최근 시각 —" : age < 1000 ? `${Math.round(age)} ms 전` : `${(age / 1000).toFixed(1)}초 전`;
  }

  function renderSensor(name, phase, hz, age, detail) {
    $("sensor-" + name).dataset.state = phase;
    recordingText(`sensor-${name}-state`, ({ receiving: "수신 중", stale: "지연", unavailable: "미수신", waiting: "대기", error: "오류", disabled: "꺼짐" })[phase] || "대기");
    recordingText(`sensor-${name}-hz`, `${format(hz)} Hz`);
    recordingText(`sensor-${name}-age`, ageLabel(age));
    recordingText(`sensor-${name}-detail`, detail);
  }

  function resetMotionHistory(epoch) {
    motion.epoch = epoch;
    motion.cursor = 0;
    motion.histories = { accel: [], gyro: [] };
    motion.lastSequence = { accel: -1, gyro: -1 };
    motion.lastSampleAt = 0;
    motion.lastSampleObservedAt = 0;
    motion.truncated = false;
  }

  function syncMotionStatus(imu, requestStartedAt = performance.now()) {
    if (!imu || typeof imu !== "object" || requestStartedAt < motion.requestStartedAt) return;
    const epoch = imu.capture_epoch ?? imu.capture_generation;
    if (epoch !== undefined && epoch !== null && epoch !== motion.epoch) resetMotionHistory(epoch);
    motion.anchor = { data: imu, observedAt: performance.now() };
    motion.requestStartedAt = requestStartedAt;
  }

  function acceptMotionSamples(result, requestStartedAt) {
    // A slow response from an older capture must never revive old measurements.
    if (requestStartedAt < motion.requestStartedAt) return false;
    if (Number.isFinite(result.latest_id) && result.latest_id < motion.cursor) {
      // A restarted server can reuse capture_generation=1 while its sample IDs
      // return to zero. Never keep asking that server for the old high cursor.
      resetMotionHistory(result.imu?.capture_epoch ?? result.imu?.capture_generation ?? motion.epoch);
      syncMotionStatus(result.imu, requestStartedAt);
      return true;
    }
    syncMotionStatus(result.imu, requestStartedAt);
    if (result.truncated) motion.truncated = true;
    const samples = Array.isArray(result.samples) ? result.samples : [];
    let added = false;
    for (const sample of samples) {
      const name = sample.stream ?? sample.type;
      if (name !== "accel" && name !== "gyro") continue;
      const epoch = sample.capture_epoch ?? sample.capture_generation;
      if (epoch !== undefined && epoch !== null && motion.epoch !== null && epoch !== motion.epoch) continue;
      const sequence = sample.sequence ?? sample.sample_id;
      const time = sample.received_monotonic ?? sample.received_time;
      if (!Number.isFinite(sequence) || sequence <= motion.lastSequence[name] || !Number.isFinite(time) || ![sample.x, sample.y, sample.z].every(Number.isFinite)) continue;
      motion.histories[name].push({ sequence, time, x: sample.x, y: sample.y, z: sample.z });
      motion.lastSequence[name] = sequence;
      motion.lastSampleAt = Math.max(motion.lastSampleAt, time);
      added = true;
    }
    if (added) motion.lastSampleObservedAt = performance.now();
    const latest = result.latest_sequence;
    if (Number.isFinite(latest)) motion.cursor = Math.max(motion.cursor, latest);
    for (const name of ["accel", "gyro"]) motion.histories[name] = motion.histories[name].filter((sample) => sample.time >= motion.lastSampleAt - 12).slice(-3000);
    return false;
  }

  function drawMotionChart(name) {
    const end = motion.lastSampleAt + (motion.lastSampleObservedAt ? Math.max(0, performance.now() - motion.lastSampleObservedAt) / 1000 : 0);
    const start = end - 10;
    const points = motion.histories[name].filter((sample) => sample.time >= start && sample.time <= end + 0.05);
    const range = points.length ? Math.max(name === "accel" ? 1 : 0.1, ...points.flatMap((sample) => [Math.abs(sample.x), Math.abs(sample.y), Math.abs(sample.z)])) * 1.1 : 0;
    recordingText(`imu-${name}-range`, range ? `± ${format(range, name === "accel" ? 1 : 2)}` : "수신 대기");
    $(`imu-${name}-empty`).hidden = points.length > 0;
    for (const axis of ["x", "y", "z"]) {
      let previous = null;
      const path = points.map((point) => {
        const x = Math.max(0, Math.min(600, (point.time - start) * 60));
        const y = 50 - point[axis] / range * 44;
        const move = previous === null || point.time - previous > 0.2;
        previous = point.time;
        return `${move ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ");
      $(`imu-${name}-${axis}-path`).setAttribute("d", path);
    }
  }

  function renderSensorData() {
    const camera = state.status?.camera || {};
    const pose = state.status?.pose || {};
    const running = state.online && camera.state === "streaming";
    const cameraAge = sensorAge(camera.frame_age_ms, sensorStatusObservedAt);
    // Status is polled every 1.2 seconds; allow that normal polling interval
    // before classifying the anchored receipt age as a stalled image source.
    const cameraPhase = !state.online ? "unavailable" : !running ? camera.state === "error" ? "error" : "waiting" : cameraAge !== null && cameraAge > 2500 ? "stale" : Number(camera.frame_id) > 0 ? "receiving" : "waiting";
    renderSensor("rgb", cameraPhase, running ? camera.fps : null, cameraAge, camera.frame_id ? `프레임 ${camera.frame_id} · 컬러` : "컬러 원본 프레임 대기");
    renderSensor("depth", cameraPhase, running ? camera.fps : null, cameraAge, camera.frame_id ? `프레임 ${camera.frame_id} · 정렬 Z16` : "정렬된 깊이 프레임 대기");
    const poseAge = sensorAge(pose.frame_age_ms, sensorStatusObservedAt);
    const disabled = state.status?.settings?.pose_enabled === false;
    const posePhase = !state.online ? "unavailable" : disabled ? "disabled" : pose.state === "error" ? "error" : running && poseIsCurrent() ? poseAge !== null && poseAge > 2000 ? "stale" : "receiving" : "waiting";
    const poseDetail = posePhase === "receiving" || posePhase === "stale" ? `${Number.isFinite(pose.visible_keypoints) ? pose.visible_keypoints + "점" : "관절"} · 유효 깊이 ${Number.isFinite(pose.valid_depth_keypoints) ? pose.valid_depth_keypoints + "점" : "—"}` : disabled ? "Pose 추론 꺼짐" : pose.error || "모델의 추론 결과 대기";
    renderSensor("pose", posePhase, posePhase === "receiving" || posePhase === "stale" ? pose.fps : null, poseAge, poseDetail);
    const imu = motion.anchor?.data || state.status?.imu || {};
    const observedAt = motion.anchor?.observedAt || sensorStatusObservedAt;
    const unavailable = !state.online || motion.unavailable || imu.available === false || ["unavailable", "unsupported", "disconnected", "disabled"].includes(imu.state);
    for (const name of ["accel", "gyro"]) {
      const sample = imu[name] || {};
      const hasValue = [sample.x, sample.y, sample.z].every(Number.isFinite);
      const age = sensorAge(sample.age_ms, observedAt);
      const staleAfter = sample.hz > 0 ? Math.max(500, 5000 / sample.hz) : 1000;
      const phase = unavailable ? "unavailable" : imu.state === "error" ? "error" : hasValue ? imu.stream_stale?.[name] || (age !== null && age > staleAfter) ? "stale" : "receiving" : "waiting";
      const unit = name === "accel" ? "m/s²" : "rad/s";
      renderSensor(name, phase, sample.hz, age, hasValue ? `${Number.isFinite(sample.count) ? sample.count.toLocaleString("ko-KR") + "개 · " : ""}${unit}` : `IMU · ${unit}`);
      $(`imu-${name}-chart`).dataset.state = phase;
      for (const axis of ["x", "y", "z"]) recordingText(`imu-${name}-${axis}`, hasValue && !unavailable ? format(sample[axis], 3) : "—");
      drawMotionChart(name);
    }
    const imuError = imu.reason || imu.error || motion.error;
    const activeIMU = !unavailable && [imu.accel, imu.gyro].some((sample) => sample && [sample.x, sample.y, sample.z].every(Number.isFinite));
    recordingText("imu-reason", !state.online ? "서버 연결을 확인하고 있습니다. 이전 값의 수신 시각이 계속 증가하면 데이터가 멈춘 상태입니다." : imuError ? String(imuError) : activeIMU ? "D435i IMU 원본 측정값 · 가속도에는 중력이 포함됩니다." : "IMU 원본 측정값 대기 중입니다. 센서가 인식되지 않으면 이곳에 원인을 표시합니다.");
    $("imu-reason").dataset.state = activeIMU && !imuError ? "receiving" : unavailable || imu.state === "error" ? "unavailable" : "waiting";
    recordingText("imu-history-note", motion.truncated ? "최근 10초 · 화면 기록 일부 생략 · CSV 저장 상태는 위에서 확인" : "최근 10초 · 수신 시각 기준 · X / Y / Z");
  }

  async function pollMotion() {
    if (motion.busy || document.hidden || !state.online || performance.now() < motion.nextPollAt) return;
    motion.busy = true;
    const requestStartedAt = performance.now();
    try {
      const result = await api(`/api/imu?after=${encodeURIComponent(motion.cursor)}`, undefined, 4000);
      if (acceptMotionSamples(result, requestStartedAt)) {
        const retryStartedAt = performance.now();
        acceptMotionSamples(await api("/api/imu?after=0", undefined, 4000), retryStartedAt);
      }
      motion.unavailable = false;
      motion.error = "";
    } catch (error) {
      motion.error = `IMU 수신 확인: ${safeError(error)}`;
      motion.nextPollAt = performance.now() + 4000;
      // One missing history endpoint must not hide the image or Pose streams.
      if (!motion.anchor && !state.status?.imu) motion.unavailable = true;
    } finally {
      motion.busy = false;
      renderSensorData();
    }
  }

  async function api(path, body, timeout = 10000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(path, { method: body === undefined ? "GET" : "POST", cache: "no-store", headers: body === undefined ? {} : { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal });
      const data = await response.json().catch(() => null);
      if (!response.ok || data?.ok === false) throw new Error(data?.error || `서버 응답 오류 (${response.status})`);
      if (!data) throw new Error("서버 응답을 읽을 수 없습니다.");
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("서버 응답 시간이 초과되었습니다.");
      throw error;
    } finally { clearTimeout(timer); }
  }

  function toast(message, kind = "success") {
    clearTimeout(state.toastTimer);
    $("toast").textContent = message;
    $("toast").dataset.kind = kind;
    $("toast").hidden = false;
    state.toastTimer = setTimeout(() => { $("toast").hidden = true; }, kind === "error" ? 6500 : 4000);
  }

  function notice(message, kind = "info") {
    $("notice").hidden = !message;
    $("notice").dataset.kind = kind;
    $("notice-text").textContent = message || "";
  }

  function cardMessage(name, title, description) {
    cards[name].querySelector(".placeholder-title").textContent = title;
    cards[name].querySelector(".placeholder-description").textContent = description;
  }

  function closeStream(name) {
    clearTimeout(state.streamRetry[name]);
    delete state.streamRetry[name];
    images[name].removeAttribute("src");
    images[name].dataset.attached = "false";
    cards[name].dataset.feed = "idle";
  }

  function streamWanted(name) {
    const wanted = state.online && state.pendingCameraAction !== "stop" && state.status?.camera?.state === "streaming" && !document.hidden && (state.layout === "all" || state.layout === name);
    if (!wanted || name !== "pose") return wanted;
    return state.status?.settings?.pose_enabled === false || poseIsCurrent();
  }

  function attachStream(name) {
    if (!streamWanted(name) || images[name].dataset.attached === "true") return;
    images[name].dataset.attached = "true";
    cards[name].dataset.feed = "pending";
    cardMessage(name, "프레임 수신 중", "카메라에서 실시간 영상을 가져오고 있습니다.");
    const endpoint = name === "pose" && state.poseBackground === "rgbd" ? "pose_rgbd" : name;
    images[name].src = `/stream/${endpoint}.mjpg?v=${Date.now()}-${++state.streamSerial}`;
  }

  function syncStreams() {
    for (const name of streams) {
      if (streamWanted(name)) {
        attachStream(name);
        if (images[name].dataset.attached === "true" && images[name].naturalWidth > 0 && cards[name].dataset.feed === "pending") cards[name].dataset.feed = "live";
      } else {
        if (images[name].hasAttribute("src")) closeStream(name);
        const camera = state.status?.camera || {};
        if (!state.online) cardMessage(name, "서버 연결 대기", "Jetson과의 네트워크 연결을 확인해 주세요.");
        else if (name === "pose" && camera.state === "streaming") {
          const model = models[selectedModel()] || models.yolo;
          const failed = state.status?.pose?.state === "error" && !state.modelRequestPending;
          cardMessage(name, failed ? `${model.name} 실행 오류` : `${model.name} 준비 중`, failed ? "다른 모델을 선택하거나 아래 오류를 확인해 주세요." : "모델을 준비한 뒤 해당 모델의 실제 추론 영상을 표시합니다.");
        }
        else if (camera.state === "error") cardMessage(name, "카메라 연결 오류", "USB 연결을 확인하고 카메라를 다시 시작해 주세요.");
        else if (camera.state === "connecting") cardMessage(name, "카메라 연결 중", "센서와 스트림을 준비하고 있습니다.");
        else if (camera.active) cardMessage(name, "카메라 연결 대기", "D435i를 인식하면 영상이 자동으로 표시됩니다.");
        else cardMessage(name, "카메라 연결 대기", "카메라 시작 버튼으로 실시간 영상을 확인하세요.");
      }
    }
  }

  for (const name of streams) {
    images[name].addEventListener("load", () => { if (streamWanted(name) && images[name].dataset.attached === "true") cards[name].dataset.feed = "live"; });
    images[name].addEventListener("error", () => {
      if (!streamWanted(name) || images[name].dataset.attached !== "true") return;
      cards[name].dataset.feed = "error";
      cardMessage(name, "영상 수신이 중단되었습니다", "연결을 자동으로 다시 시도하고 있습니다.");
      clearTimeout(state.streamRetry[name]);
      state.streamRetry[name] = setTimeout(() => { closeStream(name); attachStream(name); }, 5000);
    });
  }

  function setLayout(layout) {
    state.layout = layout;
    $("view-grid").dataset.layout = layout;
    document.querySelectorAll(".view-tab").forEach((button) => {
      const active = button.dataset.view === layout;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    for (const name of streams) {
      cards[name].hidden = layout !== "all" && layout !== name;
      const button = cards[name].querySelector("[data-expand]");
      button.setAttribute("aria-label", layout === name ? `${name.toUpperCase()} 영상 4분할로 복귀` : `${name.toUpperCase()} 영상 확대`);
      button.title = layout === name ? "4분할로 복귀" : "화면 확대";
    }
    syncStreams();
  }

  function renderStatus(status, requestStartedAt = performance.now()) {
    state.status = status;
    state.online = true;
    sensorStatusObservedAt = performance.now();
    const camera = status.camera || {};
    const pose = status.pose || {};
    const settings = status.settings || {};
    if (state.requestedModel && !state.modelRequestPending && settings.pose_model === state.requestedModel) state.requestedModel = null;
    const modelId = selectedModel();
    if (state.activeModel !== modelId) { closeStream("pose"); state.activeModel = modelId; }
    const device = camera.device || {};
    const running = camera.state === "streaming";
    const active = camera.active === true || running;
    $("connection-status").dataset.state = camera.state || "disconnected";
    $("connection-label").textContent = labels[camera.state] || "카메라 대기";
    $("device-name").textContent = device.name || "Intel RealSense D435i";
    $("device-serial").textContent = device.serial || "—";
    $("device-usb").textContent = device.usb || "—";
    $("device-resolution").textContent = camera.width && camera.height ? `${camera.width} × ${camera.height}` : "—";
    $("camera-fps").textContent = running ? format(camera.fps) : "—";
    const poseReady = running && pose.enabled !== false && settings.pose_enabled !== false && poseIsCurrent();
    $("pose-fps").textContent = poseReady ? format(pose.fps) : "—";
    $("inference-ms").textContent = poseReady ? format(pose.inference_ms) : "—";
    const countValue = pose.instances ?? pose.people;
    $("people-count").textContent = poseReady ? (typeof countValue === "number" ? countValue : Array.isArray(countValue) ? countValue.length : "—") : "—";
    const currentModel = models[modelId] || models.yolo;
    $("pose-model").textContent = pose.model_id && pose.model_id !== modelId ? currentModel.name : pose.model || currentModel.name;
    $("pose-model").title = pose.model_note || pose.model || currentModel.name;
    $("pose-backend").textContent = pose.model_id && pose.model_id !== modelId ? "모델 준비 중" : pose.backend || "모델 준비 대기";
    $("pose-backend").title = pose.backend || "";
    const modelState = state.modelRequestPending || (pose.model_id && pose.model_id !== modelId) ? "loading" : settings.pose_enabled === false ? "disabled" : pose.state;
    $("pose-state").textContent = ({ loading: "준비 중", ready: running ? "실행 중" : "준비됨", disabled: "꺼짐", error: "오류" })[modelState] || "대기";
    $("pose-state").dataset.state = modelState || "";
    if (poseReady && pose.frame_id !== undefined && pose.frame_id !== null) {
      const age = Number.isFinite(pose.frame_age_ms) ? (pose.frame_age_ms >= 1000 ? `${(pose.frame_age_ms / 1000).toFixed(1)}초 전` : `${Math.round(pose.frame_age_ms)}ms 전`) : "";
      $("pose-frame-note").textContent = `프레임 ${pose.frame_id}${age ? ` · ${age}` : ""}`;
    } else $("pose-frame-note").textContent = "선택 모델의 Pose 원본 프레임";
    $("live-badge").dataset.live = String(running);
    $("live-badge").replaceChildren(Object.assign(document.createElement("span"), {}), document.createTextNode(running ? "LIVE" : "STANDBY"));
    const cameraButton = $("camera-button");
    cameraButton.classList.toggle("is-running", active);
    cameraButton.querySelector("use").setAttribute("href", active ? "#i-stop" : "#i-play");
    cameraButton.querySelector("span").textContent = state.busy ? "처리 중…" : active ? "카메라 정지" : camera.state === "error" ? "카메라 다시 시작" : "카메라 시작";
    cameraButton.disabled = state.busy;
    $("snapshot-button").disabled = !running || state.busy || state.modelRequestPending || (settings.pose_enabled !== false && (!poseIsCurrent() || pose.frame_id === null));
    $("camera-help").textContent = running ? "카메라가 실시간 영상을 전송하고 있습니다." : active ? "자동 재연결 대기 중입니다. USB 3 연결을 확인하거나 카메라를 정지해 주세요." : "USB 3 연결 상태를 확인한 후 시작해 주세요.";
    if (!state.settingsReady) {
      if (Number.isFinite(settings.confidence)) $("confidence").value = settings.confidence;
      if (Number.isFinite(settings.depth_min)) $("depth-min").value = settings.depth_min;
      if (Number.isFinite(settings.depth_max)) $("depth-max").value = settings.depth_max;
      $("pose-toggle").checked = settings.pose_enabled !== false;
      state.settingsReady = true;
      updateSettingsLabels();
    }
    syncROI(settings.pose_roi);
    syncModelControls();
    syncRecording(status.recording, requestStartedAt);
    syncMotionStatus(status.imu, requestStartedAt);
    renderSensorData();
    if (camera.error && (active || camera.state === "error")) notice(`카메라: ${camera.error}`, "error");
    else if (pose.error && settings.pose_enabled !== false && !state.modelRequestPending && (!pose.model_id || pose.model_id === modelId)) notice(`Pose: ${pose.error} RGB와 깊이 영상은 계속 확인할 수 있습니다.`, "error");
    else if (modelState === "loading") notice(`${currentModel.name} 모델을 준비하고 있습니다. 처음 실행할 때는 시간이 걸릴 수 있습니다.`, "info");
    else if (camera.state === "error") notice("카메라 USB 연결 상태를 확인한 후 카메라를 다시 시작해 주세요.", "error");
    else notice("");
    $("last-update").textContent = `${new Date().toLocaleTimeString("ko-KR", { hour12: false })} 업데이트`;
    syncStreams();
  }

  async function pollStatus() {
    if (state.requestInFlight || document.hidden) return;
    state.requestInFlight = true;
    const requestStartedAt = performance.now();
    try { renderStatus(await api("/api/status", undefined, 6000), requestStartedAt); }
    catch (error) {
      state.online = false;
      $("connection-status").dataset.state = "offline";
      $("connection-label").textContent = labels.offline;
      $("camera-button").disabled = true;
      $("snapshot-button").disabled = true;
      $("pose-model-select").disabled = true;
      $("confidence").disabled = true;
      for (const id of [...roiIds, "pose-roi-apply", "pose-roi-clear"]) $(id).disabled = true;
      $("live-badge").dataset.live = "false";
      $("live-badge").replaceChildren(document.createElement("span"), document.createTextNode("OFFLINE"));
      for (const id of ["camera-fps", "pose-fps", "inference-ms", "people-count"]) $(id).textContent = "—";
      $("last-update").textContent = "서버 연결 재시도 중";
      notice(`Jetson 서버에 연결할 수 없습니다. 같은 네트워크와 서버 실행 상태를 확인해 주세요. ${safeError(error)}`, "error");
      renderRecording();
      renderSensorData();
      syncStreams();
    } finally { state.requestInFlight = false; }
  }

  function updateSettingsLabels() {
    $("confidence-value").value = selectedModel() === "yolo" || selectedModel() === "rtmpose" ? Number($("confidence").value).toFixed(2) : "모든 점";
    const near = Number($("depth-min").value), far = Number($("depth-max").value);
    if (Number.isFinite(near) && Number.isFinite(far) && near >= 0.1 && far <= 10 && near < far) $("depth-range-label").textContent = `${near.toFixed(1)}–${far.toFixed(1)} m`;
  }

  async function saveSettings() {
    updateSettingsLabels();
    const settings = { confidence: Number($("confidence").value), depth_min: Number($("depth-min").value), depth_max: Number($("depth-max").value), pose_enabled: $("pose-toggle").checked };
    if (!Number.isFinite(settings.depth_min) || !Number.isFinite(settings.depth_max) || settings.depth_min < 0.1 || settings.depth_max > 10 || settings.depth_min >= settings.depth_max) {
      toast("깊이 범위는 0.1–10 m 안에서 최소 거리보다 최대 거리가 커야 합니다.", "error");
      return;
    }
    try { await api("/api/settings", settings); await pollStatus(); }
    catch (error) { toast(`설정을 적용하지 못했습니다. ${safeError(error)}`, "error"); }
  }

  $("recording-button").addEventListener("click", async () => {
    if (state.recordingRequestPending || !state.online || !state.recordingSupported) return;
    const phase = state.recordingAnchor?.data?.state || "idle";
    if (phase === "stopping") return;
    const stopping = phase === "countdown" || phase === "recording";
    if (!stopping && state.status?.camera?.state !== "streaming") return;
    state.recordingRequestPending = true;
    renderRecording();
    try {
      const response = await api(`/api/recording/${stopping ? "stop" : "start"}`, {}, 20000);
      const recording = response.result?.recording || response.result || response.recording;
      state.recordingConfirmedAt = performance.now();
      if (recording && typeof recording.state === "string") syncRecording(recording, state.recordingConfirmedAt);
    } catch (error) {
      toast(`녹화 ${stopping ? "정지" : "시작"} 요청에 실패했습니다. ${safeError(error)}`, "error");
    } finally {
      state.recordingRequestPending = false;
      renderRecording();
      await pollStatus();
    }
  });

  $("pose-model-select").addEventListener("change", async () => {
    const modelId = $("pose-model-select").value;
    if (!models[modelId] || state.modelRequestPending) return;
    state.requestedModel = modelId;
    state.modelRequestPending = true;
    closeStream("pose");
    for (const id of ["pose-fps", "inference-ms", "people-count"]) $(id).textContent = "—";
    $("pose-state").textContent = "준비 중";
    $("pose-state").dataset.state = "loading";
    $("pose-model").textContent = models[modelId].name;
    $("pose-backend").textContent = "모델 변경 중";
    syncModelControls();
    syncStreams();
    notice(`${models[modelId].name} 모델로 변경하고 있습니다.`, "info");
    try {
      await api("/api/settings", { pose_model: modelId });
    } catch (error) {
      state.requestedModel = null;
      toast(`모델을 변경하지 못했습니다. ${safeError(error)}`, "error");
    } finally {
      state.modelRequestPending = false;
      syncModelControls();
      await pollStatus();
    }
  });

  async function applyROI(roi) {
    if (state.roiSaving) return;
    state.roiSaving = true;
    closeStream("pose");
    syncModelControls();
    try {
      await api("/api/settings", { pose_roi: roi });
      state.roiDirty = false;
      toast(roi ? "손·팔 영역을 실시간 추론에 적용했습니다." : "실시간 추론 영역을 전체 영상으로 되돌렸습니다.");
    } catch (error) {
      toast(`영역을 적용하지 못했습니다. ${safeError(error)}`, "error");
    } finally {
      state.roiSaving = false;
      syncModelControls();
      await pollStatus();
    }
  }

  roiIds.forEach((id) => $(id).addEventListener("input", () => { state.roiDirty = true; syncModelControls(); }));
  $("pose-roi-apply").addEventListener("click", () => {
    const fields = roiIds.map((id) => $(id).value);
    if (fields.every((value) => value === "")) return applyROI(null);
    const roi = fields.map(Number);
    const width = state.status?.camera?.width || 640;
    const height = state.status?.camera?.height || 480;
    if (fields.some((value) => value === "") || roi.some((value) => !Number.isFinite(value)) || roi[0] < 0 || roi[1] < 0 || roi[2] > width || roi[3] > height || roi[2] - roi[0] < 12 || roi[3] - roi[1] < 12) {
      toast(`영역의 네 좌표를 입력하세요. ${width} × ${height} 영상 안에서 가로·세로가 12픽셀 이상이어야 합니다.`, "error");
      return;
    }
    applyROI(roi.map(Math.round));
  });
  $("pose-roi-clear").addEventListener("click", () => applyROI(null));

  $("camera-button").addEventListener("click", async () => {
    if (state.busy) return;
    const camera = state.status?.camera || {};
    const stopping = camera.active === true || camera.state === "streaming";
    state.busy = true;
    state.pendingCameraAction = stopping ? "stop" : "start";
    $("camera-button").disabled = true;
    $("camera-button").querySelector("span").textContent = stopping ? "카메라 정지 중…" : "카메라 시작 중…";
    if (stopping) for (const name of streams) closeStream(name);
    try { await api(`/api/camera/${stopping ? "stop" : "start"}`, {}, 30000); }
    catch (error) { toast(`카메라 ${stopping ? "정지" : "시작"}에 실패했습니다. ${safeError(error)}`, "error"); }
    finally { state.busy = false; state.pendingCameraAction = null; await pollStatus(); }
  });
  $("pose-toggle").addEventListener("change", saveSettings);
  $("confidence").addEventListener("input", updateSettingsLabels);
  $("confidence").addEventListener("change", saveSettings);
  for (const id of ["depth-min", "depth-max"]) $(id).addEventListener("change", saveSettings);
  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => setLayout(button.dataset.view)));
  document.querySelectorAll("[data-expand]").forEach((button) => button.addEventListener("click", () => setLayout(state.layout === button.dataset.expand ? "all" : button.dataset.expand)));
  document.querySelectorAll("[data-pose-base]").forEach((button) => button.addEventListener("click", () => {
    if (state.poseBackground === button.dataset.poseBase) return;
    state.poseBackground = button.dataset.poseBase;
    document.querySelectorAll("[data-pose-base]").forEach((item) => item.setAttribute("aria-pressed", String(item.dataset.poseBase === state.poseBackground)));
    closeStream("pose");
    attachStream("pose");
  }));
  $("mobile-settings-toggle").addEventListener("click", () => {
    const expanded = $("mobile-settings-toggle").getAttribute("aria-expanded") !== "true";
    $("mobile-settings-toggle").setAttribute("aria-expanded", String(expanded));
    $("mobile-settings-toggle").replaceChildren(document.createTextNode(expanded ? "모델 · Pose · Depth 설정 접기 " : "모델 · Pose · Depth 설정 펼치기 "), Object.assign(document.createElement("span"), { textContent: expanded ? "−" : "+" }));
    document.querySelector(".sidebar").classList.toggle("is-expanded", expanded);
  });

  function snapshotLinks(files) {
    const entries = Object.entries(files || {});
    const links = document.createElement("div");
    links.className = "snapshot-links";
    for (const [name, value] of entries) {
      const url = typeof value === "string" ? value : value?.url;
      if (!url) continue;
      let parsed;
      try { parsed = new URL(url, window.location.origin); } catch { continue; }
      if (parsed.origin !== window.location.origin || !/^https?:$/.test(parsed.protocol)) continue;
      const link = document.createElement("a");
      link.href = parsed.href;
      link.textContent = name.toUpperCase();
      link.setAttribute("download", "");
      link.setAttribute("rel", "noopener");
      links.appendChild(link);
    }
    return links;
  }

  $("snapshot-button").addEventListener("click", async () => {
    const button = $("snapshot-button");
    button.disabled = true;
    try {
      const result = await api("/api/snapshot", {}, 20000);
      const container = $("snapshot-result");
      const title = document.createElement("strong");
      title.textContent = "프레임 저장 완료";
      container.replaceChildren(title, snapshotLinks(result.files));
      const id = document.createElement("span");
      id.textContent = result.id || "Jetson의 captures 폴더에 저장했습니다.";
      container.appendChild(id);
      container.hidden = false;
      toast("Pose와 같은 시점의 RGB·깊이·관절 데이터를 저장했습니다.");
    } catch (error) { toast(`프레임 저장에 실패했습니다. ${safeError(error)}`, "error"); }
    finally { button.disabled = state.status?.camera?.state !== "streaming" || !state.online || state.modelRequestPending || (state.status?.settings?.pose_enabled !== false && !poseIsCurrent()); }
  });

  document.addEventListener("visibilitychange", () => { if (document.hidden) for (const name of streams) closeStream(name); else { pollStatus(); pollMotion(); syncStreams(); } });
  window.addEventListener("pagehide", () => { for (const name of streams) closeStream(name); });
  setInterval(pollStatus, 1200);
  setInterval(renderRecording, 200);
  setInterval(pollMotion, 250);
  setInterval(() => { if (!document.hidden) renderSensorData(); }, 200);
  renderRecording();
  renderSensorData();
  pollStatus();
})();
