'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const names = { mediapipe: 'MediaPipe Hand', rtmpose: 'RTMPose-m', egoforce: 'EgoForce HALO' };
  const descriptions = {
    mediapipe: { task: '손 · 손가락 21점', detail: '전체 영상에서 손을 찾아 손가락과 손목을 추정합니다. 점별 신뢰도는 제공되지 않으며, 좌우 손 분류 점수는 관절의 정확도가 아닙니다.' },
    rtmpose: { task: '몸 관절 · 어깨 / 팔꿈치 / 손목', detail: '지정한 팔 영역 또는 전체 영상으로 몸 관절을 추정합니다. 손가락 관절은 출력하지 않습니다. 화면 밖 관절은 확인할 수 없으므로 추정 결과를 원본과 함께 살펴보세요.' },
    egoforce: { task: '손 + 전완 · 2D HALO 비교', detail: 'MediaPipe가 찾은 손 영역과 지정한 팔 영역을 함께 사용합니다. 여기서는 2D HALO 관절 출력을 비교하며, 3D 메시나 힘 추정 결과는 포함하지 않습니다.' },
  };
  const statusLabels = { idle: '실행 대기', queued: '대기 중', loading: '모델 준비 중', running: '추론 중', done: '완료', error: '실행 오류', skipped: '실행 생략', cancelled: '중지됨' };
  const state = { connected: false, connectionError: false, models: [], captures: [], latest: null, display: null, historical: false, sourceEdited: false, roi: null, dragStart: null, pollBusy: false, submitting: false, autoTimer: null, autoScheduledId: null, previewKey: null, historyKey: null };
  const terminal = (job) => !!job && ['done', 'error', 'cancelled'].includes(job.state) && !job.busy;
  const busy = () => state.submitting || !!(state.latest && (state.latest.busy || ['queued', 'running'].includes(state.latest.state)));
  const number = (value, digits = 0) => Number.isFinite(Number(value)) && value !== null && value !== '' ? Number(value).toFixed(digits) : '—';
  const put = (id, text) => { $(id).textContent = text; };

  async function api(path, payload) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 9000);
    try {
      const response = await fetch(path, { method: payload === undefined ? 'GET' : 'POST', cache: 'no-store', signal: controller.signal, ...(payload === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }) });
      const result = await response.json();
      if (!response.ok || result.ok === false) throw new Error(result.error || `서버 응답 오류 (${response.status})`);
      return result;
    } finally { clearTimeout(timer); }
  }

  function notice(message = '') {
    put('notice', message);
    $('notice').hidden = !message;
  }

  function imageDimensions() {
    const image = $('source-preview');
    return [image.naturalWidth || 640, image.naturalHeight || 480];
  }

  function setPreview(url, key, tag, caption) {
    put('preview-tag', tag);
    put('source-caption', caption);
    if (state.previewKey === key) return;
    state.previewKey = key;
    $('source-preview').hidden = true;
    $('preview-empty').hidden = false;
    $('preview-frame').classList.add('is-empty');
    if (url) $('source-preview').src = url;
    else $('source-preview').removeAttribute('src');
  }

  function showSelectedSource() {
    if (busy()) return;
    const selected = $('source-select').value;
    const capture = state.captures.find((item) => item.id === selected);
    if (selected === 'live') setPreview('/stream/rgb.mjpg', 'live', 'LIVE PREVIEW', '현재 카메라 프레임 · 실행 시 한 장을 고정합니다.');
    else if (capture) setPreview(capture.rgb_url, `capture:${capture.id}`, 'SAVED FRAME', capture.id);
    else setPreview(null, 'empty', 'NO INPUT', '저장한 프레임을 찾을 수 없습니다.');
  }

  function paintROI(syncInputs = true) {
    const canvas = $('roi-canvas');
    const [width, height] = imageDimensions();
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    const roi = state.roi;
    if (roi) {
      const [x1, y1, x2, y2] = roi;
      ctx.fillStyle = 'rgba(0, 8, 12, .35)';
      ctx.beginPath(); ctx.rect(0, 0, width, height); ctx.rect(x1, y1, x2 - x1, y2 - y1); ctx.fill('evenodd');
      const scale = width / Math.max(1, canvas.clientWidth);
      ctx.lineWidth = 1.5 * scale;
      ctx.strokeStyle = '#68f5cc';
      ctx.setLineDash([6 * scale, 4 * scale]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
      ctx.fillStyle = '#68f5cc';
      ctx.font = `${11 * scale}px ui-monospace, Consolas, monospace`;
      ctx.fillText('ARM ROI', x1 + 7 * scale, Math.min(y1 + 17 * scale, y2 - 3 * scale));
    }
    put('roi-caption', roi ? `ROI ${roi[2] - roi[0]} × ${roi[3] - roi[1]}` : '전체 영상');
    if (syncInputs) ['x1', 'y1', 'x2', 'y2'].forEach((key, index) => { $('roi-' + key).value = roi ? roi[index] : ''; });
    $('clear-roi').disabled = busy() || state.historical || (!roi && !['x1', 'y1', 'x2', 'y2'].some((key) => $('roi-' + key).value !== ''));
  }

  function pointerPoint(event) {
    const bounds = $('roi-canvas').getBoundingClientRect();
    const [width, height] = imageDimensions();
    return [Math.round(Math.max(0, Math.min(width, (event.clientX - bounds.left) / bounds.width * width))), Math.round(Math.max(0, Math.min(height, (event.clientY - bounds.top) / bounds.height * height)))];
  }

  function createCards() {
    for (const [index, id] of Object.keys(names).entries()) {
      const article = document.createElement('article');
      article.className = 'model-card'; article.id = `card-${id}`; article.dataset.state = 'idle';
      article.innerHTML = `<header class="model-header"><div class="model-title"><span class="model-index">0${index + 1}</span><div><h3></h3><p></p></div></div><span class="status-pill" data-role="status" data-state="idle">실행 대기</span></header><div class="result-image"><button type="button" data-role="expand" hidden><img data-role="image"><span class="image-hint">확대 ↗</span></button><div class="result-placeholder" data-role="placeholder"><span aria-hidden="true">◇</span><strong data-role="placeholder-title">같은 프레임으로 비교</strong><small data-role="placeholder-detail">비교를 실행하면 결과가 표시됩니다.</small></div></div><dl class="metrics"><div class="metric"><dt>추론 시간</dt><dd><span data-role="inference">—</span><small>ms</small></dd></div><div class="metric"><dt>표시된 관절</dt><dd><span data-role="points">—</span><small>개</small></dd></div><div class="metric"><dt>유효 깊이 관절</dt><dd><span data-role="depth">—</span><small>개</small></dd></div></dl><div class="model-details"><div class="timing-line"><span data-role="backend">모델 준비 대기</span><span>모델 로딩 <strong data-role="load">—</strong> ms</span></div><p class="no-detections" data-role="detection-note" hidden></p><p class="model-description"></p><pre class="error-detail" data-role="error" hidden></pre><a class="result-link" data-role="json" download hidden>관절 데이터 JSON ↓</a></div>`;
      article.querySelector('h3').textContent = names[id];
      article.querySelector('.model-title p').textContent = descriptions[id].task;
      article.querySelector('.model-description').textContent = descriptions[id].detail;
      article.querySelector('[data-role="image"]').alt = `${names[id]}의 실제 관절 추정 결과`;
      article.querySelector('[data-role="expand"]').addEventListener('click', () => {
        const result = state.display && state.display.results && state.display.results[id];
        if (!result || !result.image_url) return;
        $('dialog-image').src = result.image_url;
        put('dialog-title', `${names[id]} · 결과 확대`);
        put('dialog-caption', `프레임 ${state.display.source?.frame_id ?? '—'} · 표시 관절 ${result.visible_keypoints ?? '—'}개`);
        $('dialog-download').href = result.image_url;
        $('dialog-download').download = `${state.display.id}_${id}.jpg`;
        $('image-dialog').showModal();
      });
      article.querySelector('[data-role="image"]').addEventListener('error', () => {
        article.querySelector('[data-role="expand"]').hidden = true;
        article.querySelector('[data-role="placeholder"]').hidden = false;
        article.querySelector('[data-role="placeholder-title"]').textContent = '결과 이미지를 불러오지 못했습니다';
        article.querySelector('[data-role="placeholder-detail"]').textContent = '결과 JSON에서 실행 상태를 확인할 수 있습니다.';
      });
      $('model-grid').append(article);
    }
  }

  function renderCard(id, job) {
    const card = $('card-' + id);
    const find = (role) => card.querySelector(`[data-role="${role}"]`);
    const result = job?.results?.[id];
    const model = state.models.find((item) => item.id === id);
    const status = result?.state || (model && !model.available ? 'skipped' : 'idle');
    card.dataset.state = status;
    find('status').dataset.state = status;
    find('status').textContent = statusLabels[status] || status;
    const complete = status === 'done';
    find('inference').textContent = complete ? number(result.inference_ms, 1) : '—';
    find('points').textContent = complete ? number(result.visible_keypoints) : '—';
    find('depth').textContent = complete ? number(result.valid_depth_keypoints) : '—';
    find('backend').textContent = result?.backend || (model?.available ? '실행 준비 완료' : '모델 준비 대기');
    find('load').textContent = number(result?.load_ms, 0);
    const error = result?.error || (status === 'skipped' && model && !model.available ? model.error || '모델 준비가 완료되지 않았습니다. 준비된 모델부터 비교할 수 있습니다.' : '');
    find('error').hidden = !error;
    find('error').textContent = typeof error === 'string' ? error : JSON.stringify(error, null, 2);
    const instances = Array.isArray(result?.instances) ? result.instances : [];
    find('detection-note').hidden = !complete;
    const instanceLabel = id === 'rtmpose' ? 'ROI 추정' : id === 'egoforce' ? '처리한 손 영역' : '감지된 손';
    find('detection-note').textContent = !instances.length ? (id === 'mediapipe' ? '이 프레임에서 감지된 손이 없습니다.' : '이 프레임에서 출력된 관절이 없습니다.') : `${instanceLabel} ${instances.length}개 · ${number(result.visible_keypoints)}개 관절 표시`;
    if (complete && id === 'egoforce' && !instances.length && result.metadata?.no_hand_reason) {
      find('detection-note').textContent = '공유할 손 검출이 없습니다. 손을 감싸는 작은 영역을 지정하면 수동 손 crop으로 비교할 수 있습니다.';
    }
    const image = complete && result.image_url;
    find('expand').hidden = !image;
    find('placeholder').hidden = !!image;
    if (image && find('image').getAttribute('src') !== image) find('image').src = image;
    if (!image) {
      const messages = {
        idle: ['같은 프레임으로 비교', '비교를 실행하면 결과가 표시됩니다.'],
        queued: ['다른 모델의 실행을 기다립니다', '고정된 동일 프레임을 순서대로 처리합니다.'],
        loading: ['모델을 준비하고 있습니다', '첫 실행의 로딩 시간은 추론 시간과 별도입니다.'],
        running: ['관절을 추정하고 있습니다', id === 'egoforce' ? '손과 전완 모델을 실행 중입니다. CPU에서는 더 오래 걸릴 수 있습니다.' : '실제 추론 결과를 기다리고 있습니다.'],
        done: ['결과 이미지 없음', '아래 실행 상태와 JSON을 확인하세요.'],
        error: ['모델 실행을 완료하지 못했습니다', '아래 오류 내용을 확인하세요.'],
        skipped: ['모델 준비 대기', '준비된 다른 모델의 결과는 확인할 수 있습니다.'],
        cancelled: ['실행을 중지했습니다', '완료된 다른 모델의 결과는 유지됩니다.'],
      };
      const message = status === 'skipped' && job?.state === 'cancelled'
        ? ['비교 중지로 실행을 생략했습니다', '완료된 다른 모델의 결과는 유지됩니다.']
        : messages[status] || messages.idle;
      find('placeholder-title').textContent = message[0];
      find('placeholder-detail').textContent = message[1];
    }
    find('json').hidden = !complete;
    if (complete) { find('json').href = `/comparisons/${encodeURIComponent(job.id)}/result.json`; find('json').download = `${job.id}_results.json`; }
  }

  function renderJob(job) {
    state.display = job;
    for (const id of Object.keys(names)) renderCard(id, job);
    const status = job?.state || 'idle';
    $('job-state').dataset.state = status;
    put('job-state', state.historical ? `기록 · ${statusLabels[status] || status}` : statusLabels[status] || status);
    $('json-download').hidden = !job || !terminal(job);
    if (job) {
      const source = job.source || {};
      const dimensions = source.width && source.height ? ` · ${source.width} × ${source.height}` : '';
      put('job-caption', `${state.historical ? '저장된 비교' : '동일 프레임 비교'} · 프레임 ${source.frame_id ?? '—'}${dimensions} · ${job.id}`);
      $('json-download').href = `/comparisons/${encodeURIComponent(job.id)}/result.json`;
      $('json-download').download = `${job.id}_results.json`;
      if (source.image_url && !state.sourceEdited && state.previewKey !== `job:${job.id}`) {
        if (source.capture_id && Array.from($('source-select').options).some((option) => option.value === source.capture_id)) $('source-select').value = source.capture_id;
        if (Number.isFinite(job.draw_threshold)) { $('draw-threshold').value = job.draw_threshold; put('threshold-value', Number(job.draw_threshold).toFixed(2)); }
        setPreview(source.image_url, `job:${job.id}`, 'FIXED INPUT · SAME FRAME', `비교에 사용한 고정 프레임 ${source.frame_id ?? '—'}`);
        state.roi = job.roi || null;
        paintROI();
      }
      if (job.error) notice(typeof job.error === 'string' ? job.error : JSON.stringify(job.error));
    } else put('job-caption', '비교 실행 후 같은 원본 위의 관절을 확인할 수 있습니다.');
    $('latest-button').hidden = !state.historical;
    updateControls();
  }

  function updateControls() {
    const active = busy();
    $('run-button').disabled = active || !state.connected || state.historical;
    $('cancel-button').disabled = !active;
    $('cancel-button').textContent = '중지';
    for (const id of ['source-select', 'draw-threshold', 'roi-x1', 'roi-y1', 'roi-x2', 'roi-y2']) $(id).disabled = active || state.historical;
    $('clear-roi').disabled = active || state.historical || (!state.roi && !['x1', 'y1', 'x2', 'y2'].some((key) => $('roi-' + key).value !== ''));
    $('refresh-source').hidden = $('source-select').value !== 'live';
    $('refresh-source').disabled = active || state.historical;
    const repeatDisabled = $('source-select').value !== 'live' || state.historical;
    $('auto-repeat').disabled = repeatDisabled;
    $('auto-repeat').closest('label').classList.toggle('is-disabled', repeatDisabled);
    if (repeatDisabled) { $('auto-repeat').checked = false; clearTimeout(state.autoTimer); state.autoTimer = null; }
    $('roi-canvas').style.pointerEvents = active || state.historical ? 'none' : '';
    if (active) $('run-button').innerHTML = '<span aria-hidden="true">◌</span> 비교 진행 중';
    else $('run-button').innerHTML = '<span aria-hidden="true">▶</span> 세 모델 비교 실행';
  }

  function renderCaptures(captures) {
    const previous = $('source-select').value;
    const currentIds = Array.from($('source-select').options).slice(1).map((item) => item.value).join('|');
    if (currentIds === captures.map((item) => item.id).join('|')) return;
    $('source-select').replaceChildren(new Option('현재 카메라 프레임', 'live'));
    for (const capture of captures) $('source-select').add(new Option(`${capture.id} · 프레임 ${capture.frame_id ?? '—'}`, capture.id));
    $('source-select').value = previous === 'live' || captures.some((item) => item.id === previous) ? previous : 'live';
  }

  function renderHistory(history) {
    const key = history.map((job) => `${job.id}:${job.state}`).join('|') + ':' + (state.display?.id || '');
    if (state.historyKey === key) return;
    state.historyKey = key;
    if (!history.length) { $('history-list').innerHTML = '<p class="help">완료한 비교가 여기에 표시됩니다.</p>'; return; }
    const fragment = document.createDocumentFragment();
    for (const job of history.slice(0, 15)) {
      const button = document.createElement('button');
      button.className = 'history-item'; button.type = 'button'; button.setAttribute('aria-current', String(state.display?.id === job.id));
      const title = document.createElement('strong'); title.textContent = job.id;
      const caption = document.createElement('span'); caption.textContent = `프레임 ${job.source?.frame_id ?? '—'} · ${job.source?.capture_id || '카메라 입력'}`;
      const status = document.createElement('span'); status.className = 'history-status'; status.textContent = statusLabels[job.state] || job.state;
      button.append(title, caption, status);
      button.addEventListener('click', async () => {
        $('auto-repeat').checked = false; clearTimeout(state.autoTimer); state.autoTimer = null;
        try {
          const response = await api(`/api/compare/job/${encodeURIComponent(job.id)}`);
          state.historical = true; state.sourceEdited = false;
          renderJob(response.job || response);
          state.historyKey = null;
          renderHistory(history);
          notice();
        } catch (error) { notice(error.message); }
      });
      fragment.append(button);
    }
    $('history-list').replaceChildren(fragment);
  }

  function scheduleRepeat() {
    const job = state.latest;
    if (!$('auto-repeat').checked || state.historical || !terminal(job) || job.state !== 'done' || state.autoScheduledId === job.id || state.autoTimer) return;
    state.autoScheduledId = job.id;
    state.autoTimer = setTimeout(() => {
      state.autoTimer = null;
      if ($('auto-repeat').checked && !busy() && !state.historical && state.connected) runComparison();
    }, 2000);
  }

  async function poll() {
    if (state.pollBusy) return;
    state.pollBusy = true;
    try {
      const response = await api('/api/compare/state');
      state.connected = true;
      if (state.connectionError) { notice(); state.connectionError = false; }
      state.models = response.models || [];
      state.captures = response.captures || [];
      state.latest = response.job || null;
      renderCaptures(state.captures);
      if (!state.historical) renderJob(state.latest);
      renderHistory(response.history || []);
      $('server-state').dataset.state = 'ready'; put('server-state', '● 비교 서버 연결됨');
      put('updated-at', `${new Date().toLocaleTimeString('ko-KR', { hour12: false })} 업데이트`);
      updateControls(); scheduleRepeat();
      if (!state.previewKey) showSelectedSource();
    } catch (error) {
      state.connected = false; state.connectionError = true;
      $('server-state').dataset.state = 'error'; put('server-state', '● 서버 연결 확인 필요');
      notice(`서버에 연결하지 못했습니다. ${error.name === 'AbortError' ? '응답 시간이 초과되었습니다.' : error.message}`);
      updateControls();
    } finally { state.pollBusy = false; }
  }

  async function runComparison() {
    if (busy()) return;
    if (!state.roi && ['x1', 'y1', 'x2', 'y2'].some((key) => $('roi-' + key).value !== '')) { notice('팔 영역의 네 좌표를 모두 올바르게 입력하거나 초기화한 뒤 실행하세요.'); return; }
    clearTimeout(state.autoTimer); state.autoTimer = null;
    state.submitting = true; state.historical = false; state.sourceEdited = false; notice(); updateControls();
    const payload = { source: $('source-select').value, roi: state.roi, draw_threshold: Number($('draw-threshold').value) };
    try {
      const response = await api('/api/compare/run', payload);
      state.latest = response.job || response;
      renderJob(state.latest);
      state.autoScheduledId = null;
    } catch (error) {
      $('auto-repeat').checked = false;
      notice(`비교를 시작하지 못했습니다. ${error.message}`);
    } finally { state.submitting = false; updateControls(); poll(); }
  }

  $('source-preview').addEventListener('load', () => {
    $('source-preview').hidden = false;
    $('preview-empty').hidden = true;
    $('preview-frame').classList.remove('is-empty');
    const [width, height] = imageDimensions();
    $('preview-frame').style.aspectRatio = `${width} / ${height}`;
    paintROI(false);
  });
  $('source-preview').addEventListener('error', () => {
    $('source-preview').hidden = true;
    $('preview-empty').hidden = false;
    $('preview-frame').classList.add('is-empty');
    $('preview-empty').querySelector('strong').textContent = '입력 영상을 불러오지 못했습니다';
  });
  $('roi-canvas').addEventListener('pointerdown', (event) => {
    if (busy() || state.historical || $('source-preview').hidden || event.button !== 0) return;
    state.dragStart = pointerPoint(event);
    $('roi-canvas').setPointerCapture(event.pointerId);
  });
  $('roi-canvas').addEventListener('pointermove', (event) => {
    if (!state.dragStart) return;
    const end = pointerPoint(event), start = state.dragStart;
    state.roi = [Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])];
    paintROI();
  });
  function endDrag() {
    if (!state.dragStart) return;
    state.dragStart = null;
    if (state.roi && (state.roi[2] - state.roi[0] < 12 || state.roi[3] - state.roi[1] < 12)) state.roi = null;
    paintROI();
  }
  $('roi-canvas').addEventListener('pointerup', endDrag);
  $('roi-canvas').addEventListener('pointercancel', endDrag);
  function readNumericROI(event) {
    const elements = ['x1', 'y1', 'x2', 'y2'].map((value) => $('roi-' + value));
    if (elements.some((input) => input.value === '')) { state.roi = null; paintROI(false); return; }
    const [width, height] = imageDimensions();
    const roi = elements.map((input, index) => Math.round(Math.max(0, Math.min(index % 2 === 0 ? width : height, Number(input.value)))));
    if (roi[2] - roi[0] < 12 || roi[3] - roi[1] < 12) { state.roi = null; paintROI(false); if (event.type === 'change') notice('팔 영역은 가로와 세로가 12픽셀 이상이어야 합니다. X2·Y2를 X1·Y1보다 크게 입력하세요.'); return; }
    notice(); state.roi = roi; paintROI();
  }
  for (const key of ['x1', 'y1', 'x2', 'y2']) {
    $('roi-' + key).addEventListener('input', readNumericROI);
    $('roi-' + key).addEventListener('change', readNumericROI);
  }
  $('clear-roi').addEventListener('click', () => { state.roi = null; paintROI(); });
  $('source-select').addEventListener('change', () => { state.roi = null; state.historical = false; state.sourceEdited = true; state.previewKey = null; showSelectedSource(); paintROI(); updateControls(); });
  $('refresh-source').addEventListener('click', () => { state.sourceEdited = true; state.previewKey = null; showSelectedSource(); });
  $('draw-threshold').addEventListener('input', () => { put('threshold-value', Number($('draw-threshold').value).toFixed(2)); });
  $('run-button').addEventListener('click', runComparison);
  $('cancel-button').addEventListener('click', async () => {
    $('auto-repeat').checked = false; clearTimeout(state.autoTimer); state.autoTimer = null;
    $('cancel-button').disabled = true; put('cancel-button', '중지 요청 중');
    try { await api('/api/compare/cancel', {}); notice('중지를 요청했습니다. 완료된 결과는 유지됩니다.'); }
    catch (error) { notice(`중지 요청에 실패했습니다. ${error.message}`); }
    finally { poll(); }
  });
  $('auto-repeat').addEventListener('change', () => { if (!$('auto-repeat').checked) { clearTimeout(state.autoTimer); state.autoTimer = null; } else { state.autoScheduledId = null; scheduleRepeat(); } });
  $('latest-button').addEventListener('click', () => { state.historical = false; state.sourceEdited = false; state.historyKey = null; state.previewKey = null; renderJob(state.latest); notice(); poll(); });
  $('dialog-close').addEventListener('click', () => $('image-dialog').close());
  $('image-dialog').addEventListener('click', (event) => { if (event.target === $('image-dialog')) { const b = $('image-dialog').getBoundingClientRect(); if (event.clientX < b.left || event.clientX > b.right || event.clientY < b.top || event.clientY > b.bottom) $('image-dialog').close(); } });
  window.addEventListener('resize', () => paintROI(false));
  createCards();
  $('preview-frame').classList.add('is-empty');
  updateControls(); poll(); setInterval(poll, 1000);
})();
