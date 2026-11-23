/* ==========================================================================
   KINETIC-X // Industrial Pose Telemetry Frontend Controller
   Handles WebSocket GPU streaming, Webcam/Image/Video/Synthetic Demo modes,
   Mechanical HUD Skeleton Renderer, 2D Blueprint Radar & Telemetry Gauges.
   ========================================================================== */

(() => {
  "use strict";

  // ---- DOM Elements ----
  const el = {
    gpuName: document.getElementById("gpu-name"),
    vramText: document.getElementById("vram-text"),
    vramFill: document.getElementById("vram-fill"),
    infMs: document.getElementById("inf-ms"),
    rttMs: document.getElementById("rtt-ms"),
    fpsVal: document.getElementById("fps-val"),
    ledCuda: document.getElementById("led-cuda"),
    computeModeLabel: document.getElementById("compute-mode-label"),
    ledWs: document.getElementById("led-ws"),
    ledLock: document.getElementById("led-lock"),

    modeBtns: document.querySelectorAll(".mode-btn"),
    ctrlCamera: document.getElementById("ctrl-camera"),
    ctrlImage: document.getElementById("ctrl-image"),
    ctrlVideo: document.getElementById("ctrl-video"),

    btnCamToggle: document.getElementById("btn-cam-toggle"),
    camToggleText: document.getElementById("cam-toggle-text"),
    btnDemoToggle: document.getElementById("btn-demo-toggle"),
    btnFlipCam: document.getElementById("btn-flip-cam"),

    inputImage: document.getElementById("input-image"),
    imageDropzone: document.getElementById("image-dropzone"),
    btnRescanImage: document.getElementById("btn-rescan-image"),

    inputVideo: document.getElementById("input-video"),
    videoDropzone: document.getElementById("video-dropzone"),
    btnVidPlay: document.getElementById("btn-vid-play"),
    vidSeek: document.getElementById("vid-seek"),
    vidSpeed: document.getElementById("vid-speed"),

    engineSelect: document.getElementById("engine-select"),
    confSlider: document.getElementById("conf-slider"),
    confVal: document.getElementById("conf-val"),
    imgszSelect: document.getElementById("imgsz-select"),
    hwTierBadge: document.getElementById("hw-tier-badge"),
    netProfileSelect: document.getElementById("net-profile-select"),
    streamBwBadge: document.getElementById("stream-bw-badge"),

    btnTogglePose: document.getElementById("btn-toggle-pose"),
    poseStateTxt: document.getElementById("pose-state-txt"),
    btnToggleHands: document.getElementById("btn-toggle-hands"),
    handsStateTxt: document.getElementById("hands-state-txt"),
    handsTelemetry: document.getElementById("hands-telemetry"),

    chkSkeleton: document.getElementById("chk-skeleton"),
    chkAngles: document.getElementById("chk-angles"),
    chkBbox: document.getElementById("chk-bbox"),
    chkScanlines: document.getElementById("chk-scanlines"),
    chkMirror: document.getElementById("chk-mirror"),

    btnSnapPng: document.getElementById("btn-snap-png"),
    btnExportJson: document.getElementById("btn-export-json"),

    hudCanvas: document.getElementById("hud-canvas"),
    radarCanvas: document.getElementById("radar-canvas"),
    captureCanvas: document.getElementById("capture-canvas"),
    sourceVideo: document.getElementById("source-video"),
    scanlineOverlay: document.getElementById("scanline-overlay"),
    standbyOverlay: document.getElementById("standby-overlay"),

    vpRes: document.getElementById("vp-res"),
    vpTargets: document.getElementById("vp-targets"),
    vpClock: document.getElementById("vp-clock"),
    vpEngineBadge: document.getElementById("vp-engine-badge"),

    unitTabs: document.getElementById("unit-tabs"),
    postureState: document.getElementById("posture-state"),
    integrityVal: document.getElementById("integrity-val"),
    anglesList: document.getElementById("angles-list"),
    consoleLog: document.getElementById("console-log"),
  };

  const hudCtx = el.hudCanvas.getContext("2d");
  const radarCtx = el.radarCanvas.getContext("2d");
  const capCtx = el.captureCanvas.getContext("2d");

  // ---- Application State ----
  const state = {
    mode: "camera", // 'camera' | 'image' | 'video'
    enablePose: true,
    enableHands: true,
    camActive: false,
    demoActive: false,
    camStream: null,
    ws: null,
    wsBusy: false,
    rafId: null,
    lastPose: null,
    selectedUnitIdx: 0,
    vramTotalMb: 6144,
    skeletons: {
      coco17: {
        edges: [
          [0, 1, "head"], [0, 2, "head"], [1, 3, "head"], [2, 4, "head"],
          [5, 6, "torso"], [5, 11, "torso"], [6, 12, "torso"], [11, 12, "torso"],
          [5, 7, "left_arm"], [7, 9, "left_arm"],
          [6, 8, "right_arm"], [8, 10, "right_arm"],
          [11, 13, "left_leg"], [13, 15, "left_leg"],
          [12, 14, "right_leg"], [14, 16, "right_leg"],
        ],
      },
      mediapipe33: {
        edges: [
          [0, 2, "head"], [0, 5, "head"], [2, 7, "head"], [5, 8, "head"],
          [11, 12, "torso"], [11, 23, "torso"], [12, 24, "torso"], [23, 24, "torso"],
          [11, 13, "left_arm"], [13, 15, "left_arm"], [15, 19, "left_arm"],
          [12, 14, "right_arm"], [14, 16, "right_arm"], [16, 20, "right_arm"],
          [23, 25, "left_leg"], [25, 27, "left_leg"], [27, 29, "left_leg"], [27, 31, "left_leg"],
          [24, 26, "right_leg"], [26, 28, "right_leg"], [28, 30, "right_leg"], [28, 32, "right_leg"],
        ],
      },
      hand21: {
        edges: [
          [0, 1, "thumb"], [1, 2, "thumb"], [2, 3, "thumb"], [3, 4, "thumb"],
          [0, 5, "palm"], [5, 9, "palm"], [9, 13, "palm"], [13, 17, "palm"], [0, 17, "palm"],
          [5, 6, "index"], [6, 7, "index"], [7, 8, "index"],
          [9, 10, "middle"], [10, 11, "middle"], [11, 12, "middle"],
          [13, 14, "ring"], [14, 15, "ring"], [15, 16, "ring"],
          [17, 18, "pinky"], [18, 19, "pinky"], [19, 20, "pinky"],
        ],
      },
    },
    loadedImage: null,
    loadedImageFile: null,
    loadedVideoUrl: null,
    fpsFrames: 0,
    fpsLastTs: performance.now(),
    liveFps: 0,
    demoStartTs: performance.now(),
    facingMode: "user",
    recommendedStream: { capture_width: 480, jpeg_quality: 0.72, target_fps: 25 },
    lastFrameSentTs: 0,
    rttMs: 0,
    smoothedPose: null,
    wsReconnectTimer: null,
  };

  const HAND_GROUP_COLORS = {
    palm: "#94a3b8",
    thumb: "#f59e0b",
    index: "#00e5ff",
    middle: "#38bdf8",
    ring: "#22c55e",
    pinky: "#a78bfa",
  };

  const ANGLE_KEYS = [
    { key: "L_ELBOW", label: "左肘 L-ELBOW", max: 180 },
    { key: "R_ELBOW", label: "右肘 R-ELBOW", max: 180 },
    { key: "L_SHOULDER", label: "左肩 L-SHLDR", max: 180 },
    { key: "R_SHOULDER", label: "右肩 R-SHLDR", max: 180 },
    { key: "L_KNEE", label: "左膝 L-KNEE", max: 180 },
    { key: "R_KNEE", label: "右膝 R-KNEE", max: 180 },
    { key: "L_HIP", label: "左髋 L-HIP", max: 180 },
    { key: "R_HIP", label: "右髋 R-HIP", max: 180 },
    { key: "SPINE_TILT", label: "脊柱倾角 SPINE", max: 90 },
  ];

  const GROUP_COLORS = {
    head: "#94a3b8",
    torso: "#f59e0b",
    left_arm: "#00e5ff",
    right_arm: "#f59e0b",
    left_leg: "#00e5ff",
    right_leg: "#f59e0b",
  };

  // ---- Console Logger ----
  function logEvent(msg, level = "info") {
    const now = new Date();
    const ts = now.toTimeString().slice(0, 8);
    const line = document.createElement("div");
    line.className = "log-line";
    const cls = level === "ok" ? "log-ok" : level === "warn" ? "log-warn" : level === "err" ? "log-err" : "";
    line.innerHTML = `<span class="log-ts">[${ts}]</span><span class="${cls}">${msg}</span>`;
    el.consoleLog.appendChild(line);
    while (el.consoleLog.children.length > 60) {
      el.consoleLog.removeChild(el.consoleLog.firstChild);
    }
    el.consoleLog.scrollTop = el.consoleLog.scrollHeight;
  }

  // ---- Build Angle Gauge Rows ----
  function initAngleRows() {
    el.anglesList.innerHTML = "";
    for (const item of ANGLE_KEYS) {
      const row = document.createElement("div");
      row.className = "angle-row";
      row.innerHTML = `
        <span class="angle-name">${item.label}</span>
        <div class="angle-bar-bg"><div class="angle-bar-fill" id="bar-${item.key}"></div></div>
        <span class="angle-val" id="val-${item.key}">--°</span>
      `;
      el.anglesList.appendChild(row);
    }
  }

  // ---- Live Clock ----
  setInterval(() => {
    el.vpClock.textContent = new Date().toTimeString().slice(0, 8);
  }, 500);

  // ---- System Info Fetch (Auto Hardware Profile) ----
  async function fetchSystemInfo() {
    try {
      const res = await fetch("/api/system");
      const info = await res.json();
      el.gpuName.textContent = info.gpu_name || "UNKNOWN";
      state.vramTotalMb = info.vram_total_mb || 0;
      if (info.cuda_available && state.vramTotalMb > 0) {
        updateVram(info.vram_used_mb || 0, state.vramTotalMb);
      } else {
        el.vramText.textContent = `RAM ${info.ram_gb || "--"} GB`;
        el.vramFill.style.width = `${Math.min(100, info.cpu_load_pct || 25)}%`;
      }

      const modeTag = info.compute_mode || (info.cuda_available ? "CUDA" : "CPU");
      if (el.computeModeLabel) el.computeModeLabel.textContent = modeTag;
      el.ledCuda.className = "led " + (info.cuda_available ? "on-green" : "on-cyan");

      if (info.hardware_tier && el.hwTierBadge) {
        el.hwTierBadge.textContent = info.hardware_tier;
      }
      if (info.recommended_imgsz) {
        const recStr = String(info.recommended_imgsz);
        if ([...el.imgszSelect.options].some((o) => o.value === recStr)) {
          el.imgszSelect.value = recStr;
        }
      }
      if (info.recommended_stream) {
        state.recommendedStream = info.recommended_stream;
      }
      if (info.skeletons) {
        state.skeletons = info.skeletons;
      }

      // Populate / disable unavailable engines
      if (info.engines) {
        for (const opt of el.engineSelect.options) {
          const meta = info.engines[opt.value];
          if (meta && !meta.available) {
            opt.disabled = true;
            opt.textContent = opt.textContent + " [未安装]";
          }
        }
      }

      el.engineSelect.value = info.active_engine || "yolo11n-pose";
      el.vpEngineBadge.textContent = `CORE: ${el.engineSelect.value.toUpperCase()} [${modeTag}]`;

      if (info.hand_ready === false) {
        state.enableHands = false;
        el.btnToggleHands.disabled = true;
        syncModuleButtons();
        logEvent("HAND 21-PT ENGINE UNAVAILABLE", "warn");
      } else {
        logEvent("HAND 21-PT LANDMARKER ONLINE", "ok");
      }

      logEvent(
        `AUTO-CONFIG [${info.hardware_tier || modeTag}] // ${info.gpu_name} // TENSOR ${el.imgszSelect.value}PX`,
        info.cuda_available ? "ok" : "warn"
      );
    } catch (err) {
      logEvent(`SYSTEM INIT ERROR: ${err.message}`, "err");
    }
  }

  function syncModuleButtons() {
    el.btnTogglePose.classList.toggle("active-amber", state.enablePose);
    el.btnTogglePose.setAttribute("aria-pressed", String(state.enablePose));
    el.poseStateTxt.textContent = state.enablePose ? "ON" : "OFF";

    el.btnToggleHands.classList.toggle("active-cyan", state.enableHands);
    el.btnToggleHands.setAttribute("aria-pressed", String(state.enableHands));
    el.handsStateTxt.textContent = state.enableHands ? "ON" : "OFF";
  }

  function updateVram(used, total) {
    const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0;
    el.vramText.textContent = `${Math.round(used)} / ${Math.round(total)} MB`;
    el.vramFill.style.width = `${pct.toFixed(1)}%`;
  }

  // ---- WebSocket Connection (Auto-Reconnect + RTT Monitoring) ----
  function ensureWebSocket() {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    if (state.wsReconnectTimer) {
      clearTimeout(state.wsReconnectTimer);
      state.wsReconnectTimer = null;
    }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/pose`);
    state.ws = ws;
    state.wsBusy = false;

    ws.onopen = () => {
      el.ledWs.className = "led on-cyan";
      sendWsConfig();
      logEvent("WS TELEMETRY LINK ESTABLISHED", "ok");
    };

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.type === "pose") {
          state.wsBusy = false;
          if (state.lastFrameSentTs > 0) {
            const totalRoundTrip = performance.now() - state.lastFrameSentTs;
            const netOnly = Math.max(1, totalRoundTrip - (data.inference_ms || 0));
            state.rttMs = state.rttMs === 0 ? netOnly : state.rttMs * 0.7 + netOnly * 0.3;
            if (el.rttMs) {
              el.rttMs.textContent = `${Math.round(state.rttMs)} ms`;
              el.rttMs.className =
                "tele-val " + (state.rttMs < 80 ? "accent-cyan" : state.rttMs < 180 ? "accent-amber" : "");
            }
          }
          handlePoseTelemetry(data);
        }
      } catch (_) {
        state.wsBusy = false;
      }
    };

    ws.onclose = () => {
      el.ledWs.className = "led";
      state.wsBusy = false;
      if (state.camActive || (state.mode === "video" && state.loadedVideoUrl)) {
        logEvent("WS LINK DROPPED — RECONNECTING IN 2S...", "warn");
        state.wsReconnectTimer = setTimeout(ensureWebSocket, 2000);
      }
    };

    ws.onerror = () => {
      el.ledWs.className = "led on-red";
      state.wsBusy = false;
    };
  }

  // ---- Adaptive External Network Stream Parameters ----
  function getAdaptiveStreamParams() {
    const profile = el.netProfileSelect ? el.netProfileSelect.value : "auto";
    if (profile === "wan") {
      return { width: 360, quality: 0.58, label: "WAN 360P" };
    }
    if (profile === "lan") {
      return { width: 640, quality: 0.78, label: "LAN 640P" };
    }
    // AUTO: adapt capture size & JPEG quality to live network RTT & server hardware
    const baseW = state.recommendedStream.capture_width || 480;
    const baseQ = state.recommendedStream.jpeg_quality || 0.72;
    if (state.rttMs > 160) {
      return { width: 320, quality: 0.55, label: "AUTO-WAN 320P" };
    }
    if (state.rttMs > 85) {
      return { width: Math.min(baseW, 400), quality: 0.62, label: "AUTO-MID 400P" };
    }
    return { width: baseW, quality: baseQ, label: `AUTO ${baseW}P` };
  }

  // ---- EMA Temporal Keypoint Smoothing for Butter-Smooth 60FPS HUD ----
  function smoothPoseData(rawPose) {
    if (!rawPose) return null;
    const prev = state.smoothedPose;
    if (
      !prev ||
      prev.topology !== rawPose.topology ||
      (prev.persons?.length || 0) !== (rawPose.persons?.length || 0) ||
      (prev.hands?.length || 0) !== (rawPose.hands?.length || 0)
    ) {
      state.smoothedPose = rawPose;
      return rawPose;
    }

    const alpha = state.rttMs > 100 ? 0.52 : 0.68;
    const blendPts = (curPts, prevPts) =>
      curPts.map((kp, i) => {
        const p = prevPts[i];
        if (!p || kp.conf < 0.2 || p.conf < 0.2) return kp;
        // Skip smoothing on large sudden jumps (e.g. target swap)
        if (Math.hypot(kp.x - p.x, kp.y - p.y) > 0.22) return kp;
        return {
          ...kp,
          x: p.x + (kp.x - p.x) * alpha,
          y: p.y + (kp.y - p.y) * alpha,
        };
      });

    const smoothed = {
      ...rawPose,
      persons: (rawPose.persons || []).map((person, idx) => ({
        ...person,
        keypoints: blendPts(person.keypoints || [], prev.persons[idx]?.keypoints || []),
      })),
      hands: (rawPose.hands || []).map((hand, idx) => ({
        ...hand,
        keypoints: blendPts(hand.keypoints || [], prev.hands[idx]?.keypoints || []),
      })),
    };
    state.smoothedPose = smoothed;
    return smoothed;
  }

  function sendWsConfig() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(
        JSON.stringify({
          type: "config",
          conf: parseFloat(el.confSlider.value),
          engine: el.engineSelect.value,
          imgsz: parseInt(el.imgszSelect.value, 10),
          enable_pose: state.enablePose,
          enable_hands: state.enableHands,
        })
      );
    }
  }

  function handlePoseTelemetry(data) {
    state.lastPose = smoothPoseData(data);
    el.infMs.textContent = `${data.inference_ms.toFixed(1)} ms`;
    if (data.vram_used_mb > 0 && state.vramTotalMb > 0) {
      updateVram(data.vram_used_mb, state.vramTotalMb);
    }

    // Track real stream FPS
    state.fpsFrames += 1;
    const now = performance.now();
    if (now - state.fpsLastTs >= 500) {
      state.liveFps = (state.fpsFrames * 1000) / (now - state.fpsLastTs);
      el.fpsVal.textContent = `${state.liveFps.toFixed(1)} FPS`;
      state.fpsFrames = 0;
      state.fpsLastTs = now;
    }

    const bodyCount = data.person_count || 0;
    const handCount = data.hand_count || (data.hands ? data.hands.length : 0);
    el.vpTargets.textContent = `BODY: ${String(bodyCount).padStart(2, "0")} | HAND: ${String(handCount).padStart(2, "0")}`;
    el.ledLock.className = "led " + (bodyCount > 0 || handCount > 0 ? "on-amber" : "");

    updateRightTelemetry(data);
  }

  // ---- Right Rack Telemetry & 2D Blueprint Radar ----
  function updateHandsTelemetryUI(hands) {
    if (!state.enableHands) {
      el.handsTelemetry.innerHTML = `<div class="hand-empty">HAND DETECTOR // DISABLED</div>`;
      return;
    }
    if (!hands || hands.length === 0) {
      el.handsTelemetry.innerHTML = `<div class="hand-empty">HAND 21-PT // NO TARGET</div>`;
      return;
    }
    el.handsTelemetry.innerHTML = "";
    const labels = [
      ["THUMB", "T"],
      ["INDEX", "I"],
      ["MIDDLE", "M"],
      ["RING", "R"],
      ["PINKY", "P"],
    ];
    for (const h of hands) {
      const kin = h.kinematics || {};
      const fMap = kin.fingers || {};
      const pips = labels
        .map(([k, ch]) => `<span class="f-pip ${fMap[k] ? "up" : ""}">${ch}</span>`)
        .join("");
      const card = document.createElement("div");
      card.className = "hand-card";
      card.innerHTML = `
        <span class="hand-id">${h.hand_id}</span>
        <span class="hand-gesture">${kin.gesture || "LOCKED"}</span>
        <span class="finger-pips">${pips}</span>
      `;
      el.handsTelemetry.appendChild(card);
    }
  }

  function updateRightTelemetry(data) {
    updateHandsTelemetryUI(data.hands || []);

    const persons = data.persons || [];
    if (state.selectedUnitIdx >= persons.length) {
      state.selectedUnitIdx = 0;
    }

    // Unit selector tabs
    el.unitTabs.innerHTML = "";
    if (!state.enablePose) {
      el.unitTabs.innerHTML = `<span class="unit-pill">POSE OFF</span>`;
      el.postureState.textContent = "DISABLED";
      el.integrityVal.textContent = "INT: --";
      for (const item of ANGLE_KEYS) {
        document.getElementById(`bar-${item.key}`).style.width = "0%";
        document.getElementById(`val-${item.key}`).textContent = "--°";
      }
      drawWireframeRadar(null, data.topology);
      return;
    }

    if (persons.length === 0) {
      el.unitTabs.innerHTML = `<span class="unit-pill">NO LOCK</span>`;
      el.postureState.textContent = "NO TARGET";
      el.integrityVal.textContent = "INT: 0%";
      for (const item of ANGLE_KEYS) {
        document.getElementById(`bar-${item.key}`).style.width = "0%";
        document.getElementById(`val-${item.key}`).textContent = "--°";
      }
      drawWireframeRadar(null, data.topology);
      return;
    }

    persons.forEach((p, idx) => {
      const pill = document.createElement("button");
      pill.className = "unit-pill" + (idx === state.selectedUnitIdx ? " active" : "");
      pill.textContent = p.unit_id;
      pill.onclick = () => {
        state.selectedUnitIdx = idx;
        updateRightTelemetry(state.lastPose);
      };
      el.unitTabs.appendChild(pill);
    });

    const activePerson = persons[state.selectedUnitIdx];
    const kin = activePerson.kinematics || {};
    el.postureState.textContent = kin.posture || "UNKNOWN";
    el.integrityVal.textContent = `INT: ${kin.integrity_pct ?? 0}%`;

    const angles = kin.angles || {};
    for (const item of ANGLE_KEYS) {
      const val = angles[item.key];
      const bar = document.getElementById(`bar-${item.key}`);
      const txt = document.getElementById(`val-${item.key}`);
      if (val === null || val === undefined) {
        bar.style.width = "0%";
        txt.textContent = "--°";
      } else {
        const pct = Math.min(100, (val / item.max) * 100);
        bar.style.width = `${pct.toFixed(1)}%`;
        txt.textContent = `${val.toFixed(1)}°`;
      }
    }

    drawWireframeRadar(activePerson, data.topology || "coco17");
  }

  function drawWireframeRadar(person, topology) {
    const w = el.radarCanvas.width;
    const h = el.radarCanvas.height;
    radarCtx.clearRect(0, 0, w, h);

    // Blueprint grid
    radarCtx.strokeStyle = "rgba(0, 229, 255, 0.08)";
    radarCtx.lineWidth = 1;
    for (let x = 0; x <= w; x += 20) {
      radarCtx.beginPath();
      radarCtx.moveTo(x, 0);
      radarCtx.lineTo(x, h);
      radarCtx.stroke();
    }
    for (let y = 0; y <= h; y += 20) {
      radarCtx.beginPath();
      radarCtx.moveTo(0, y);
      radarCtx.lineTo(w, y);
      radarCtx.stroke();
    }

    // Center crosshair
    radarCtx.strokeStyle = "rgba(245, 158, 11, 0.25)";
    radarCtx.beginPath();
    radarCtx.moveTo(w / 2, 8);
    radarCtx.lineTo(w / 2, h - 8);
    radarCtx.moveTo(8, h / 2);
    radarCtx.lineTo(w - 8, h / 2);
    radarCtx.stroke();

    if (!person) {
      radarCtx.fillStyle = "#4c5a6d";
      radarCtx.font = "10px 'JetBrains Mono', monospace";
      radarCtx.textAlign = "center";
      radarCtx.fillText("AWAITING KINEMATIC LOCK", w / 2, h / 2 + 4);
      return;
    }

    const minConf = parseFloat(el.confSlider.value);
    const kps = person.keypoints || [];
    const validPts = kps.filter((k) => k.conf >= minConf);
    if (validPts.length < 2) return;

    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (const p of validPts) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const spanX = Math.max(0.12, maxX - minX);
    const spanY = Math.max(0.12, maxY - minY);
    const pad = 24;
    const scale = Math.min((w - pad * 2) / spanX, (h - pad * 2) / spanY);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;

    const proj = (kp) => ({
      x: w / 2 + (kp.x - cx) * scale,
      y: h / 2 + (kp.y - cy) * scale,
    });

    const edges = (state.skeletons[topology] || state.skeletons.coco17).edges;
    for (const [i, j, grp] of edges) {
      if (i >= kps.length || j >= kps.length) continue;
      if (kps[i].conf < minConf || kps[j].conf < minConf) continue;
      const p1 = proj(kps[i]);
      const p2 = proj(kps[j]);
      radarCtx.strokeStyle = GROUP_COLORS[grp] || "#00e5ff";
      radarCtx.lineWidth = 2.2;
      radarCtx.beginPath();
      radarCtx.moveTo(p1.x, p1.y);
      radarCtx.lineTo(p2.x, p2.y);
      radarCtx.stroke();
    }

    for (const kp of validPts) {
      const pt = proj(kp);
      radarCtx.fillStyle = "#080a0d";
      radarCtx.strokeStyle = "#f59e0b";
      radarCtx.lineWidth = 1.5;
      radarCtx.beginPath();
      radarCtx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
      radarCtx.fill();
      radarCtx.stroke();
    }
  }

  // ---- Mechanical HUD Skeleton Canvas Renderer ----
  function drawMechanicalHUD(poseData, mirror = false) {
    const w = el.hudCanvas.width;
    const h = el.hudCanvas.height;
    if (!w || !h) return;

    // Corner HUD L-Brackets on Viewport
    drawViewportReticles(hudCtx, w, h);

    if (!poseData) return;

    const minConf = parseFloat(el.confSlider.value);
    const showSkel = el.chkSkeleton.checked;
    const showAngles = el.chkAngles.checked;
    const showBbox = el.chkBbox.checked;
    const topology = poseData.topology || "coco17";
    const edges = (state.skeletons[topology] || state.skeletons.coco17).edges;

    const mapX = (nx) => (mirror ? (1 - nx) * w : nx * w);
    const mapY = (ny) => ny * h;

    const persons = state.enablePose ? (poseData.persons || []) : [];
    for (const person of persons) {
      const kps = person.keypoints || [];

      // 1. Target Lock Bounding Brackets
      if (showBbox && person.bbox) {
        const bx1 = mirror ? (1 - person.bbox.x2) * w : person.bbox.x1 * w;
        const bx2 = mirror ? (1 - person.bbox.x1) * w : person.bbox.x2 * w;
        const by1 = person.bbox.y1 * h;
        const by2 = person.bbox.y2 * h;
        const bw = bx2 - bx1;
        const bh = by2 - by1;
        const corner = Math.min(22, bw * 0.22, bh * 0.22);

        hudCtx.strokeStyle = "rgba(245, 158, 11, 0.85)";
        hudCtx.lineWidth = 2;
        hudCtx.beginPath();
        // TL
        hudCtx.moveTo(bx1, by1 + corner); hudCtx.lineTo(bx1, by1); hudCtx.lineTo(bx1 + corner, by1);
        // TR
        hudCtx.moveTo(bx2 - corner, by1); hudCtx.lineTo(bx2, by1); hudCtx.lineTo(bx2, by1 + corner);
        // BL
        hudCtx.moveTo(bx1, by2 - corner); hudCtx.lineTo(bx1, by2); hudCtx.lineTo(bx1 + corner, by2);
        // BR
        hudCtx.moveTo(bx2 - corner, by2); hudCtx.lineTo(bx2, by2); hudCtx.lineTo(bx2, by2 - corner);
        hudCtx.stroke();

        // Unit Header Tag
        const tag = `${person.unit_id} // ${(person.det_conf * 100).toFixed(0)}% // ${person.kinematics?.posture || ""}`;
        hudCtx.font = "bold 11px 'JetBrains Mono', monospace";
        const tw = hudCtx.measureText(tag).width + 12;
        const ty = Math.max(20, by1 - 6);
        hudCtx.fillStyle = "rgba(11, 14, 18, 0.82)";
        hudCtx.fillRect(bx1, ty - 15, tw, 17);
        hudCtx.fillStyle = "#f59e0b";
        hudCtx.fillRect(bx1, ty - 15, 3, 17);
        hudCtx.fillStyle = "#e2e8f0";
        hudCtx.fillText(tag, bx1 + 7, ty - 3);
      }

      // 2. Mechanical Hydraulic Struts / Skeleton Pistons
      if (showSkel) {
        for (const [i, j, grp] of edges) {
          if (i >= kps.length || j >= kps.length) continue;
          const a = kps[i], b = kps[j];
          if (a.conf < minConf || b.conf < minConf) continue;

          const x1 = mapX(a.x), y1 = mapY(a.y);
          const x2 = mapX(b.x), y2 = mapY(b.y);
          const color = GROUP_COLORS[grp] || "#00e5ff";

          // Outer dark steel casing
          hudCtx.strokeStyle = "rgba(8, 10, 14, 0.85)";
          hudCtx.lineWidth = 7;
          hudCtx.lineCap = "butt";
          hudCtx.beginPath();
          hudCtx.moveTo(x1, y1);
          hudCtx.lineTo(x2, y2);
          hudCtx.stroke();

          // Inner illuminated hydraulic piston core
          hudCtx.strokeStyle = color;
          hudCtx.lineWidth = 2.8;
          hudCtx.beginPath();
          hudCtx.moveTo(x1, y1);
          hudCtx.lineTo(x2, y2);
          hudCtx.stroke();

          // Midpoint mechanical collar tick
          const mx = (x1 + x2) / 2;
          const my = (y1 + y2) / 2;
          const angle = Math.atan2(y2 - y1, x2 - x1) + Math.PI / 2;
          hudCtx.strokeStyle = "#ffffff";
          hudCtx.lineWidth = 1.4;
          hudCtx.beginPath();
          hudCtx.moveTo(mx - Math.cos(angle) * 4, my - Math.sin(angle) * 4);
          hudCtx.lineTo(mx + Math.cos(angle) * 4, my + Math.sin(angle) * 4);
          hudCtx.stroke();
        }

        // Precision Bearing Rings at Joints
        for (const kp of kps) {
          if (kp.conf < minConf) continue;
          const x = mapX(kp.x);
          const y = mapY(kp.y);
          const ringColor = kp.conf >= 0.7 ? "#00e5ff" : kp.conf >= 0.45 ? "#f59e0b" : "#ef4444";

          // Outer bearing casing
          hudCtx.beginPath();
          hudCtx.arc(x, y, 6.5, 0, Math.PI * 2);
          hudCtx.fillStyle = "rgba(8, 10, 14, 0.9)";
          hudCtx.fill();
          hudCtx.strokeStyle = ringColor;
          hudCtx.lineWidth = 1.8;
          hudCtx.stroke();

          // Inner core pin
          hudCtx.beginPath();
          hudCtx.arc(x, y, 2.2, 0, Math.PI * 2);
          hudCtx.fillStyle = "#ffffff";
          hudCtx.fill();
        }
      }

      // 3. Live Protractor Angle Arcs at Elbows & Knees
      if (showAngles && person.kinematics?.angles) {
        const triplets =
          topology === "mediapipe33"
            ? [
                [11, 13, 15, "L_ELBOW"],
                [12, 14, 16, "R_ELBOW"],
                [23, 25, 27, "L_KNEE"],
                [24, 26, 28, "R_KNEE"],
              ]
            : [
                [5, 7, 9, "L_ELBOW"],
                [6, 8, 10, "R_ELBOW"],
                [11, 13, 15, "L_KNEE"],
                [12, 14, 16, "R_KNEE"],
              ];

        for (const [ia, ib, ic, key] of triplets) {
          const deg = person.kinematics.angles[key];
          if (deg === null || deg === undefined) continue;
          if (ia >= kps.length || ib >= kps.length || ic >= kps.length) continue;
          const a = kps[ia], b = kps[ib], c = kps[ic];
          if (a.conf < minConf || b.conf < minConf || c.conf < minConf) continue;

          const ax = mapX(a.x), ay = mapY(a.y);
          const bx = mapX(b.x), by = mapY(b.y);
          const cx = mapX(c.x), cy = mapY(c.y);

          const ang1 = Math.atan2(ay - by, ax - bx);
          const ang2 = Math.atan2(cy - by, cx - bx);
          let diff = ang2 - ang1;
          while (diff > Math.PI) diff -= Math.PI * 2;
          while (diff < -Math.PI) diff += Math.PI * 2;

          // Protractor arc
          hudCtx.beginPath();
          hudCtx.arc(bx, by, 18, ang1, ang1 + diff, diff < 0);
          hudCtx.strokeStyle = "rgba(245, 158, 11, 0.9)";
          hudCtx.lineWidth = 2;
          hudCtx.stroke();

          // Degree badge
          const label = `${Math.round(deg)}°`;
          hudCtx.font = "bold 10px 'JetBrains Mono', monospace";
          const lw = hudCtx.measureText(label).width + 6;
          hudCtx.fillStyle = "rgba(8, 10, 14, 0.85)";
          hudCtx.fillRect(bx + 12, by - 14, lw, 14);
          hudCtx.fillStyle = "#00e5ff";
          hudCtx.fillText(label, bx + 15, by - 4);
        }
      }
    }

    // 4. Mechanical 21-Point Hand Skeleton Overlay
    const hands = state.enableHands ? (poseData.hands || []) : [];
    const handEdges = (state.skeletons.hand21 || { edges: [] }).edges;

    for (const hand of hands) {
      const kps = hand.keypoints || [];
      if (kps.length < 21) continue;

      // Hand Lock Brackets (Cyan)
      if (showBbox && hand.bbox) {
        const bx1 = mirror ? (1 - hand.bbox.x2) * w : hand.bbox.x1 * w;
        const bx2 = mirror ? (1 - hand.bbox.x1) * w : hand.bbox.x2 * w;
        const by1 = hand.bbox.y1 * h;
        const by2 = hand.bbox.y2 * h;
        const bw = bx2 - bx1;
        const bh = by2 - by1;
        const c = Math.min(14, bw * 0.25, bh * 0.25);

        hudCtx.strokeStyle = "rgba(0, 229, 255, 0.85)";
        hudCtx.lineWidth = 1.6;
        hudCtx.beginPath();
        hudCtx.moveTo(bx1, by1 + c); hudCtx.lineTo(bx1, by1); hudCtx.lineTo(bx1 + c, by1);
        hudCtx.moveTo(bx2 - c, by1); hudCtx.lineTo(bx2, by1); hudCtx.lineTo(bx2, by1 + c);
        hudCtx.moveTo(bx1, by2 - c); hudCtx.lineTo(bx1, by2); hudCtx.lineTo(bx1 + c, by2);
        hudCtx.moveTo(bx2 - c, by2); hudCtx.lineTo(bx2, by2); hudCtx.lineTo(bx2, by2 - c);
        hudCtx.stroke();

        const gesture = hand.kinematics?.gesture || "LOCKED";
        const tag = `${hand.hand_id} // ${gesture}`;
        hudCtx.font = "bold 10px 'JetBrains Mono', monospace";
        const tw = hudCtx.measureText(tag).width + 10;
        const ty = Math.max(18, by1 - 4);
        hudCtx.fillStyle = "rgba(8, 10, 14, 0.85)";
        hudCtx.fillRect(bx1, ty - 13, tw, 15);
        hudCtx.fillStyle = "#00e5ff";
        hudCtx.fillRect(bx1, ty - 13, 3, 15);
        hudCtx.fillStyle = "#e2e8f0";
        hudCtx.fillText(tag, bx1 + 6, ty - 2);
      }

      // Articulated Finger Struts
      if (showSkel) {
        for (const [i, j, grp] of handEdges) {
          const a = kps[i], b = kps[j];
          const x1 = mapX(a.x), y1 = mapY(a.y);
          const x2 = mapX(b.x), y2 = mapY(b.y);
          const col = HAND_GROUP_COLORS[grp] || "#00e5ff";

          hudCtx.strokeStyle = "rgba(8, 10, 14, 0.8)";
          hudCtx.lineWidth = 4.2;
          hudCtx.beginPath();
          hudCtx.moveTo(x1, y1);
          hudCtx.lineTo(x2, y2);
          hudCtx.stroke();

          hudCtx.strokeStyle = col;
          hudCtx.lineWidth = 1.8;
          hudCtx.beginPath();
          hudCtx.moveTo(x1, y1);
          hudCtx.lineTo(x2, y2);
          hudCtx.stroke();
        }

        // Knuckle Bearings & Fingertip Diamonds
        for (const kp of kps) {
          const x = mapX(kp.x);
          const y = mapY(kp.y);
          const isTip = kp.id > 0 && kp.id % 4 === 0;
          if (isTip) {
            // Fingertip diamond reticle
            const r = 5;
            hudCtx.beginPath();
            hudCtx.moveTo(x, y - r);
            hudCtx.lineTo(x + r, y);
            hudCtx.lineTo(x, y + r);
            hudCtx.lineTo(x - r, y);
            hudCtx.closePath();
            hudCtx.fillStyle = "#080a0d";
            hudCtx.fill();
            hudCtx.strokeStyle = "#f59e0b";
            hudCtx.lineWidth = 1.5;
            hudCtx.stroke();
          } else {
            hudCtx.beginPath();
            hudCtx.arc(x, y, kp.id === 0 ? 4.5 : 3, 0, Math.PI * 2);
            hudCtx.fillStyle = "#080a0d";
            hudCtx.fill();
            hudCtx.strokeStyle = "#00e5ff";
            hudCtx.lineWidth = 1.3;
            hudCtx.stroke();
          }
        }
      }
    }
  }

  function drawViewportReticles(ctx, w, h) {
    const m = 14;
    const len = 18;
    ctx.strokeStyle = "rgba(0, 229, 255, 0.4)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    // Corners
    ctx.moveTo(m, m + len); ctx.lineTo(m, m); ctx.lineTo(m + len, m);
    ctx.moveTo(w - m - len, m); ctx.lineTo(w - m, m); ctx.lineTo(w - m, m + len);
    ctx.moveTo(m, h - m - len); ctx.lineTo(m, h - m); ctx.lineTo(m + len, h - m);
    ctx.moveTo(w - m - len, h - m); ctx.lineTo(w - m, h - m); ctx.lineTo(w - m, h - m - len);
    ctx.stroke();
  }

  // ---- Stop All Active Loops ----
  function stopActiveLoops() {
    if (state.rafId) {
      cancelAnimationFrame(state.rafId);
      state.rafId = null;
    }
    if (state.camStream) {
      state.camStream.getTracks().forEach((t) => t.stop());
      state.camStream = null;
    }
    state.camActive = false;
    state.demoActive = false;
    el.btnCamToggle.classList.remove("active");
    el.camToggleText.textContent = "启动光学采集 // ENGAGE CAM";
    el.btnDemoToggle.classList.remove("active");
    el.sourceVideo.pause();
  }

  // ---- Mode Switching ----
  el.modeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const newMode = btn.dataset.mode;
      if (newMode === state.mode) return;
      stopActiveLoops();
      state.mode = newMode;

      el.modeBtns.forEach((b) => b.classList.toggle("active", b.dataset.mode === newMode));
      el.ctrlCamera.classList.toggle("hidden", newMode !== "camera");
      el.ctrlImage.classList.toggle("hidden", newMode !== "image");
      el.ctrlVideo.classList.toggle("hidden", newMode !== "video");

      el.standbyOverlay.classList.remove("hidden");
      hudCtx.clearRect(0, 0, el.hudCanvas.width, el.hudCanvas.height);
      logEvent(`SWITCHED INPUT MODE -> ${newMode.toUpperCase()}`);
    });
  });

  // ---- MODE 01: LIVE WEBCAM ----
  async function startCameraStream() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const isSecure = window.isSecureContext;
      const msg = isSecure
        ? "当前浏览器不支持摄像头 API，请使用 Chrome/Edge/Safari。"
        : "外部网络通过 HTTP 访问时浏览器会安全禁用摄像头，请为服务器配置 HTTPS（或使用「图像解析 / 视频解析 / 虚拟机械人测试」模式）。";
      logEvent("CAMERA BLOCKED: HTTPS REQUIRED FOR REMOTE ORIGIN", "err");
      alert(msg);
      return;
    }

    stopActiveLoops();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 960 },
          height: { ideal: 540 },
          facingMode: { ideal: state.facingMode },
        },
        audio: false,
      });
      state.camStream = stream;
      state.camActive = true;
      state.smoothedPose = null;
      el.sourceVideo.srcObject = stream;
      await el.sourceVideo.play();

      el.btnCamToggle.classList.add("active");
      el.camToggleText.textContent = "紧急停机 // HALT CAM";
      el.standbyOverlay.classList.add("hidden");

      ensureWebSocket();
      logEvent(`OPTICAL CAMERA ENGAGED (${state.facingMode.toUpperCase()})`, "ok");
      runLiveVideoLoop(true);
    } catch (err) {
      logEvent(`CAMERA ACCESS FAILED: ${err.message}`, "err");
      alert("无法访问摄像头，请检查权限（外网设备需 HTTPS），或点击「虚拟机械人测试」/「视频解析」。");
    }
  }

  el.btnCamToggle.addEventListener("click", async () => {
    if (state.camActive) {
      stopActiveLoops();
      el.standbyOverlay.classList.remove("hidden");
      logEvent("OPTICAL CAMERA HALTED", "warn");
      return;
    }
    await startCameraStream();
  });

  if (el.btnFlipCam) {
    el.btnFlipCam.addEventListener("click", async () => {
      state.facingMode = state.facingMode === "user" ? "environment" : "user";
      el.chkMirror.checked = state.facingMode === "user";
      logEvent(`CAMERA FACING -> ${state.facingMode.toUpperCase()}`);
      if (state.camActive) {
        await startCameraStream();
      }
    });
  }

  // ---- SYNTHETIC RIG DEMO (No Webcam Required) ----
  el.btnDemoToggle.addEventListener("click", () => {
    if (state.demoActive) {
      stopActiveLoops();
      el.standbyOverlay.classList.remove("hidden");
      logEvent("SYNTHETIC RIG DEMO STOPPED", "warn");
      return;
    }
    stopActiveLoops();
    state.demoActive = true;
    state.demoStartTs = performance.now();
    el.btnDemoToggle.classList.add("active");
    el.standbyOverlay.classList.add("hidden");
    el.hudCanvas.width = 960;
    el.hudCanvas.height = 540;
    el.vpRes.textContent = "960 × 540 PX [SIM]";
    logEvent("SYNTHETIC BIOMECHANICAL RIG ENGAGED", "ok");
    runSyntheticDemoLoop();
  });

  function calcDeg(a, b, c) {
    const bax = a.x - b.x, bay = a.y - b.y;
    const bcx = c.x - b.x, bcy = c.y - b.y;
    const m1 = Math.hypot(bax, bay), m2 = Math.hypot(bcx, bcy);
    if (m1 < 1e-6 || m2 < 1e-6) return 0;
    const cos = Math.max(-1, Math.min(1, (bax * bcx + bay * bcy) / (m1 * m2)));
    return Math.round((Math.acos(cos) * 180) / Math.PI * 10) / 10;
  }

  function runSyntheticDemoLoop() {
    if (!state.demoActive) return;

    const t = (performance.now() - state.demoStartTs) / 1000;
    const w = el.hudCanvas.width;
    const h = el.hudCanvas.height;

    // Draw dark industrial blueprint backdrop
    hudCtx.fillStyle = "#080b10";
    hudCtx.fillRect(0, 0, w, h);
    hudCtx.strokeStyle = "rgba(0, 229, 255, 0.06)";
    hudCtx.lineWidth = 1;
    for (let x = 0; x <= w; x += 32) {
      hudCtx.beginPath(); hudCtx.moveTo(x, 0); hudCtx.lineTo(x, h); hudCtx.stroke();
    }
    for (let y = 0; y <= h; y += 32) {
      hudCtx.beginPath(); hudCtx.moveTo(0, y); hudCtx.lineTo(w, y); hudCtx.stroke();
    }

    // Animated biomechanical kinematics (squat + arm elevation cycle)
    const squat = (Math.sin(t * 1.4) * 0.5 + 0.5) * 0.11;
    const armWave = Math.sin(t * 2.1);
    const sway = Math.sin(t * 0.9) * 0.03;

    const cx = 0.5 + sway;
    const hipY = 0.53 + squat;
    const shY = 0.30 + squat;
    const headY = 0.19 + squat;

    const lElbX = cx - 0.13, lElbY = shY + 0.09 - armWave * 0.09;
    const lWriX = cx - 0.19, lWriY = lElbY + 0.07 - armWave * 0.12;
    const rElbX = cx + 0.13, rElbY = shY + 0.09 + armWave * 0.06;
    const rWriX = cx + 0.19, rWriY = rElbY + 0.08 + armWave * 0.08;

    const lKneeX = cx - 0.065 - squat * 0.35, lKneeY = 0.72 + squat * 0.45;
    const rKneeX = cx + 0.065 + squat * 0.35, rKneeY = 0.72 + squat * 0.45;

    const rawPts = [
      [cx, headY], [cx - 0.015, headY - 0.012], [cx + 0.015, headY - 0.012],
      [cx - 0.03, headY], [cx + 0.03, headY],
      [cx - 0.075, shY], [cx + 0.075, shY],
      [lElbX, lElbY], [rElbX, rElbY],
      [lWriX, lWriY], [rWriX, rWriY],
      [cx - 0.055, hipY], [cx + 0.055, hipY],
      [lKneeX, lKneeY], [rKneeX, rKneeY],
      [cx - 0.06, 0.90], [cx + 0.06, 0.90],
    ];

    const kps = rawPts.map(([x, y], id) => ({ id, name: `KP_${id}`, x, y, conf: 0.97 }));
    const angles = {
      L_ELBOW: calcDeg(kps[5], kps[7], kps[9]),
      R_ELBOW: calcDeg(kps[6], kps[8], kps[10]),
      L_SHOULDER: calcDeg(kps[11], kps[5], kps[7]),
      R_SHOULDER: calcDeg(kps[12], kps[6], kps[8]),
      L_KNEE: calcDeg(kps[11], kps[13], kps[15]),
      R_KNEE: calcDeg(kps[12], kps[14], kps[16]),
      L_HIP: calcDeg(kps[5], kps[11], kps[13]),
      R_HIP: calcDeg(kps[6], kps[12], kps[14]),
      SPINE_TILT: Math.round(Math.abs(sway) * 180 * 10) / 10,
    };

    const posture = squat > 0.07 ? "SQUAT_FLEX" : armWave > 0.7 ? "ARMS_RAISED" : "NOMINAL_UPRIGHT";

    // Build synthetic 21-pt hand attached to a wrist point
    const makeSimHand = (wx, wy, side, phase) => {
      const dir = side === "LEFT" ? -1 : 1;
      const open = Math.sin(t * 2.2 + phase) > -0.2;
      const spread = open ? 1.0 : 0.35;
      const pts = [[wx, wy]];
      // Thumb (1..4)
      for (let s = 1; s <= 4; s++) {
        pts.push([wx + dir * (0.012 + s * 0.009 * spread), wy - s * 0.008 * spread]);
      }
      // Index, Middle, Ring, Pinky (5..20)
      const offsets = [-0.018, -0.006, 0.006, 0.018];
      const lens = [0.014, 0.016, 0.015, 0.012];
      for (let f = 0; f < 4; f++) {
        const bx = wx + dir * 0.018 + offsets[f] * dir;
        const by = wy - 0.022;
        pts.push([bx, by]);
        for (let s = 1; s <= 3; s++) {
          pts.push([bx + offsets[f] * dir * s * 0.28 * spread, by - s * lens[f] * spread]);
        }
      }
      const hkps = pts.map(([x, y], id) => ({ id, name: `H_${id}`, x, y, conf: 0.96 }));
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      return {
        hand_id: side === "LEFT" ? "HAND-L1" : "HAND-R1",
        handedness: side,
        score: 0.96,
        bbox: {
          x1: Math.min(...xs) - 0.015,
          y1: Math.min(...ys) - 0.015,
          x2: Math.max(...xs) + 0.015,
          y2: Math.max(...ys) + 0.015,
        },
        keypoints: hkps,
        kinematics: {
          gesture: open ? "OPEN_PALM" : "FIST_LOCK",
          extended_count: open ? 5 : 0,
          pinch_ratio: open ? 0.82 : 0.35,
          fingers: { THUMB: open, INDEX: open, MIDDLE: open, RING: open, PINKY: open },
        },
      };
    };

    const simPersons = state.enablePose
      ? [
          {
            unit_id: "SIM-01",
            det_conf: 0.99,
            bbox: { x1: cx - 0.23, y1: headY - 0.06, x2: cx + 0.23, y2: 0.94 },
            keypoints: kps,
            kinematics: { angles, posture, integrity_pct: 100.0 },
          },
        ]
      : [];

    const simHands = state.enableHands
      ? [makeSimHand(lWriX, lWriY, "LEFT", 0), makeSimHand(rWriX, rWriY, "RIGHT", 1.8)]
      : [];

    const simPose = {
      engine: "synthetic-rig",
      topology: "coco17",
      inference_ms: 4.2,
      enable_pose: state.enablePose,
      enable_hands: state.enableHands,
      person_count: simPersons.length,
      hand_count: simHands.length,
      vram_used_mb: 0,
      persons: simPersons,
      hands: simHands,
    };

    handlePoseTelemetry(simPose);
    drawMechanicalHUD(simPose, false);
    state.rafId = requestAnimationFrame(runSyntheticDemoLoop);
  }

  // ---- Shared Real-Time Video/Webcam Render & WS Inference Loop ----
  function runLiveVideoLoop(isWebcam) {
    const v = el.sourceVideo;

    function tick() {
      if (isWebcam && !state.camActive) return;
      if (!isWebcam && state.mode !== "video") return;

      if (v.readyState >= 2 && v.videoWidth > 0) {
        if (el.hudCanvas.width !== v.videoWidth || el.hudCanvas.height !== v.videoHeight) {
          el.hudCanvas.width = v.videoWidth;
          el.hudCanvas.height = v.videoHeight;
          el.vpRes.textContent = `${v.videoWidth} × ${v.videoHeight} PX`;
        }

        const mirror = isWebcam && el.chkMirror.checked;
        hudCtx.save();
        if (mirror) {
          hudCtx.translate(el.hudCanvas.width, 0);
          hudCtx.scale(-1, 1);
        }
        hudCtx.drawImage(v, 0, 0, el.hudCanvas.width, el.hudCanvas.height);
        hudCtx.restore();

        // Overlay latest GPU telemetry
        drawMechanicalHUD(state.lastPose, mirror);

        // Send frame over WebSocket using adaptive external-network sizing
        if (
          state.ws &&
          state.ws.readyState === WebSocket.OPEN &&
          !state.wsBusy &&
          (!v.paused || isWebcam)
        ) {
          state.wsBusy = true;
          const sp = getAdaptiveStreamParams();
          const capW = Math.min(sp.width, v.videoWidth);
          const capH = Math.max(160, Math.round((capW / v.videoWidth) * v.videoHeight));
          el.captureCanvas.width = capW;
          el.captureCanvas.height = capH;
          capCtx.drawImage(v, 0, 0, capW, capH);
          el.captureCanvas.toBlob(
            (blob) => {
              if (blob && state.ws && state.ws.readyState === WebSocket.OPEN) {
                state.lastFrameSentTs = performance.now();
                if (el.streamBwBadge) {
                  el.streamBwBadge.textContent = `${sp.label} · ${(blob.size / 1024).toFixed(0)}KB`;
                }
                state.ws.send(blob);
              } else {
                state.wsBusy = false;
              }
            },
            "image/jpeg",
            sp.quality
          );
        }

        // Update video seek bar
        if (!isWebcam && v.duration) {
          el.vidSeek.value = ((v.currentTime / v.duration) * 100).toFixed(1);
        }
      }

      state.rafId = requestAnimationFrame(tick);
    }

    state.rafId = requestAnimationFrame(tick);
  }

  // ---- MODE 02: IMAGE SCAN ----
  function handleImageFile(file) {
    if (!file) return;
    state.loadedImageFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        state.loadedImage = img;
        el.hudCanvas.width = img.naturalWidth;
        el.hudCanvas.height = img.naturalHeight;
        el.vpRes.textContent = `${img.naturalWidth} × ${img.naturalHeight} PX`;
        el.standbyOverlay.classList.add("hidden");
        el.btnRescanImage.disabled = false;
        scanLoadedImage();
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  async function scanLoadedImage() {
    if (!state.loadedImageFile || !state.loadedImage) return;
    logEvent(`SCANNING IMAGE: ${state.loadedImageFile.name}...`);

    const form = new FormData();
    form.append("file", state.loadedImageFile);
    form.append("conf", el.confSlider.value);
    form.append("engine_name", el.engineSelect.value);
    form.append("imgsz", el.imgszSelect.value);
    form.append("enable_pose", String(state.enablePose));
    form.append("enable_hands", String(state.enableHands));

    try {
      const res = await fetch("/api/detect/image", { method: "POST", body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      hudCtx.clearRect(0, 0, el.hudCanvas.width, el.hudCanvas.height);
      hudCtx.drawImage(state.loadedImage, 0, 0);
      handlePoseTelemetry(data);
      drawMechanicalHUD(data, false);
      logEvent(
        `IMAGE SCAN COMPLETE // BODY:${data.person_count} HAND:${data.hand_count ?? 0} IN ${data.inference_ms}ms`,
        "ok"
      );
    } catch (err) {
      logEvent(`IMAGE SCAN ERROR: ${err.message}`, "err");
    }
  }

  el.inputImage.addEventListener("change", (e) => handleImageFile(e.target.files[0]));
  el.btnRescanImage.addEventListener("click", scanLoadedImage);
  setupDropzone(el.imageDropzone, handleImageFile);

  // ---- MODE 03: VIDEO STREAM ----
  function handleVideoFile(file) {
    if (!file) return;
    stopActiveLoops();
    if (state.loadedVideoUrl) URL.revokeObjectURL(state.loadedVideoUrl);
    state.loadedVideoUrl = URL.createObjectURL(file);
    el.sourceVideo.srcObject = null;
    el.sourceVideo.src = state.loadedVideoUrl;
    el.sourceVideo.loop = true;
    el.sourceVideo.playbackRate = parseFloat(el.vidSpeed.value);

    el.sourceVideo.onloadedmetadata = () => {
      el.btnVidPlay.disabled = false;
      el.vidSeek.disabled = false;
      el.vidSpeed.disabled = false;
      el.standbyOverlay.classList.add("hidden");
      ensureWebSocket();
      el.sourceVideo.play();
      el.btnVidPlay.textContent = "PAUSE";
      logEvent(`VIDEO LOADED: ${file.name}`, "ok");
      runLiveVideoLoop(false);
    };
  }

  el.inputVideo.addEventListener("change", (e) => handleVideoFile(e.target.files[0]));
  setupDropzone(el.videoDropzone, handleVideoFile);

  el.btnVidPlay.addEventListener("click", () => {
    if (el.sourceVideo.paused) {
      el.sourceVideo.play();
      el.btnVidPlay.textContent = "PAUSE";
    } else {
      el.sourceVideo.pause();
      el.btnVidPlay.textContent = "PLAY";
    }
  });

  el.vidSeek.addEventListener("input", () => {
    if (el.sourceVideo.duration) {
      el.sourceVideo.currentTime = (parseFloat(el.vidSeek.value) / 100) * el.sourceVideo.duration;
    }
  });

  el.vidSpeed.addEventListener("change", () => {
    el.sourceVideo.playbackRate = parseFloat(el.vidSpeed.value);
  });

  function setupDropzone(zone, onFile) {
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onFile(e.dataTransfer.files[0]);
      }
    });
  }

  // ---- Independent Detector Switches (Body Skeleton & Hand 21-PT) ----
  el.btnTogglePose.addEventListener("click", () => {
    state.enablePose = !state.enablePose;
    syncModuleButtons();
    sendWsConfig();
    logEvent(
      `BODY SKELETON DETECTOR -> ${state.enablePose ? "ENGAGED (ON)" : "BYPASSED (OFF)"}`,
      state.enablePose ? "ok" : "warn"
    );
    if (state.mode === "image" && state.loadedImage) {
      scanLoadedImage();
    } else if (!state.camActive && !state.demoActive && state.lastPose) {
      updateRightTelemetry(state.lastPose);
    }
  });

  el.btnToggleHands.addEventListener("click", () => {
    state.enableHands = !state.enableHands;
    syncModuleButtons();
    sendWsConfig();
    logEvent(
      `HAND 21-PT DETECTOR -> ${state.enableHands ? "ENGAGED (ON)" : "BYPASSED (OFF)"}`,
      state.enableHands ? "ok" : "warn"
    );
    if (state.mode === "image" && state.loadedImage) {
      scanLoadedImage();
    } else if (!state.camActive && !state.demoActive && state.lastPose) {
      updateRightTelemetry(state.lastPose);
    }
  });

  // ---- Engine & Parameter Controls ----
  el.engineSelect.addEventListener("change", async () => {
    const eng = el.engineSelect.value;
    el.vpEngineBadge.textContent = `CORE: ${eng.toUpperCase()}`;
    logEvent(`LOADING NEURAL CORE: ${eng.toUpperCase()}...`, "warn");
    try {
      const res = await fetch("/api/engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine: eng }),
      });
      if (!res.ok) throw new Error(await res.text());
      const info = await res.json();
      updateVram(info.vram_used_mb || 0, state.vramTotalMb);
      sendWsConfig();
      logEvent(`CORE ONLINE: ${eng.toUpperCase()}`, "ok");
      if (state.mode === "image" && state.loadedImage) scanLoadedImage();
    } catch (err) {
      logEvent(`CORE SWITCH FAILED: ${err.message}`, "err");
    }
  });

  el.confSlider.addEventListener("input", () => {
    el.confVal.textContent = parseFloat(el.confSlider.value).toFixed(2);
    sendWsConfig();
    if (state.mode === "image" && state.loadedImage && state.lastPose) {
      hudCtx.drawImage(state.loadedImage, 0, 0);
      drawMechanicalHUD(state.lastPose, false);
      updateRightTelemetry(state.lastPose);
    }
  });

  el.imgszSelect.addEventListener("change", () => {
    sendWsConfig();
    logEvent(`TENSOR RESOLUTION SET -> ${el.imgszSelect.value}PX`);
  });

  // Visual Layer Toggles
  [el.chkSkeleton, el.chkAngles, el.chkBbox].forEach((chk) => {
    chk.addEventListener("change", () => {
      if (state.mode === "image" && state.loadedImage && state.lastPose) {
        hudCtx.drawImage(state.loadedImage, 0, 0);
        drawMechanicalHUD(state.lastPose, false);
      }
    });
  });

  el.chkScanlines.addEventListener("change", () => {
    el.scanlineOverlay.classList.toggle("hidden", !el.chkScanlines.checked);
  });

  // ---- Export Snapshot & JSON ----
  el.btnSnapPng.addEventListener("click", () => {
    if (!el.hudCanvas.width) return;
    const link = document.createElement("a");
    link.download = `kinetic-x-hud-${Date.now()}.png`;
    link.href = el.hudCanvas.toDataURL("image/png");
    link.click();
    logEvent("EXPORTED HUD SNAPSHOT (PNG)", "ok");
  });

  el.btnExportJson.addEventListener("click", () => {
    if (!state.lastPose) {
      logEvent("NO TELEMETRY DATA TO EXPORT", "warn");
      return;
    }
    const blob = new Blob([JSON.stringify(state.lastPose, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.download = `kinetic-x-telemetry-${Date.now()}.json`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
    logEvent("EXPORTED BIOMECHANICAL LOG (JSON)", "ok");
  });

  // ---- Boot ----
  initAngleRows();
  fetchSystemInfo();
})();
