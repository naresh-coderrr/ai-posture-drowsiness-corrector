/**
 * SpineGuard AI — Real-Time Posture, Drowsiness & Background Study Sentinel
 * © 2026 Naresh (naresh-coderrr). All Rights Reserved.
 */

(() => {
  // --- Session & Trial State ---
  const SESSION_ID = localStorage.getItem("spineguard_session_id") || ("sess_" + Math.random().toString(36).substring(2, 11));
  localStorage.setItem("spineguard_session_id", SESSION_ID);

  const FREE_TRIAL_TOTAL_SEC = 5 * 60; // 5 minutes (300 seconds)
  let trialRemainingSec = parseInt(localStorage.getItem("spineguard_trial_remaining") ?? String(FREE_TRIAL_TOTAL_SEC), 10);
  if (isNaN(trialRemainingSec) || trialRemainingSec < 0) {
    trialRemainingSec = FREE_TRIAL_TOTAL_SEC;
  }

  let currentUser = null;
  try {
    const savedUser = localStorage.getItem("spineguard_user");
    if (savedUser) currentUser = JSON.parse(savedUser);
  } catch (_) {}

  let isTrackingPaused = false;
  let isTrialLocked = false;
  let forceCalibrateNext = false;
  let audioMode = "funny"; // 'funny' | 'alarm' | 'custom'
  let customAudioUrl = null;
  let customAudioElement = null;
  let lastAlertTimestamp = 0;
  let lastNotificationTimestamp = 0;
  let frameCounter = 0;
  let lastFpsCheck = performance.now();

  // Telemetry Waveform History
  const MAX_HISTORY = 60;
  const earHistory = new Array(MAX_HISTORY).fill(0.30);
  const spineHistory = new Array(MAX_HISTORY).fill(100);

  // Funny Drill-Sergeant & Study Coach Voice Lines
  const FUNNY_VOICE_LINES = {
    DROWSY: [
      "Wake up buddy! Your eyes just closed! Coffee time or sit up straight right now!",
      "Excuse me! Sleeping on your keyboard will not pass your exams! Open those eyes!",
      "Alert! Alert! Brain entering power-saving mode! Sit up straight and blink!",
      "Hey! Gravity is winning against your eyelids! Sit up and lock back in!"
    ],
    SLOUCH: [
      "Whoa there shrimp posture! Un-curve your spine and pull your shoulders back!",
      "Posture check! You look like a question mark right now. Sit up straight like a CEO!",
      "Your spine called—it says stop slouching before you turn into a banana!",
      "Back straight, shoulders level, chin up! Instant posture upgrade required!"
    ],
    PHONE_DISTRACTION: [
      "Caught red-handed! Put that mobile phone down and get back to studying!",
      "Step away from the phone! Instagram Reels will not write your code or finish your syllabus!",
      "Distraction alert! Hand and mobile detected! Eyes back on the laptop screen!",
      "Drop the phone soldier! Five minutes of scrolling turns into two hours!"
    ],
    DISTRACTED: [
      "Hey! Where are you looking? Eyes back on your study session!",
      "Focus drift detected! Lock your gaze back on your work!"
    ]
  };

  // --- DOM Elements ---
  const webcamVideo = document.getElementById("webcamVideo");
  const hudCanvas = document.getElementById("hudCanvas");
  const hudCtx = hudCanvas.getContext("2d");
  const captureCanvas = document.getElementById("captureCanvas");
  const captureCtx = captureCanvas.getContext("2d");
  const pipCanvas = document.getElementById("pipCanvas");
  const pipCtx = pipCanvas.getContext("2d");
  const pipVideo = document.getElementById("pipVideo");
  const telemetryChart = document.getElementById("telemetryChart");
  const chartCtx = telemetryChart.getContext("2d");

  const trialBadgeContainer = document.getElementById("trialBadgeContainer");
  const trialLabelText = document.getElementById("trialLabelText");
  const trialCountdownText = document.getElementById("trialCountdownText");
  const navUserText = document.getElementById("navUserText");

  const emergencyBanner = document.getElementById("emergencyBanner");
  const emergencyBannerTitle = document.getElementById("emergencyBannerTitle");
  const emergencyBannerSub = document.getElementById("emergencyBannerSub");
  const videoViewport = document.getElementById("videoViewport");
  const videoAlertOverlay = document.getElementById("videoAlertOverlay");
  const videoAlertTitle = document.getElementById("videoAlertTitle");
  const videoAlertDesc = document.getElementById("videoAlertDesc");

  // --- Web Audio API Synthesizer & Keep-Alive Engine ---
  let audioCtx = null;
  function ensureAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContextClass();
    }
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    return audioCtx;
  }

  // Silent ultra-low-gain oscillator so Chrome/Edge never throttles this tab when user switches tabs to study
  function startBackgroundKeepAliveAudio() {
    try {
      const ctx = ensureAudioContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      gain.gain.value = 0.00001; // Completely inaudible
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
    } catch (_) {}
  }

  document.addEventListener("click", () => {
    ensureAudioContext();
    startBackgroundKeepAliveAudio();
  }, { once: true });

  // Synthesize high-visibility tactical alarm siren via Web Audio API
  function playTacticalSiren(severity = "critical") {
    try {
      const ctx = ensureAudioContext();
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = severity === "critical" ? "sawtooth" : "square";
      osc.frequency.setValueAtTime(580, now);
      osc.frequency.linearRampToValueAtTime(980, now + 0.22);
      osc.frequency.linearRampToValueAtTime(520, now + 0.45);
      osc.frequency.linearRampToValueAtTime(1020, now + 0.70);

      gain.gain.setValueAtTime(0.28, now);
      gain.gain.exponentialRampToValueAtTime(0.008, now + 0.85);

      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.85);
    } catch (e) {
      console.warn("Audio siren error:", e);
    }
  }

  // Funny cartoon boing/horn intro before TTS roast
  function playFunnyIntroChime() {
    try {
      const ctx = ensureAudioContext();
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "triangle";
      osc.frequency.setValueAtTime(320, now);
      osc.frequency.exponentialRampToValueAtTime(760, now + 0.16);
      osc.frequency.exponentialRampToValueAtTime(420, now + 0.32);
      gain.gain.setValueAtTime(0.25, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.35);
    } catch (_) {}
  }

  // Trigger Audio Alert based on user's selected mode (Funny Roast, Tactical Siren, or Custom MP3)
  function triggerAudioAlert(alertObj) {
    const isMuted = document.getElementById("chkAudioMute")?.checked;
    if (isMuted) return;

    const now = Date.now();
    if (now - lastAlertTimestamp < 3200) return; // Cooldown 3.2s between audio triggers
    lastAlertTimestamp = now;

    if (audioMode === "custom" && customAudioElement) {
      customAudioElement.currentTime = 0;
      customAudioElement.play().catch(() => playTacticalSiren(alertObj.severity));
      return;
    }

    if (audioMode === "alarm") {
      playTacticalSiren(alertObj.severity);
      return;
    }

    // Default: "funny" mode (Funny chime + SpeechSynthesis roast)
    playFunnyIntroChime();
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const lines = FUNNY_VOICE_LINES[alertObj.type] || FUNNY_VOICE_LINES.SLOUCH;
      const line = lines[Math.floor(Math.random() * lines.length)];
      const utter = new SpeechSynthesisUtterance(line);
      utter.rate = 1.08;
      utter.pitch = 1.12;
      utter.volume = 1.0;
      window.speechSynthesis.speak(utter);
    } else {
      playTacticalSiren(alertObj.severity);
    }
  }

  // --- Laptop / Desktop OS System Notification (Works when user is in another tab!) ---
  function triggerDesktopNotification(alertObj) {
    const now = Date.now();
    // Trigger immediately if tab is hidden, or every 6s if active
    const cooldown = document.hidden ? 3500 : 7000;
    if (now - lastNotificationTimestamp < cooldown) return;
    lastNotificationTimestamp = now;

    // Flash document title when user is in another study tab
    if (document.hidden) {
      document.title = `🚨 ${alertObj.title} — SpineGuard AI`;
    }

    if ("Notification" in window && Notification.permission === "granted") {
      try {
        const notif = new Notification(`🚨 ${alertObj.title}`, {
          body: alertObj.message,
          tag: "spineguard-live-alert",
          renotify: true,
          requireInteraction: false,
        });
        notif.onclick = () => {
          window.focus();
          notif.close();
        };
      } catch (_) {}
    }
  }

  // Request OS Notification Permission
  function requestNotificationPermission() {
    if (!("Notification" in window)) return;
    Notification.requestPermission().then((perm) => {
      const btnText = document.getElementById("notifBtnText");
      if (btnText) {
        btnText.textContent = perm === "granted" ? "Laptop Alerts: ON" : "Enable OS Alerts";
      }
    });
  }

  // --- 5-Minute Free Trial & User Auth State ---
  function formatMMSS(totalSec) {
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function updateAuthUI() {
    if (currentUser) {
      isTrialLocked = false;
      navUserText.textContent = `@${currentUser.username}`;
      trialBadgeContainer.className = "flex items-center gap-2 px-3 py-1.5 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-mono";
      trialLabelText.textContent = "PRO MEMBER:";
      trialCountdownText.textContent = "UNLIMITED";
      document.getElementById("trialExpiredNotice")?.classList.add("hidden");
    } else {
      navUserText.textContent = "Login / Sign Up";
      trialCountdownText.textContent = formatMMSS(trialRemainingSec);
      if (trialRemainingSec <= 0) {
        lockForTrialExpiration();
      }
    }
  }

  function lockForTrialExpiration() {
    if (currentUser) return;
    isTrialLocked = true;
    trialCountdownText.textContent = "00:00 (EXPIRED)";
    document.getElementById("trialExpiredNotice")?.classList.remove("hidden");
    document.getElementById("authModal")?.classList.remove("hidden");
    appendIncidentLog("WARN", "5-Minute Free Trial completed! Please log in or create a unique-password account to continue.");
  }

  // Tick 5-Minute Free Trial every second
  setInterval(() => {
    if (currentUser || isTrackingPaused || isTrialLocked) return;
    if (trialRemainingSec > 0) {
      trialRemainingSec -= 1;
      localStorage.setItem("spineguard_trial_remaining", String(trialRemainingSec));
      trialCountdownText.textContent = formatMMSS(trialRemainingSec);
      if (trialRemainingSec <= 60) {
        trialBadgeContainer.className = "flex items-center gap-2 px-3 py-1.5 rounded-xl bg-rose-500/20 border border-rose-500/50 text-rose-200 text-xs font-mono animate-pulse";
      }
    } else {
      lockForTrialExpiration();
    }
  }, 1000);

  // --- Incident Log Helper ---
  function appendIncidentLog(level, text) {
    const list = document.getElementById("alertHistoryList");
    if (!list) return;
    const timeStr = new Date().toLocaleTimeString();
    const item = document.createElement("div");
    const colorMap = {
      CRITICAL: "bg-rose-950/60 border-rose-500/40 text-rose-200",
      WARN: "bg-amber-950/60 border-amber-500/40 text-amber-200",
      INFO: "bg-cyan-950/50 border-cyan-500/30 text-cyan-200",
    };
    item.className = `p-2 rounded-xl border ${colorMap[level] || colorMap.INFO}`;
    item.textContent = `[${timeStr}] [${level}] ${text}`;
    list.prepend(item);
    while (list.children.length > 35) {
      list.removeChild(list.lastChild);
    }
  }

  // --- Draw Biometric Landmarks on Live Video Overlay Canvas ---
  let latestTelemetry = null;

  function drawBiometricOverlay(data) {
    if (!webcamVideo.videoWidth) return;
    if (hudCanvas.width !== webcamVideo.videoWidth || hudCanvas.height !== webcamVideo.videoHeight) {
      hudCanvas.width = webcamVideo.videoWidth;
      hudCanvas.height = webcamVideo.videoHeight;
    }

    const w = hudCanvas.width;
    const h = hudCanvas.height;
    hudCtx.clearRect(0, 0, w, h);

    if (!data || !data.overlay) return;
    const ov = data.overlay;

    // 1. Draw Eye Contours (Cyan or Red if closed)
    const eyeColor = data.ear.avg < data.ear.threshold ? "#f43f5e" : "#22d3ee";
    [ov.left_eye, ov.right_eye].forEach((eyePts) => {
      if (!eyePts || eyePts.length === 0) return;
      hudCtx.beginPath();
      eyePts.forEach((pt, idx) => {
        const x = pt[0] * w;
        const y = pt[1] * h;
        if (idx === 0) hudCtx.moveTo(x, y);
        else hudCtx.lineTo(x, y);
      });
      hudCtx.closePath();
      hudCtx.strokeStyle = eyeColor;
      hudCtx.lineWidth = 2.2;
      hudCtx.stroke();
    });

    // 2. Draw Mouth Contour (Purple or Amber if yawning)
    if (ov.mouth && ov.mouth.length > 0) {
      hudCtx.beginPath();
      ov.mouth.forEach((pt, idx) => {
        const x = pt[0] * w;
        const y = pt[1] * h;
        if (idx === 0) hudCtx.moveTo(x, y);
        else hudCtx.lineTo(x, y);
      });
      hudCtx.closePath();
      hudCtx.strokeStyle = data.mar.value > data.mar.threshold ? "#f59e0b" : "#a855f7";
      hudCtx.lineWidth = 2;
      hudCtx.stroke();
    }

    // 3. Draw Spinal / Shoulder Alignment Axis
    if (ov.pose && ov.pose.length >= 5) {
      const map = {};
      ov.pose.forEach((p) => { map[p.id] = p; });
      const lSh = map.left_shoulder;
      const rSh = map.right_shoulder;
      const nose = map.nose;

      if (lSh && rSh && nose) {
        const spineColor = data.posture.is_slouching ? "#f43f5e" : "#10b981";
        // Shoulder bar
        hudCtx.beginPath();
        hudCtx.moveTo(lSh.x * w, lSh.y * h);
        hudCtx.lineTo(rSh.x * w, rSh.y * h);
        hudCtx.strokeStyle = spineColor;
        hudCtx.lineWidth = 3.5;
        hudCtx.stroke();

        // Mid-shoulder to Nose cervical axis
        const midX = ((lSh.x + rSh.x) / 2) * w;
        const midY = ((lSh.y + rSh.y) / 2) * h;
        hudCtx.beginPath();
        hudCtx.moveTo(midX, midY);
        hudCtx.lineTo(nose.x * w, nose.y * h);
        hudCtx.strokeStyle = "#38bdf8";
        hudCtx.lineWidth = 2.5;
        hudCtx.setLineDash([6, 4]);
        hudCtx.stroke();
        hudCtx.setLineDash([]);

        // Shoulder joint nodes
        [lSh, rSh, nose].forEach((joint) => {
          hudCtx.beginPath();
          hudCtx.arc(joint.x * w, joint.y * h, 5, 0, Math.PI * 2);
          hudCtx.fillStyle = spineColor;
          hudCtx.fill();
        });
      }
    }

    // 4. Draw Hand / Mobile Distraction Skeleton
    if (ov.hands && ov.hands.length > 0) {
      ov.hands.forEach((handPts) => {
        handPts.forEach((pt) => {
          hudCtx.beginPath();
          hudCtx.arc(pt[0] * w, pt[1] * h, 4, 0, Math.PI * 2);
          hudCtx.fillStyle = data.distraction.phone_detected ? "#f43f5e" : "#f59e0b";
          hudCtx.fill();
        });
      });
    }
  }

  // --- Draw Real-Time Waveform Telemetry Graph ---
  function drawTelemetryChart() {
    const w = telemetryChart.width;
    const h = telemetryChart.height;
    chartCtx.clearRect(0, 0, w, h);

    // Subtle horizontal grid lines
    chartCtx.strokeStyle = "rgba(148, 163, 184, 0.12)";
    chartCtx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach((r) => {
      chartCtx.beginPath();
      chartCtx.moveTo(0, h * r);
      chartCtx.lineTo(w, h * r);
      chartCtx.stroke();
    });

    // Plot Spine Score (0..100%) in Emerald
    chartCtx.beginPath();
    spineHistory.forEach((val, idx) => {
      const x = (idx / (MAX_HISTORY - 1)) * w;
      const y = h - (val / 100) * (h - 16) - 8;
      if (idx === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.strokeStyle = "#10b981";
    chartCtx.lineWidth = 2.2;
    chartCtx.stroke();

    // Plot EAR (0.10..0.40) in Cyan
    chartCtx.beginPath();
    earHistory.forEach((val, idx) => {
      const norm = Math.max(0, Math.min(1, (val - 0.10) / 0.30));
      const x = (idx / (MAX_HISTORY - 1)) * w;
      const y = h - norm * (h - 16) - 8;
      if (idx === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.strokeStyle = "#22d3ee";
    chartCtx.lineWidth = 2.2;
    chartCtx.stroke();
  }

  // --- Render Floating Picture-in-Picture (PiP) Mini Study Monitor ---
  function updatePipCanvas(data) {
    const w = pipCanvas.width;
    const h = pipCanvas.height;
    pipCtx.fillStyle = "#050811";
    pipCtx.fillRect(0, 0, w, h);

    if (webcamVideo.readyState >= 2) {
      pipCtx.save();
      pipCtx.translate(w, 0);
      pipCtx.scale(-1, 1);
      pipCtx.drawImage(webcamVideo, 0, 0, w, h);
      pipCtx.restore();
    }

    // Dark HUD bar at bottom of PiP window
    pipCtx.fillStyle = "rgba(5, 8, 17, 0.86)";
    pipCtx.fillRect(0, h - 64, w, 64);

    if (data) {
      const isAlert = Boolean(data.active_alert);
      pipCtx.fillStyle = isAlert ? "#f43f5e" : "#10b981";
      pipCtx.font = "bold 15px monospace";
      pipCtx.fillText(
        isAlert ? `🚨 ${data.active_alert.title}` : `✅ POSTURE ${data.posture.spine_score}% | EAR ${data.ear.avg.toFixed(2)}`,
        12,
        h - 36
      );

      pipCtx.fillStyle = "#e2e8f0";
      pipCtx.font = "12px monospace";
      pipCtx.fillText(
        isAlert ? data.active_alert.message.slice(0, 48) : `Blinks: ${data.blinks.total} | Study Guard Active`,
        12,
        h - 14
      );
    }
  }

  // --- Update UI Cards & Trigger Alerts from Backend Telemetry ---
  let lastLoggedAlertType = null;

  function updateDashboardUI(data) {
    latestTelemetry = data;

    // FPS Counter
    frameCounter += 1;
    const nowPerf = performance.now();
    if (nowPerf - lastFpsCheck >= 1000) {
      const fpsEl = document.getElementById("fpsCounter");
      if (fpsEl) fpsEl.textContent = `${frameCounter} FPS`;
      frameCounter = 0;
      lastFpsCheck = nowPerf;
    }

    // Update History Buffers
    earHistory.push(data.ear.avg);
    if (earHistory.length > MAX_HISTORY) earHistory.shift();
    spineHistory.push(data.posture.spine_score);
    if (spineHistory.length > MAX_HISTORY) spineHistory.shift();

    // HUD Badges
    document.getElementById("hudEarVal").textContent = data.ear.avg.toFixed(3);
    const hudEyeState = document.getElementById("hudEyeState");
    if (data.ear.avg < data.ear.threshold) {
      hudEyeState.textContent = "CLOSED";
      hudEyeState.className = "text-[10px] px-1.5 py-0.2 rounded bg-rose-500/30 text-rose-300";
    } else {
      hudEyeState.textContent = "OPEN";
      hudEyeState.className = "text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300";
    }

    document.getElementById("hudMarVal").textContent = data.mar.value.toFixed(3);
    const hudMouthState = document.getElementById("hudMouthState");
    if (data.mar.value > data.mar.threshold) {
      hudMouthState.textContent = "YAWNING";
      hudMouthState.className = "text-[10px] px-1.5 py-0.2 rounded bg-amber-500/30 text-amber-200";
    } else {
      hudMouthState.textContent = "NORMAL";
      hudMouthState.className = "text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-300";
    }

    document.getElementById("hudPoseVal").textContent = `Yaw ${Math.round(data.posture.head_yaw)}° | Pitch ${Math.round(data.posture.head_pitch)}°`;
    document.getElementById("hudSpineScore").textContent = data.posture.spine_score;
    const hudPostureBadge = document.getElementById("hudPostureBadge");
    if (data.posture.is_slouching) {
      hudPostureBadge.textContent = "SLOUCHING";
      hudPostureBadge.className = "text-[10px] font-mono text-rose-400 font-bold";
    } else {
      hudPostureBadge.textContent = "UPRIGHT";
      hudPostureBadge.className = "text-[10px] font-mono text-emerald-300";
    }

    // Telemetry Cards
    document.getElementById("cardEarValue").textContent = data.ear.avg.toFixed(3);
    document.getElementById("cardBlinkRate").textContent = `${data.blinks.per_minute} blinks/m`;
    document.getElementById("cardTotalBlinks").textContent = data.blinks.total;
    document.getElementById("cardEyeClosedSec").textContent = `${data.ear.closed_seconds.toFixed(1)}s`;
    const earPct = Math.max(8, Math.min(100, Math.round((data.ear.avg / 0.36) * 100)));
    document.getElementById("barEar").style.width = `${earPct}%`;

    document.getElementById("cardSpineScore").textContent = `${data.posture.spine_score}%`;
    document.getElementById("cardNeckAngle").textContent = `${data.posture.neck_angle.toFixed(1)}° Neck`;
    document.getElementById("cardShoulderTilt").textContent = `${Math.abs(data.posture.shoulder_tilt).toFixed(1)}°`;
    document.getElementById("barSpine").style.width = `${data.posture.spine_score}%`;
    const cardPostureStatus = document.getElementById("cardPostureStatus");
    if (data.posture.is_slouching) {
      cardPostureStatus.textContent = data.posture.reasons[0] || "Slouching";
      cardPostureStatus.className = "text-rose-400 font-semibold";
    } else {
      cardPostureStatus.textContent = "Aligned";
      cardPostureStatus.className = "text-emerald-300";
    }

    // Phone / Hand Distraction Card
    const cardPhoneState = document.getElementById("cardPhoneState");
    const cardPhoneDot = document.getElementById("cardPhoneDot");
    const cardPhoneReason = document.getElementById("cardPhoneReason");
    document.getElementById("cardPhoneSec").textContent = `${data.distraction.phone_seconds.toFixed(1)}s`;
    if (data.distraction.phone_detected) {
      cardPhoneState.textContent = "PHONE / HAND!";
      cardPhoneState.className = "text-lg sm:text-xl font-bold font-mono text-rose-400";
      cardPhoneDot.className = "w-2.5 h-2.5 rounded-full bg-rose-500 animate-ping";
      cardPhoneReason.textContent = data.distraction.phone_reason || "Hand/Phone detected in study zone";
    } else {
      cardPhoneState.textContent = "NO PHONE";
      cardPhoneState.className = "text-lg sm:text-xl font-bold font-mono text-emerald-300";
      cardPhoneDot.className = "w-2 h-2 rounded-full bg-emerald-400";
      cardPhoneReason.textContent = "Hands clear of face & mobile zone";
    }

    // MAR / Gaze Card
    document.getElementById("cardMarValue").textContent = data.mar.value.toFixed(2);
    document.getElementById("cardYawnCount").textContent = `${data.blinks.yawns} Yawns`;
    document.getElementById("cardFocusReason").textContent = data.distraction.looking_away
      ? data.distraction.distraction_reason
      : "Gaze centered on study screen";

    // Draw Overlays & Charts
    drawBiometricOverlay(data);
    drawTelemetryChart();
    updatePipCanvas(data);

    // Handle Active Alert (Drowsiness, Slouch, Mobile Phone, or Gaze Distraction)
    if (data.active_alert) {
      const al = data.active_alert;
      emergencyBanner.classList.remove("hidden");
      emergencyBannerTitle.textContent = `${al.title} — ${al.message}`;
      emergencyBannerSub.textContent = `Mode: ${audioMode.toUpperCase()}`;

      videoViewport.classList.add("alert-ring-critical");
      videoAlertOverlay.classList.remove("hidden");
      videoAlertTitle.textContent = al.title;
      videoAlertDesc.textContent = al.message;

      triggerAudioAlert(al);
      triggerDesktopNotification(al);

      if (lastLoggedAlertType !== al.type) {
        lastLoggedAlertType = al.type;
        appendIncidentLog(al.severity === "critical" ? "CRITICAL" : "WARN", `${al.title}: ${al.message}`);
      }
    } else {
      emergencyBanner.classList.add("hidden");
      videoViewport.classList.remove("alert-ring-critical");
      videoAlertOverlay.classList.add("hidden");
      lastLoggedAlertType = null;
      if (!document.hidden) {
        document.title = "SpineGuard AI — Real-Time Posture & Drowsiness Corrector | © Naresh";
      }
    }
  }

  // --- WebSocket + Unthrottled Background Web Worker Loop ---
  let ws = null;
  let wsConnected = false;
  let frameInFlight = false;

  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/analyze/${SESSION_ID}`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      wsConnected = true;
      frameInFlight = false;
    };

    ws.onmessage = (event) => {
      frameInFlight = false;
      try {
        const data = JSON.parse(event.data);
        if (data && data.ear) {
          updateDashboardUI(data);
        }
      } catch (_) {}
    };

    ws.onclose = () => {
      wsConnected = false;
      frameInFlight = false;
      setTimeout(connectWebSocket, 1500);
    };

    ws.onerror = () => {
      wsConnected = false;
      frameInFlight = false;
    };
  }

  async function captureAndSendFrame() {
    if (isTrackingPaused || isTrialLocked || frameInFlight) return;
    if (!webcamVideo || webcamVideo.readyState < 2) return;

    captureCtx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
    const dataUri = captureCanvas.toDataURL("image/jpeg", 0.62);
    const shouldCalibrate = forceCalibrateNext;
    forceCalibrateNext = false;

    if (wsConnected && ws && ws.readyState === WebSocket.OPEN) {
      frameInFlight = true;
      ws.send(JSON.stringify({
        type: "frame",
        image: dataUri,
        calibrate: shouldCalibrate
      }));
    } else {
      // HTTP Fallback if WebSocket is reconnecting
      frameInFlight = true;
      try {
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: SESSION_ID,
            image_base64: dataUri,
            force_calibrate: shouldCalibrate
          })
        });
        if (res.ok) {
          const telemetry = await res.json();
          updateDashboardUI(telemetry);
        }
      } catch (_) {
      } finally {
        frameInFlight = false;
      }
    }
  }

  // Create an inline Web Worker timer so frame capture continues at 8-10 FPS even when the user switches browser tabs to study!
  function startUnthrottledWorkerTimer() {
    const workerCode = `
      let intervalId = null;
      self.onmessage = function(e) {
        if (e.data === 'start') {
          if (intervalId) clearInterval(intervalId);
          intervalId = setInterval(() => self.postMessage('tick'), 110);
        } else if (e.data === 'stop') {
          if (intervalId) clearInterval(intervalId);
        }
      };
    `;
    const blob = new Blob([workerCode], { type: "application/javascript" });
    const worker = new Worker(URL.createObjectURL(blob));
    worker.onmessage = () => {
      captureAndSendFrame();
    };
    worker.postMessage("start");
  }

  // Track Tab Visibility Changes (When user switches tabs to study)
  document.addEventListener("visibilitychange", () => {
    const badge = document.getElementById("tabVisibilityBadge");
    if (document.hidden) {
      if (badge) {
        badge.textContent = "STUDY TAB: BACKGROUND GUARD ON";
        badge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/40";
      }
      appendIncidentLog("INFO", "Switched to study tab — Background Webcam & Audio Sentinel remains active!");
    } else {
      if (badge) {
        badge.textContent = "TAB: ACTIVE";
        badge.className = "text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30";
      }
      document.title = "SpineGuard AI — Real-Time Posture & Drowsiness Corrector | © Naresh";
    }
  });

  // --- Initialize Webcam ---
  async function initWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false
      });
      webcamVideo.srcObject = stream;
      await webcamVideo.play();

      // Prepare PiP stream
      if (pipCanvas.captureStream) {
        pipVideo.srcObject = pipCanvas.captureStream(15);
        pipVideo.play().catch(() => {});
      }
    } catch (err) {
      appendIncidentLog("CRITICAL", "Camera permission denied or no webcam found. Please allow camera access!");
      document.getElementById("liveStatusText").textContent = "CAMERA ACCESS REQUIRED";
    }
  }

  // --- UI Event Listeners ---
  document.getElementById("btnToggleCamera")?.addEventListener("click", () => {
    isTrackingPaused = !isTrackingPaused;
    const btnText = document.getElementById("btnCameraText");
    const btnIcon = document.getElementById("btnCameraIcon");
    const statusText = document.getElementById("liveStatusText");
    if (isTrackingPaused) {
      btnText.textContent = "Resume Guard";
      btnIcon.textContent = "▶";
      statusText.textContent = "GUARD PAUSED";
    } else {
      btnText.textContent = "Pause Guard";
      btnIcon.textContent = "⏸";
      statusText.textContent = "CAMERA ACTIVE • BACKGROUND TAB GUARD ON";
    }
  });

  document.getElementById("btnCalibrate")?.addEventListener("click", () => {
    forceCalibrateNext = true;
    const calBadge = document.getElementById("calibrationBadge");
    if (calBadge) {
      calBadge.textContent = "Calibrated Upright ✓";
      calBadge.className = "px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
    }
    appendIncidentLog("INFO", "Baseline upright posture calibrated to your current sitting position!");
  });

  document.getElementById("btnEnableNotifications")?.addEventListener("click", () => {
    requestNotificationPermission();
    triggerDesktopNotification({
      title: "Laptop Study Notifications Enabled!",
      message: "You will now receive instant alerts even when switching tabs to study."
    });
  });

  document.getElementById("btnTestAlert")?.addEventListener("click", () => {
    lastAlertTimestamp = 0;
    triggerAudioAlert({
      type: "SLOUCH",
      severity: "critical",
      title: "TEST AUDIO ALERT",
      message: "Sit up straight! Audio alert verified."
    });
  });

  document.getElementById("btnAcknowledgeAlert")?.addEventListener("click", () => {
    videoAlertOverlay.classList.add("hidden");
    emergencyBanner.classList.add("hidden");
  });

  document.getElementById("btnClearLog")?.addEventListener("click", () => {
    const list = document.getElementById("alertHistoryList");
    if (list) list.innerHTML = "";
  });

  // Floating Picture-in-Picture (PiP) Toggle
  document.getElementById("btnPipToggle")?.addEventListener("click", async () => {
    try {
      if (document.pictureInPictureElement) {
        await document.exitPictureInPicture();
      } else {
        updatePipCanvas(latestTelemetry);
        await pipVideo.play();
        await pipVideo.requestPictureInPicture();
      }
    } catch (e) {
      appendIncidentLog("INFO", "Tip: Keep this browser tab un-minimized while switching tabs to study!");
    }
  });

  // Audio Mode Toggle Buttons (Funny Roast vs Tactical Siren vs Custom MP3)
  document.querySelectorAll(".audio-mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".audio-mode-btn").forEach((b) => b.classList.remove("active-mode"));
      btn.classList.add("active-mode");
      audioMode = btn.getAttribute("data-audio-mode") || "funny";
      appendIncidentLog("INFO", `Switched audio alert mode to: ${audioMode.toUpperCase()}`);
    });
  });

  // Custom MP3/WAV File Upload
  document.getElementById("customAudioInput")?.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (customAudioUrl) URL.revokeObjectURL(customAudioUrl);
    customAudioUrl = URL.createObjectURL(file);
    customAudioElement = new Audio(customAudioUrl);
    audioMode = "custom";
    document.querySelectorAll(".audio-mode-btn").forEach((b) => {
      b.classList.toggle("active-mode", b.getAttribute("data-audio-mode") === "custom");
    });
    appendIncidentLog("INFO", `Loaded custom alert audio: ${file.name}`);
  });

  // Sensitivity Sliders
  const sliderEar = document.getElementById("sliderEar");
  const sliderPosture = document.getElementById("sliderPosture");
  function syncThresholds() {
    const earVal = parseFloat(sliderEar.value);
    const posVal = parseFloat(sliderPosture.value);
    document.getElementById("lblEarThresh").textContent = earVal.toFixed(2);
    document.getElementById("lblPostureSens").textContent = `${posVal.toFixed(1)}x`;
    if (wsConnected && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: "thresholds",
        ear_threshold: earVal,
        mar_threshold: 0.58,
        posture_sensitivity: posVal
      }));
    }
  }
  sliderEar?.addEventListener("input", syncThresholds);
  sliderPosture?.addEventListener("input", syncThresholds);

  // --- Auth Modal & Globally Unique Password Handlers ---
  const authModal = document.getElementById("authModal");
  const formSignup = document.getElementById("formSignup");
  const formLogin = document.getElementById("formLogin");
  const tabBtnSignup = document.getElementById("tabBtnSignup");
  const tabBtnLogin = document.getElementById("tabBtnLogin");

  document.getElementById("btnOpenAuth")?.addEventListener("click", () => {
    authModal.classList.remove("hidden");
  });

  document.getElementById("btnCloseAuthModal")?.addEventListener("click", () => {
    if (isTrialLocked && !currentUser) {
      alert("Your 5-minute free trial has ended! Please sign up or log in to continue using SpineGuard AI.");
      return;
    }
    authModal.classList.add("hidden");
  });

  tabBtnSignup?.addEventListener("click", () => {
    formSignup.classList.remove("hidden");
    formLogin.classList.add("hidden");
    tabBtnSignup.className = "py-2 rounded-lg text-xs font-bold bg-cyan-500 text-slate-950 transition";
    tabBtnLogin.className = "py-2 rounded-lg text-xs font-bold text-slate-400 hover:text-white transition";
  });

  tabBtnLogin?.addEventListener("click", () => {
    formLogin.classList.remove("hidden");
    formSignup.classList.add("hidden");
    tabBtnLogin.className = "py-2 rounded-lg text-xs font-bold bg-emerald-400 text-slate-950 transition";
    tabBtnSignup.className = "py-2 rounded-lg text-xs font-bold text-slate-400 hover:text-white transition";
  });

  // Generate 100% Unique Strong Password Helper
  document.getElementById("btnGenerateUniquePw")?.addEventListener("click", () => {
    const rand = Math.random().toString(36).slice(2, 7).toUpperCase();
    const num = Math.floor(100 + Math.random() * 900);
    const pwInput = document.getElementById("regPassword");
    pwInput.type = "text";
    pwInput.value = `Spine#${rand}_${num}!a`;
    pwInput.dispatchEvent(new Event("input"));
  });

  // Live Password Uniqueness Check
  let pwCheckTimer = null;
  document.getElementById("regPassword")?.addEventListener("input", (e) => {
    const val = e.target.value;
    const fb = document.getElementById("pwUniqueFeedback");
    clearTimeout(pwCheckTimer);
    if (!val) {
      fb.textContent = "Every user must have a 100% unique password not used by anyone else.";
      fb.className = "text-[11px] font-mono text-slate-400 mt-1";
      return;
    }
    pwCheckTimer = setTimeout(async () => {
      try {
        const res = await fetch("/api/auth/check-password", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ password: val })
        });
        const data = await res.json();
        fb.textContent = data.message;
        fb.className = data.unique
          ? "text-[11px] font-mono text-emerald-400 mt-1"
          : "text-[11px] font-mono text-rose-400 mt-1";
      } catch (_) {}
    }, 280);
  });

  // Submit Sign Up
  formSignup?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errBox = document.getElementById("signupErrorBox");
    errBox.classList.add("hidden");
    const username = document.getElementById("regUsername").value.trim();
    const email = document.getElementById("regEmail").value.trim();
    const password = document.getElementById("regPassword").value;

    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password })
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        errBox.textContent = data.error || "Sign up failed.";
        errBox.classList.remove("hidden");
        return;
      }
      currentUser = data.user;
      localStorage.setItem("spineguard_user", JSON.stringify(currentUser));
      updateAuthUI();
      authModal.classList.add("hidden");
      appendIncidentLog("INFO", `Registered & logged in as @${currentUser.username}! Unlimited access unlocked.`);
    } catch (err) {
      errBox.textContent = "Network error while creating account.";
      errBox.classList.remove("hidden");
    }
  });

  // Submit Login
  formLogin?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errBox = document.getElementById("loginErrorBox");
    errBox.classList.add("hidden");
    const identifier = document.getElementById("loginIdentifier").value.trim();
    const password = document.getElementById("loginPassword").value;

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identifier, password })
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        errBox.textContent = data.error || "Invalid credentials.";
        errBox.classList.remove("hidden");
        return;
      }
      currentUser = data.user;
      localStorage.setItem("spineguard_user", JSON.stringify(currentUser));
      updateAuthUI();
      authModal.classList.add("hidden");
      appendIncidentLog("INFO", `Welcome back @${currentUser.username}! Unlimited session active.`);
    } catch (err) {
      errBox.textContent = "Network error while logging in.";
      errBox.classList.remove("hidden");
    }
  });

  // --- Supabase Connection Modal ---
  const supabaseModal = document.getElementById("supabaseModal");
  document.getElementById("btnSupabaseModal")?.addEventListener("click", () => {
    supabaseModal.classList.remove("hidden");
  });
  document.getElementById("btnCloseSupabaseModal")?.addEventListener("click", () => {
    supabaseModal.classList.add("hidden");
  });

  document.getElementById("btnCopySql")?.addEventListener("click", () => {
    const sql = document.getElementById("sqlSchemaBlock").textContent;
    navigator.clipboard.writeText(sql);
    document.getElementById("btnCopySql").textContent = "Copied! ✓";
    setTimeout(() => { document.getElementById("btnCopySql").textContent = "Copy SQL"; }, 2000);
  });

  async function refreshSupabaseStatus() {
    try {
      const res = await fetch("/api/supabase/status");
      const st = await res.json();
      const dot = document.getElementById("supabaseStatusDot");
      const lbl = document.getElementById("supabaseBtnLabel");
      if (st.supabase_connected) {
        dot.className = "w-2 h-2 rounded-full bg-emerald-400";
        lbl.textContent = "Supabase: Connected";
      } else {
        dot.className = "w-2 h-2 rounded-full bg-amber-400";
        lbl.textContent = "Connect Supabase";
      }
    } catch (_) {}
  }

  document.getElementById("formSupabaseConnect")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = document.getElementById("inputSupabaseUrl").value.trim();
    const key = document.getElementById("inputSupabaseKey").value.trim();
    const fb = document.getElementById("supabaseConnectFeedback");
    fb.classList.remove("hidden");
    fb.className = "p-2.5 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 text-xs font-mono";
    fb.textContent = "Connecting to Supabase...";

    try {
      const res = await fetch("/api/supabase/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ supabase_url: url, supabase_key: key })
      });
      const data = await res.json();
      await refreshSupabaseStatus();
      if (data.supabase_connected) {
        fb.className = "p-2.5 rounded-xl bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 text-xs font-mono";
        fb.textContent = data.connection_error
          ? `Connected! Note: ${data.connection_error}`
          : "Connected & verified `app_users` table in Supabase!";
      } else {
        fb.className = "p-2.5 rounded-xl bg-rose-500/15 border border-rose-500/40 text-rose-300 text-xs font-mono";
        fb.textContent = data.connection_error || "Could not connect to Supabase. Check URL and Key.";
      }
    } catch (err) {
      fb.className = "p-2.5 rounded-xl bg-rose-500/15 border border-rose-500/40 text-rose-300 text-xs font-mono";
      fb.textContent = "Failed to connect to server.";
    }
  });

  // --- Boot Sequence ---
  updateAuthUI();
  refreshSupabaseStatus();
  requestNotificationPermission();
  initWebcam();
  connectWebSocket();
  startUnthrottledWorkerTimer();
  drawTelemetryChart();
})();
