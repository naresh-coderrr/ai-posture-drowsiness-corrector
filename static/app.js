/**
 * SpineGuard AI — User-Friendly Real-Time Posture, Drowsiness & True Mobile Phone Detector
 * © 2026 Naresh (naresh-coderrr). All Rights Reserved.
 */

(() => {
  const SESSION_ID = localStorage.getItem("spineguard_session_id") || ("sess_" + Math.random().toString(36).substring(2, 11));
  localStorage.setItem("spineguard_session_id", SESSION_ID);

  // 5-Minute Free Trial (300 seconds)
  const FREE_TRIAL_TOTAL_SEC = 5 * 60;
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
  let latestTelemetry = null;

  // Funny Voice Roasts
  const FUNNY_VOICE_LINES = {
    DROWSY: [
      "Wake up buddy! Your eyes just closed! Sit up straight right now!",
      "Excuse me! Sleeping on your desk will not pass your exams! Open those eyes!",
      "Hey! Gravity is winning against your eyelids! Sit up and lock back in!"
    ],
    SLOUCH: [
      "Whoa there shrimp posture! Pull your shoulders back and sit up straight!",
      "Posture check! You look like a question mark right now. Sit up straight like a CEO!",
      "Your spine called—it says stop slouching before you turn into a banana!"
    ],
    PHONE_DISTRACTION: [
      "Caught red-handed with your mobile phone! Put the phone down and get back to studying!",
      "Step away from the mobile phone! Scrolling Reels won't finish your study session!",
      "Phone detected! Put your mobile away and eyes back on the laptop!"
    ],
    DISTRACTED: [
      "Hey! Where are you looking? Eyes back on your study screen!"
    ]
  };

  // DOM References
  const webcamVideo = document.getElementById("webcamVideo");
  const hudCanvas = document.getElementById("hudCanvas");
  const hudCtx = hudCanvas.getContext("2d");
  const captureCanvas = document.getElementById("captureCanvas");
  const captureCtx = captureCanvas.getContext("2d");
  const pipCanvas = document.getElementById("pipCanvas");
  const pipCtx = pipCanvas.getContext("2d");
  const pipVideo = document.getElementById("pipVideo");

  const trialBadgeContainer = document.getElementById("trialBadgeContainer");
  const trialLabelText = document.getElementById("trialLabelText");
  const trialCountdownText = document.getElementById("trialCountdownText");
  const navUserText = document.getElementById("navUserText");

  const emergencyBanner = document.getElementById("emergencyBanner");
  const emergencyBannerTitle = document.getElementById("emergencyBannerTitle");
  const videoViewport = document.getElementById("videoViewport");
  const videoAlertOverlay = document.getElementById("videoAlertOverlay");
  const videoAlertTitle = document.getElementById("videoAlertTitle");
  const videoAlertDesc = document.getElementById("videoAlertDesc");

  // --- Web Audio API & Background Keep-Alive ---
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

  function startBackgroundKeepAliveAudio() {
    try {
      const ctx = ensureAudioContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      gain.gain.value = 0.00001;
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
    } catch (_) {}
  }

  document.addEventListener("click", () => {
    ensureAudioContext();
    startBackgroundKeepAliveAudio();
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission();
    }
  }, { once: true });

  function playLoudAlarm() {
    try {
      const ctx = ensureAudioContext();
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(600, now);
      osc.frequency.linearRampToValueAtTime(980, now + 0.22);
      osc.frequency.linearRampToValueAtTime(540, now + 0.45);
      osc.frequency.linearRampToValueAtTime(1020, now + 0.70);
      gain.gain.setValueAtTime(0.30, now);
      gain.gain.exponentialRampToValueAtTime(0.008, now + 0.85);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.85);
    } catch (_) {}
  }

  function playFunnyChime() {
    try {
      const ctx = ensureAudioContext();
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "triangle";
      osc.frequency.setValueAtTime(340, now);
      osc.frequency.exponentialRampToValueAtTime(780, now + 0.16);
      osc.frequency.exponentialRampToValueAtTime(440, now + 0.32);
      gain.gain.setValueAtTime(0.25, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.35);
    } catch (_) {}
  }

  function triggerAudioAlert(alertObj) {
    if (document.getElementById("chkAudioMute")?.checked) return;
    const now = Date.now();
    if (now - lastAlertTimestamp < 3200) return;
    lastAlertTimestamp = now;

    if (audioMode === "custom" && customAudioElement) {
      customAudioElement.currentTime = 0;
      customAudioElement.play().catch(() => playLoudAlarm());
      return;
    }

    if (audioMode === "alarm") {
      playLoudAlarm();
      return;
    }

    playFunnyChime();
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const lines = FUNNY_VOICE_LINES[alertObj.type] || FUNNY_VOICE_LINES.SLOUCH;
      const line = lines[Math.floor(Math.random() * lines.length)];
      const utter = new SpeechSynthesisUtterance(line);
      utter.rate = 1.06;
      utter.pitch = 1.1;
      window.speechSynthesis.speak(utter);
    } else {
      playLoudAlarm();
    }
  }

  function triggerDesktopNotification(alertObj) {
    const now = Date.now();
    const cooldown = document.hidden ? 3500 : 7500;
    if (now - lastNotificationTimestamp < cooldown) return;
    lastNotificationTimestamp = now;

    if (document.hidden) {
      document.title = `🚨 ${alertObj.title} — SpineGuard AI`;
    }

    if ("Notification" in window && Notification.permission === "granted") {
      try {
        const notif = new Notification(`🚨 ${alertObj.title}`, {
          body: alertObj.message,
          tag: "spineguard-alert",
          renotify: true,
        });
        notif.onclick = () => {
          window.focus();
          notif.close();
        };
      } catch (_) {}
    }
  }

  // --- 5-Minute Free Trial & Auth ---
  function formatMMSS(totalSec) {
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function updateAuthUI() {
    if (currentUser) {
      isTrialLocked = false;
      navUserText.textContent = `@${currentUser.username}`;
      trialBadgeContainer.className = "flex items-center gap-2 px-3.5 py-2 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-semibold";
      trialLabelText.textContent = "Member:";
      trialCountdownText.textContent = "Unlimited";
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
    trialCountdownText.textContent = "00:00";
    document.getElementById("trialExpiredNotice")?.classList.remove("hidden");
    document.getElementById("authModal")?.classList.remove("hidden");
  }

  setInterval(() => {
    if (currentUser || isTrackingPaused || isTrialLocked) return;
    if (trialRemainingSec > 0) {
      trialRemainingSec -= 1;
      localStorage.setItem("spineguard_trial_remaining", String(trialRemainingSec));
      trialCountdownText.textContent = formatMMSS(trialRemainingSec);
    } else {
      lockForTrialExpiration();
    }
  }, 1000);

  // --- Draw Clean Visual Overlays on Video (Eyes, Shoulders, and Phone Bounding Box) ---
  function drawVideoOverlay(data) {
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

    // 1. Subtle Eye Outlines
    const eyeColor = data.ear.avg < data.ear.threshold ? "#f43f5e" : "#22d3ee";
    [ov.left_eye, ov.right_eye].forEach((eyePts) => {
      if (!eyePts || eyePts.length === 0) return;
      hudCtx.beginPath();
      eyePts.forEach((pt, idx) => {
        if (idx === 0) hudCtx.moveTo(pt[0] * w, pt[1] * h);
        else hudCtx.lineTo(pt[0] * w, pt[1] * h);
      });
      hudCtx.closePath();
      hudCtx.strokeStyle = eyeColor;
      hudCtx.lineWidth = 2;
      hudCtx.stroke();
    });

    // 2. Shoulder Posture Line
    if (ov.pose && ov.pose.length >= 3) {
      const map = {};
      ov.pose.forEach((p) => { map[p.id] = p; });
      const lSh = map.left_shoulder;
      const rSh = map.right_shoulder;
      if (lSh && rSh) {
        const spineColor = data.posture.is_slouching ? "#f43f5e" : "#10b981";
        hudCtx.beginPath();
        hudCtx.moveTo(lSh.x * w, lSh.y * h);
        hudCtx.lineTo(rSh.x * w, rSh.y * h);
        hudCtx.strokeStyle = spineColor;
        hudCtx.lineWidth = 4;
        hudCtx.stroke();

        [lSh, rSh].forEach((joint) => {
          hudCtx.beginPath();
          hudCtx.arc(joint.x * w, joint.y * h, 6, 0, Math.PI * 2);
          hudCtx.fillStyle = spineColor;
          hudCtx.fill();
        });
      }
    }

    // 3. Mobile Phone Bounding Box (Only drawn when actual cell phone is detected!)
    if (ov.phone_boxes && ov.phone_boxes.length > 0) {
      ov.phone_boxes.forEach((box) => {
        const bx = box.x * w;
        const by = box.y * h;
        const bw = box.w * w;
        const bh = box.h * h;
        hudCtx.strokeStyle = "#f43f5e";
        hudCtx.lineWidth = 3.5;
        hudCtx.strokeRect(bx, by, bw, bh);

        hudCtx.fillStyle = "rgba(244, 63, 94, 0.22)";
        hudCtx.fillRect(bx, by, bw, bh);
      });
    }
  }

  // --- Update Floating Mini Study Window (PiP) ---
  function updatePipCanvas(data) {
    const w = pipCanvas.width;
    const h = pipCanvas.height;
    pipCtx.fillStyle = "#090d16";
    pipCtx.fillRect(0, 0, w, h);

    if (webcamVideo.readyState >= 2) {
      pipCtx.save();
      pipCtx.translate(w, 0);
      pipCtx.scale(-1, 1);
      pipCtx.drawImage(webcamVideo, 0, 0, w, h);
      pipCtx.restore();
    }

    pipCtx.fillStyle = "rgba(9, 13, 22, 0.88)";
    pipCtx.fillRect(0, h - 56, w, 56);

    if (data) {
      const isAlert = Boolean(data.active_alert);
      pipCtx.fillStyle = isAlert ? "#f43f5e" : "#10b981";
      pipCtx.font = "bold 15px sans-serif";
      pipCtx.fillText(
        isAlert ? `🚨 ${data.active_alert.title}` : `✅ Sitting Straight (${data.posture.spine_score}%)`,
        14,
        h - 30
      );
      pipCtx.fillStyle = "#cbd5e1";
      pipCtx.font = "12px sans-serif";
      pipCtx.fillText(
        isAlert ? data.active_alert.message.slice(0, 46) : `Blinks: ${data.blinks.total} • No Phone Detected`,
        14,
        h - 11
      );
    }
  }

  // --- Update User-Friendly Dashboard UI ---
  function updateDashboardUI(data) {
    latestTelemetry = data;

    // Video Top-Left Pills
    document.getElementById("hudBlinkCount").textContent = data.blinks.total;
    document.getElementById("hudSpineScore").textContent = `${data.posture.spine_score}%`;

    // 1. Posture Card
    document.getElementById("cardSpineScore").textContent = `${data.posture.spine_score}%`;
    const cardPostureStatus = document.getElementById("cardPostureStatus");
    const cardPostureSub = document.getElementById("cardPostureSub");
    const cardBoxPosture = document.getElementById("cardBoxPosture");
    if (data.posture.is_slouching) {
      cardPostureStatus.textContent = "Slouching — Sit Up!";
      cardPostureStatus.className = "text-lg font-extrabold text-rose-400";
      cardPostureSub.textContent = data.posture.reasons[0] || "Straighten your back and neck";
      cardBoxPosture.className = "bg-rose-950/30 rounded-3xl p-4 border border-rose-500/50 flex items-center justify-between gap-4 transition";
    } else {
      cardPostureStatus.textContent = "Sitting Straight";
      cardPostureStatus.className = "text-lg font-extrabold text-emerald-400";
      cardPostureSub.textContent = "Shoulders balanced & aligned";
      cardBoxPosture.className = "bg-[#111827] rounded-3xl p-4 border border-slate-800 flex items-center justify-between gap-4 transition";
    }

    // 2. Eyes & Blink Card
    document.getElementById("cardEarValue").textContent = data.ear.avg.toFixed(2);
    document.getElementById("cardTotalBlinks").textContent = data.blinks.total;
    document.getElementById("cardBlinkRate").textContent = `${data.blinks.per_minute} / min`;
    const cardEyeStatus = document.getElementById("cardEyeStatus");
    if (data.ear.closed_seconds >= 1.0) {
      cardEyeStatus.textContent = "Eyes Closing / Drowsy!";
      cardEyeStatus.className = "text-lg font-extrabold text-rose-400";
    } else if (data.mar.value > data.mar.threshold) {
      cardEyeStatus.textContent = "Yawning Detected";
      cardEyeStatus.className = "text-lg font-extrabold text-amber-300";
    } else {
      cardEyeStatus.textContent = "Awake & Focused";
      cardEyeStatus.className = "text-lg font-extrabold text-cyan-300";
    }

    // 3. Mobile Phone Card (True AI Object Detector)
    const cardPhoneState = document.getElementById("cardPhoneState");
    const cardPhoneReason = document.getElementById("cardPhoneReason");
    const cardPhoneBadge = document.getElementById("cardPhoneBadge");
    const cardBoxPhone = document.getElementById("cardBoxPhone");
    const iconBoxPhone = document.getElementById("iconBoxPhone");

    if (data.distraction.phone_detected) {
      cardPhoneState.textContent = "Mobile Phone Spotted!";
      cardPhoneState.className = "text-lg font-extrabold text-rose-400";
      cardPhoneReason.textContent = `Phone detected (${Math.round(data.distraction.phone_confidence)}% confidence) — Put it away!`;
      cardPhoneBadge.textContent = "Alert!";
      cardPhoneBadge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-rose-300";
      cardBoxPhone.className = "bg-rose-950/30 rounded-3xl p-4 border border-rose-500/50 flex items-center justify-between gap-4 transition";
      iconBoxPhone.textContent = "📱";
    } else {
      cardPhoneState.textContent = "No Phone Detected";
      cardPhoneState.className = "text-lg font-extrabold text-emerald-400";
      cardPhoneReason.textContent = "Hands & palms ignored — only real phones trigger alerts";
      cardPhoneBadge.textContent = "Clear";
      cardPhoneBadge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-500/15 text-emerald-300";
      cardBoxPhone.className = "bg-[#111827] rounded-3xl p-4 border border-slate-800 flex items-center justify-between gap-4 transition";
      iconBoxPhone.textContent = "📵";
    }

    drawVideoOverlay(data);
    updatePipCanvas(data);

    // Friendly Live Header Status & Alert Trigger
    const liveStatusText = document.getElementById("liveStatusText");
    const liveBadgeDot = document.getElementById("liveBadgeDot");

    if (data.active_alert) {
      const al = data.active_alert;
      liveStatusText.textContent = `${al.title} — ${al.message}`;
      liveStatusText.className = "text-sm font-bold text-rose-400";
      liveBadgeDot.className = "w-3 h-3 rounded-full bg-rose-500 animate-ping";

      emergencyBanner.classList.remove("hidden");
      emergencyBannerTitle.textContent = `${al.title}: ${al.message}`;

      videoViewport.classList.add("alert-ring-critical");
      videoAlertOverlay.classList.remove("hidden");
      videoAlertTitle.textContent = al.title;
      videoAlertDesc.textContent = al.message;

      triggerAudioAlert(al);
      triggerDesktopNotification(al);
    } else {
      liveStatusText.textContent = "You're doing great! Keep sitting straight 👍";
      liveStatusText.className = "text-sm font-bold text-emerald-300";
      liveBadgeDot.className = "w-3 h-3 rounded-full bg-emerald-400 animate-pulse";

      emergencyBanner.classList.add("hidden");
      videoViewport.classList.remove("alert-ring-critical");
      videoAlertOverlay.classList.add("hidden");
      if (!document.hidden) {
        document.title = "SpineGuard AI — Posture, Drowsiness & Phone Guard | © Naresh";
      }
    }
  }

  // --- WebSocket & Background Tab Worker ---
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
    const dataUri = captureCanvas.toDataURL("image/jpeg", 0.65);
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

  function startUnthrottledWorkerTimer() {
    const workerCode = `
      let intervalId = null;
      self.onmessage = function(e) {
        if (e.data === 'start') {
          if (intervalId) clearInterval(intervalId);
          intervalId = setInterval(() => self.postMessage('tick'), 120);
        }
      };
    `;
    const blob = new Blob([workerCode], { type: "application/javascript" });
    const worker = new Worker(URL.createObjectURL(blob));
    worker.onmessage = () => captureAndSendFrame();
    worker.postMessage("start");
  }

  document.addEventListener("visibilitychange", () => {
    const badge = document.getElementById("tabVisibilityBadge");
    if (document.hidden) {
      if (badge) {
        badge.textContent = "Watching in Background 🟢";
        badge.className = "text-xs font-bold px-2.5 py-1 rounded-full bg-indigo-500/20 text-indigo-300";
      }
    } else {
      if (badge) {
        badge.textContent = "Background Study Guard: Ready";
        badge.className = "text-xs font-medium px-2.5 py-1 rounded-full bg-slate-800 text-slate-300";
      }
      document.title = "SpineGuard AI — Posture, Drowsiness & Phone Guard | © Naresh";
    }
  });

  async function initWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false
      });
      webcamVideo.srcObject = stream;
      await webcamVideo.play();
      if (pipCanvas.captureStream) {
        pipVideo.srcObject = pipCanvas.captureStream(15);
        pipVideo.play().catch(() => {});
      }
    } catch (_) {
      document.getElementById("liveStatusText").textContent = "Please allow camera access in your browser!";
    }
  }

  // --- Button Handlers ---
  document.getElementById("btnCalibrate")?.addEventListener("click", () => {
    forceCalibrateNext = true;
    const btn = document.getElementById("btnCalibrate");
    btn.innerHTML = "<span>✅</span><span>Upright Posture Saved!</span>";
    setTimeout(() => {
      btn.innerHTML = "<span>🎯</span><span>Set My Upright Posture</span>";
    }, 2200);
  });

  document.getElementById("btnToggleCamera")?.addEventListener("click", () => {
    isTrackingPaused = !isTrackingPaused;
    document.getElementById("btnCameraText").textContent = isTrackingPaused ? "Resume Tracking" : "Pause Tracking";
    document.getElementById("btnCameraIcon").textContent = isTrackingPaused ? "▶" : "⏸";
    document.getElementById("liveStatusText").textContent = isTrackingPaused
      ? "Tracking Paused"
      : "You're doing great! Keep sitting straight 👍";
  });

  document.getElementById("btnTestAlert")?.addEventListener("click", () => {
    lastAlertTimestamp = 0;
    triggerAudioAlert({
      type: "SLOUCH",
      severity: "critical",
      title: "TEST SOUND",
      message: "Sit up straight!"
    });
  });

  document.getElementById("btnAcknowledgeAlert")?.addEventListener("click", () => {
    videoAlertOverlay.classList.add("hidden");
    emergencyBanner.classList.add("hidden");
  });

  document.getElementById("btnPipToggle")?.addEventListener("click", async () => {
    try {
      if (document.pictureInPictureElement) {
        await document.exitPictureInPicture();
      } else {
        updatePipCanvas(latestTelemetry);
        await pipVideo.play();
        await pipVideo.requestPictureInPicture();
      }
    } catch (_) {}
  });

  document.querySelectorAll(".audio-mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".audio-mode-btn").forEach((b) => b.classList.remove("active-mode"));
      btn.classList.add("active-mode");
      audioMode = btn.getAttribute("data-audio-mode") || "funny";
    });
  });

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
  });

  // --- Auth Modal & Unique Password Handlers ---
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
      alert("Your 5-minute free trial has ended! Please sign up or log in to continue.");
      return;
    }
    authModal.classList.add("hidden");
  });

  tabBtnSignup?.addEventListener("click", () => {
    formSignup.classList.remove("hidden");
    formLogin.classList.add("hidden");
    tabBtnSignup.className = "py-2.5 rounded-xl text-xs font-bold bg-indigo-500 text-white transition";
    tabBtnLogin.className = "py-2.5 rounded-xl text-xs font-bold text-slate-400 hover:text-white transition";
  });

  tabBtnLogin?.addEventListener("click", () => {
    formLogin.classList.remove("hidden");
    formSignup.classList.add("hidden");
    tabBtnLogin.className = "py-2.5 rounded-xl text-xs font-bold bg-emerald-500 text-slate-950 transition";
    tabBtnSignup.className = "py-2.5 rounded-xl text-xs font-bold text-slate-400 hover:text-white transition";
  });

  document.getElementById("btnGenerateUniquePw")?.addEventListener("click", () => {
    const rand = Math.random().toString(36).slice(2, 7).toUpperCase();
    const num = Math.floor(100 + Math.random() * 900);
    const pwInput = document.getElementById("regPassword");
    pwInput.type = "text";
    pwInput.value = `Spine#${rand}_${num}!a`;
    pwInput.dispatchEvent(new Event("input"));
  });

  let pwCheckTimer = null;
  document.getElementById("regPassword")?.addEventListener("input", (e) => {
    const val = e.target.value;
    const fb = document.getElementById("pwUniqueFeedback");
    clearTimeout(pwCheckTimer);
    if (!val) {
      fb.textContent = "Must be unique across all users in Supabase.";
      fb.className = "text-xs text-slate-400 mt-1";
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
        fb.className = data.unique ? "text-xs text-emerald-400 mt-1 font-semibold" : "text-xs text-rose-400 mt-1 font-semibold";
      } catch (_) {}
    }, 260);
  });

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
    } catch (_) {
      errBox.textContent = "Network error while creating account.";
      errBox.classList.remove("hidden");
    }
  });

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
    } catch (_) {
      errBox.textContent = "Network error while logging in.";
      errBox.classList.remove("hidden");
    }
  });

  updateAuthUI();
  initWebcam();
  connectWebSocket();
  startUnthrottledWorkerTimer();
})();
