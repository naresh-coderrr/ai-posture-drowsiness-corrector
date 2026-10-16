/**
 * SpineGuard AI — Hybrid Real-Time Posture, Drowsiness & True Mobile Phone Detector
 * Supports:
 * 1) Local Python MediaPipe Backend (WebSocket / HTTP)
 * 2) Vercel / Cloud Serverless Mode via official @mediapipe/tasks-vision WebAssembly + WebGL Engine!
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

  // Hybrid Engine State: Uses Python backend locally, or automatically activates Browser MediaPipe Tasks on Vercel!
  let useClientMediaPipe = false;
  let clientModelsReady = false;
  let clientModelsLoading = false;
  let mpFaceLandmarker = null;
  let mpPoseLandmarker = null;
  let mpObjectDetector = null;

  // Client-Side MediaPipe Stateful Session (Matches vision_engine.py 1:1)
  const clientSession = {
    blink_count: 0,
    yawn_count: 0,
    eye_closed_start: null,
    yawn_start: null,
    slouch_start: null,
    phone_start: null,
    distracted_start: null,
    was_blinking: false,
    was_yawning: false,
    baseline_neck_ratio: null,
    baseline_shoulder_y: null,
    calibrated: false,
    blink_timestamps: [],
    ear_threshold: 0.22,
    mar_threshold: 0.58,
    posture_sensitivity: 1.0,
  };

  const LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380];
  const RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144];
  const MOUTH_IDX = [61, 37, 13, 267, 291, 314, 14, 84];

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

  // --- Load Official @mediapipe/tasks-vision in Browser when running on Vercel Serverless ---
  async function initClientMediaPipeModels() {
    if (clientModelsReady || clientModelsLoading) return;
    clientModelsLoading = true;
    try {
      const vision = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs");
      const filesetResolver = await vision.FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );

      const [faceLm, poseLm, objDet] = await Promise.all([
        vision.FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          },
          runningMode: "IMAGE",
          numFaces: 1,
          minFaceDetectionConfidence: 0.45,
          minFacePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        }),
        vision.PoseLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          },
          runningMode: "IMAGE",
          numPoses: 1,
          minPoseDetectionConfidence: 0.45,
          minPosePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        }),
        vision.ObjectDetector.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
          },
          runningMode: "IMAGE",
          scoreThreshold: 0.35,
          categoryAllowlist: ["cell phone", "remote"],
          maxResults: 3,
        }),
      ]);

      mpFaceLandmarker = faceLm;
      mpPoseLandmarker = poseLm;
      mpObjectDetector = objDet;
      clientModelsReady = true;
    } catch (err) {
      console.error("Failed to load MediaPipe WebAssembly models:", err);
    } finally {
      clientModelsLoading = false;
    }
  }

  function dist2D(p1, p2) {
    return Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
  }

  function calcEAR(landmarks, indices, w, h) {
    const pts = indices.map((i) => [landmarks[i].x * w, landmarks[i].y * h]);
    const v1 = dist2D(pts[1], pts[5]);
    const v2 = dist2D(pts[2], pts[4]);
    const horiz = dist2D(pts[0], pts[3]);
    if (horiz < 1e-6) return 0.30;
    return (v1 + v2) / (2.0 * horiz);
  }

  function calcMAR(landmarks, indices, w, h) {
    const pts = indices.map((i) => [landmarks[i].x * w, landmarks[i].y * h]);
    const v1 = dist2D(pts[1], pts[7]);
    const v2 = dist2D(pts[2], pts[6]);
    const v3 = dist2D(pts[3], pts[5]);
    const horiz = dist2D(pts[0], pts[4]);
    if (horiz < 1e-6) return 0.0;
    return (v1 + v2 + v3) / (2.0 * horiz);
  }

  function analyzeFrameClientSide(canvasEl, forceCalibrate = false) {
    const sess = clientSession;
    const now = Date.now() / 1000.0;
    const w = canvasEl.width;
    const h = canvasEl.height;

    const faceRes = mpFaceLandmarker.detect(canvasEl);
    const poseRes = mpPoseLandmarker.detect(canvasEl);
    const objRes = mpObjectDetector.detect(canvasEl);

    let ear_left = 0.30, ear_right = 0.30, ear_avg = 0.30, mar = 0.15;
    let head_yaw = 0.0, head_pitch = 0.0, head_roll = 0.0;
    let face_detected = false;
    let spine_score = 100, neck_angle = 90.0, shoulder_tilt = 0.0;
    const slouch_reasons = [];
    let phone_detected = false, phone_confidence = 0.0;
    const phone_boxes = [];
    let distracted_detected = false, distraction_reason = "";

    const overlay_points = {
      left_eye: [],
      right_eye: [],
      mouth: [],
      pose: [],
      phone_boxes: [],
    };

    if (faceRes.faceLandmarks && faceRes.faceLandmarks.length > 0) {
      face_detected = true;
      const flm = faceRes.faceLandmarks[0];
      ear_left = calcEAR(flm, LEFT_EYE_IDX, w, h);
      ear_right = calcEAR(flm, RIGHT_EYE_IDX, w, h);
      ear_avg = Number(((ear_left + ear_right) / 2.0).toFixed(3));
      mar = Number(calcMAR(flm, MOUTH_IDX, w, h).toFixed(3));

      overlay_points.left_eye = LEFT_EYE_IDX.map((i) => [flm[i].x, flm[i].y]);
      overlay_points.right_eye = RIGHT_EYE_IDX.map((i) => [flm[i].x, flm[i].y]);
      overlay_points.mouth = MOUTH_IDX.map((i) => [flm[i].x, flm[i].y]);

      const nose = flm[1], forehead = flm[10], chin = flm[152], lCheek = flm[234], rCheek = flm[454];
      const cheekMidX = (lCheek.x + rCheek.x) / 2.0;
      const cheekSpan = Math.max(0.05, Math.abs(rCheek.x - lCheek.x));
      head_yaw = Number((((nose.x - cheekMidX) / cheekSpan) * 90.0).toFixed(1));

      const faceHeight = Math.max(0.05, Math.abs(chin.y - forehead.y));
      const noseRelY = (nose.y - forehead.y) / faceHeight;
      head_pitch = Number(((noseRelY - 0.56) * 160.0).toFixed(1));

      const dyEyes = (flm[263].y - flm[33].y) * h;
      const dxEyes = (flm[263].x - flm[33].x) * w;
      head_roll = Number(((Math.atan2(dyEyes, Math.max(1.0, dxEyes)) * 180) / Math.PI).toFixed(1));

      if (ear_avg < sess.ear_threshold) {
        if (sess.eye_closed_start === null) sess.eye_closed_start = now;
        sess.was_blinking = true;
      } else {
        if (sess.was_blinking) {
          const closedDur = now - (sess.eye_closed_start || now);
          if (closedDur >= 0.04 && closedDur <= 0.65) {
            sess.blink_count += 1;
            sess.blink_timestamps.push(now);
          }
        }
        sess.was_blinking = false;
        sess.eye_closed_start = null;
      }

      sess.blink_timestamps = sess.blink_timestamps.filter((t) => now - t <= 60.0);

      if (mar > sess.mar_threshold) {
        if (sess.yawn_start === null) sess.yawn_start = now;
        else if (now - sess.yawn_start > 1.1 && !sess.was_yawning) {
          sess.yawn_count += 1;
          sess.was_yawning = true;
        }
      } else {
        sess.yawn_start = null;
        sess.was_yawning = false;
      }

      if (Math.abs(head_yaw) > 30.0) {
        distracted_detected = true;
        distraction_reason = "Looking away from screen";
      } else if (head_pitch > 24.0) {
        distracted_detected = true;
        distraction_reason = "Head drooping down";
      }
    } else {
      distracted_detected = true;
      distraction_reason = "Face not in camera view";
    }

    if (poseRes.landmarks && poseRes.landmarks.length > 0) {
      const plm = poseRes.landmarks[0];
      const nose_p = plm[0], l_sh = plm[11], r_sh = plm[12];
      overlay_points.pose = [
        { id: "nose", x: nose_p.x, y: nose_p.y },
        { id: "left_shoulder", x: l_sh.x, y: l_sh.y },
        { id: "right_shoulder", x: r_sh.x, y: r_sh.y },
      ];

      const shMidX = (l_sh.x + r_sh.x) / 2.0;
      const shMidY = (l_sh.y + r_sh.y) / 2.0;
      const shWidth = Math.max(0.10, Math.abs(l_sh.x - r_sh.x));
      const neckVerticalRatio = (shMidY - nose_p.y) / shWidth;

      if (forceCalibrate || (!sess.calibrated && neckVerticalRatio > 0.55)) {
        sess.baseline_neck_ratio = neckVerticalRatio;
        sess.baseline_shoulder_y = shMidY;
        sess.calibrated = true;
      }

      const baselineRatio = sess.baseline_neck_ratio || 0.78;
      shoulder_tilt = Number(
        ((Math.atan2((r_sh.y - l_sh.y) * h, Math.max(1.0, Math.abs(r_sh.x - l_sh.x) * w)) * 180) / Math.PI).toFixed(1)
      );

      const dxSpine = (nose_p.x - shMidX) * w;
      const dySpine = (shMidY - nose_p.y) * h;
      const axialOffsetDeg = Math.abs((Math.atan2(dxSpine, Math.max(1.0, dySpine)) * 180) / Math.PI);
      neck_angle = Number((90.0 - axialOffsetDeg).toFixed(1));

      const ratioDrop = Math.max(0.0, (baselineRatio - neckVerticalRatio) / Math.max(0.2, baselineRatio));
      const shoulderDrop = Math.max(0.0, shMidY - (sess.baseline_shoulder_y || shMidY));
      const sens = sess.posture_sensitivity;
      let penalty = 0.0;

      if (ratioDrop > 0.14 / sens) {
        penalty += Math.min(45.0, ratioDrop * 180.0 * sens);
        slouch_reasons.push("Neck slouching forward");
      }
      if (Math.abs(shoulder_tilt) > 8.0 / sens) {
        penalty += Math.min(30.0, (Math.abs(shoulder_tilt) - 5.0) * 2.8 * sens);
        slouch_reasons.push("Uneven shoulders");
      }
      if (axialOffsetDeg > 12.0 / sens) {
        penalty += Math.min(30.0, (axialOffsetDeg - 8.0) * 2.2 * sens);
        slouch_reasons.push("Leaning to one side");
      }
      if (shoulderDrop > 0.05 / sens) {
        penalty += Math.min(25.0, shoulderDrop * 300.0 * sens);
        slouch_reasons.push("Slouching down in chair");
      }
      spine_score = Math.max(0, Math.min(100, Math.round(100.0 - penalty)));
    }

    if (objRes.detections && objRes.detections.length > 0) {
      objRes.detections.forEach((det) => {
        if (!det.categories || det.categories.length === 0) return;
        const score = det.categories[0].score || 0.0;
        if (score >= 0.35 && det.boundingBox) {
          phone_detected = true;
          phone_confidence = Math.max(phone_confidence, Math.round(score * 100));
          const bb = det.boundingBox;
          phone_boxes.push({
            x: bb.originX / w,
            y: bb.originY / h,
            w: bb.width / w,
            h: bb.height / h,
            score: Math.round(score * 100),
          });
        }
      });
    }
    overlay_points.phone_boxes = phone_boxes;

    const eye_closed_duration = sess.eye_closed_start ? now - sess.eye_closed_start : 0.0;
    const is_drowsy = eye_closed_duration >= 1.35 || (sess.yawn_start !== null && now - sess.yawn_start >= 1.6);

    const is_slouching = spine_score < 68;
    if (is_slouching) {
      if (sess.slouch_start === null) sess.slouch_start = now;
    } else {
      sess.slouch_start = null;
    }
    const slouch_duration = sess.slouch_start ? now - sess.slouch_start : 0.0;

    if (phone_detected) {
      if (sess.phone_start === null) sess.phone_start = now;
    } else {
      sess.phone_start = null;
    }
    const phone_duration = sess.phone_start ? now - sess.phone_start : 0.0;

    if (distracted_detected) {
      if (sess.distracted_start === null) sess.distracted_start = now;
    } else {
      sess.distracted_start = null;
    }
    const distracted_duration = sess.distracted_start ? now - sess.distracted_start : 0.0;

    let active_alert = null;
    if (is_drowsy) {
      active_alert = {
        type: "DROWSY",
        severity: "critical",
        title: "WAKE UP! DROWSINESS DETECTED",
        message: eye_closed_duration >= 1.35
          ? `Your eyes have been closed for ${eye_closed_duration.toFixed(1)}s! Sit up straight!`
          : "Heavy yawning detected! Sit up and stay focused!",
      };
    } else if (phone_detected && phone_duration >= 0.3) {
      active_alert = {
        type: "PHONE_DISTRACTION",
        severity: "critical",
        title: "MOBILE PHONE DETECTED!",
        message: `Mobile phone spotted (${phone_confidence}% confidence). Put your phone away and focus!`,
      };
    } else if (slouch_duration >= 1.5) {
      active_alert = {
        type: "SLOUCH",
        severity: "warning",
        title: "SIT UP STRAIGHT!",
        message: `${slouch_reasons[0] || "Slouching detected"} (Posture: ${spine_score}%). Straighten your back!`,
      };
    } else if (distracted_duration >= 2.5) {
      active_alert = {
        type: "DISTRACTED",
        severity: "info",
        title: "EYES ON SCREEN!",
        message: `${distraction_reason}. Lock back into your study session!`,
      };
    }

    return {
      timestamp: now,
      face_detected,
      calibrated: sess.calibrated,
      ear: {
        left: ear_left,
        right: ear_right,
        avg: ear_avg,
        threshold: sess.ear_threshold,
        closed_seconds: eye_closed_duration,
      },
      mar: { value: mar, threshold: sess.mar_threshold },
      blinks: {
        total: sess.blink_count,
        per_minute: sess.blink_timestamps.length,
        yawns: sess.yawn_count,
      },
      posture: {
        spine_score,
        neck_angle,
        shoulder_tilt,
        head_yaw,
        head_pitch,
        head_roll,
        is_slouching,
        slouch_seconds: slouch_duration,
        reasons: slouch_reasons,
      },
      distraction: {
        phone_detected,
        phone_confidence,
        phone_seconds: phone_duration,
        phone_reason: phone_detected ? `Mobile Phone Detected (${phone_confidence}%)` : "No phone detected",
        looking_away: distracted_detected,
        distracted_seconds: distracted_duration,
        distraction_reason,
      },
      active_alert,
      overlay: overlay_points,
    };
  }

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

  // --- Draw Clean Visual Overlays on Video ---
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

  function updateDashboardUI(data) {
    latestTelemetry = data;

    document.getElementById("hudBlinkCount").textContent = data.blinks.total;
    document.getElementById("hudSpineScore").textContent = `${data.posture.spine_score}%`;

    // 1. Posture Card
    document.getElementById("cardSpineScore").textContent = `${data.posture.spine_score}%`;
    const cardPostureStatus = document.getElementById("cardPostureStatus");
    const cardPostureSub = document.getElementById("cardPostureSub");
    const cardBoxPosture = document.getElementById("cardBoxPosture");
    if (data.posture.is_slouching) {
      cardPostureStatus.textContent = "Slouching — Sit Up!";
      cardPostureStatus.className = "text-lg font-extrabold text-rose-300";
      cardPostureSub.textContent = data.posture.reasons[0] || "Straighten your back and neck";
      cardBoxPosture.className = "vibrant-glass-card card-glow-alert rounded-3xl p-4 flex items-center justify-between gap-4 transition";
    } else {
      cardPostureStatus.textContent = "Sitting Straight";
      cardPostureStatus.className = "text-lg font-extrabold text-white";
      cardPostureSub.textContent = "Shoulders balanced & aligned";
      cardBoxPosture.className = "vibrant-glass-card card-glow-emerald rounded-3xl p-4 flex items-center justify-between gap-4 transition";
    }

    // 2. Eyes & Blink Card
    document.getElementById("cardEarValue").textContent = data.ear.avg.toFixed(2);
    document.getElementById("cardTotalBlinks").textContent = data.blinks.total;
    document.getElementById("cardBlinkRate").textContent = `${data.blinks.per_minute} / min`;
    const cardEyeStatus = document.getElementById("cardEyeStatus");
    if (data.ear.closed_seconds >= 1.0) {
      cardEyeStatus.textContent = "Eyes Closing / Drowsy!";
      cardEyeStatus.className = "text-lg font-extrabold text-rose-300";
    } else if (data.mar.value > data.mar.threshold) {
      cardEyeStatus.textContent = "Yawning Detected";
      cardEyeStatus.className = "text-lg font-extrabold text-amber-300";
    } else {
      cardEyeStatus.textContent = "Awake & Focused";
      cardEyeStatus.className = "text-lg font-extrabold text-white";
    }

    // 3. Mobile Phone Card
    const cardPhoneState = document.getElementById("cardPhoneState");
    const cardPhoneReason = document.getElementById("cardPhoneReason");
    const cardPhoneBadge = document.getElementById("cardPhoneBadge");
    const cardBoxPhone = document.getElementById("cardBoxPhone");
    const iconBoxPhone = document.getElementById("iconBoxPhone");

    if (data.distraction.phone_detected) {
      cardPhoneState.textContent = "Mobile Phone Spotted!";
      cardPhoneState.className = "text-lg font-extrabold text-rose-200";
      cardPhoneReason.textContent = `Phone detected (${Math.round(data.distraction.phone_confidence)}% confidence) — Put it away!`;
      cardPhoneBadge.textContent = "Alert!";
      cardPhoneBadge.className = "px-3 py-1 rounded-full text-xs font-extrabold bg-rose-500/30 text-rose-200 border border-rose-400/50";
      cardBoxPhone.className = "vibrant-glass-card card-glow-alert rounded-3xl p-4 flex items-center justify-between gap-4 transition";
      iconBoxPhone.textContent = "📱";
    } else {
      cardPhoneState.textContent = "No Phone Detected";
      cardPhoneState.className = "text-lg font-extrabold text-white";
      cardPhoneReason.textContent = "Hands ignored • AI watches for real phones";
      cardPhoneBadge.textContent = "Clear";
      cardPhoneBadge.className = "px-3 py-1 rounded-full text-xs font-extrabold bg-emerald-500/25 text-emerald-200 border border-emerald-400/40";
      cardBoxPhone.className = "vibrant-glass-card card-glow-purple rounded-3xl p-4 flex items-center justify-between gap-4 transition";
      iconBoxPhone.textContent = "📵";
    }

    drawVideoOverlay(data);
    updatePipCanvas(data);

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

  // --- Hybrid WebSocket + Client MediaPipe Fallback ---
  let ws = null;
  let wsConnected = false;
  let frameInFlight = false;

  async function detectBackendMode() {
    try {
      const res = await fetch("/api/health");
      const info = await res.json();
      if (info && info.python_vision_available === false) {
        useClientMediaPipe = true;
        initClientMediaPipeModels();
        return;
      }
    } catch (_) {}
    connectWebSocket();
  }

  function connectWebSocket() {
    if (useClientMediaPipe) return;
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
        if (data && data.use_client_mediapipe) {
          useClientMediaPipe = true;
          initClientMediaPipeModels();
          return;
        }
        if (data && data.ear) {
          updateDashboardUI(data);
        }
      } catch (_) {}
    };

    ws.onclose = () => {
      wsConnected = false;
      frameInFlight = false;
      if (!useClientMediaPipe) {
        setTimeout(connectWebSocket, 2000);
      }
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
    const shouldCalibrate = forceCalibrateNext;
    forceCalibrateNext = false;

    // Mode A: Client-Side MediaPipe Tasks (Used automatically on Vercel Serverless!)
    if (useClientMediaPipe) {
      if (!clientModelsReady) {
        initClientMediaPipeModels();
        return;
      }
      frameInFlight = true;
      try {
        const telemetry = analyzeFrameClientSide(captureCanvas, shouldCalibrate);
        updateDashboardUI(telemetry);
      } catch (e) {
        console.warn("Client MediaPipe frame error:", e);
      } finally {
        frameInFlight = false;
      }
      return;
    }

    // Mode B: Local Python MediaPipe Backend via WebSocket / HTTP
    const dataUri = captureCanvas.toDataURL("image/jpeg", 0.65);

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
        if (res.status === 503) {
          useClientMediaPipe = true;
          initClientMediaPipeModels();
        } else if (res.ok) {
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
        badge.className = "text-xs font-bold px-3 py-1 rounded-full bg-indigo-500/30 text-indigo-200 border border-indigo-400/40";
      }
    } else {
      if (badge) {
        badge.textContent = "🛡️ Study Guard: Active";
        badge.className = "text-xs font-bold px-3 py-1 rounded-full bg-gradient-to-r from-indigo-500/25 to-purple-500/25 text-indigo-200 border border-indigo-400/30";
      }
      document.title = "SpineGuard AI — Posture, Drowsiness & Phone Guard | © Naresh";
    }
  });

  async function initWebcam() {
    const promptOverlay = document.getElementById("cameraPromptOverlay");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false
      });
      webcamVideo.srcObject = stream;
      await webcamVideo.play();
      promptOverlay?.classList.add("hidden");
      document.getElementById("liveStatusText").textContent = "You're doing great! Keep sitting straight 👍";
      if (pipCanvas.captureStream) {
        pipVideo.srcObject = pipCanvas.captureStream(15);
        pipVideo.play().catch(() => {});
      }
    } catch (_) {
      promptOverlay?.classList.remove("hidden");
      document.getElementById("liveStatusText").textContent = "Click 'Start Live Camera Now' below to enable webcam!";
    }
  }

  document.getElementById("btnRetryCamera")?.addEventListener("click", () => {
    initWebcam();
  });

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
    tabBtnSignup.className = "py-2.5 rounded-xl text-xs font-extrabold bg-gradient-to-r from-fuchsia-500 to-indigo-500 text-white transition";
    tabBtnLogin.className = "py-2.5 rounded-xl text-xs font-extrabold text-slate-300 hover:text-white transition";
  });

  tabBtnLogin?.addEventListener("click", () => {
    formLogin.classList.remove("hidden");
    formSignup.classList.add("hidden");
    tabBtnLogin.className = "py-2.5 rounded-xl text-xs font-extrabold bg-gradient-to-r from-emerald-400 to-cyan-500 text-slate-950 transition";
    tabBtnSignup.className = "py-2.5 rounded-xl text-xs font-extrabold text-slate-300 hover:text-white transition";
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
      fb.className = "text-xs text-slate-300 mt-1";
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
        fb.className = data.unique ? "text-xs text-emerald-300 mt-1 font-bold" : "text-xs text-rose-300 mt-1 font-bold";
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
  detectBackendMode();
  startUnthrottledWorkerTimer();
})();
