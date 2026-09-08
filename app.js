const appEl = document.getElementById("app");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const setupMessage = document.getElementById("setupMessage");
const setup = document.getElementById("setup");
const game = document.getElementById("game");
const actionEl = document.getElementById("action");
const positionEl = document.getElementById("position");
const confidenceEl = document.getElementById("confidence");
const warning = document.getElementById("boundaryWarning");
const statusEl = document.getElementById("status");
const recalibrate = document.getElementById("recalibrate");
const cameraContainer = document.getElementById("cameraContainer");
const setupVideoHolder = document.getElementById("setupVideoHolder");
const gameVideoHolder = document.getElementById("gameVideoHolder");
const badgeEl = document.getElementById("setupOverlayBadge");

let stream = null, pose = null, camera = null, calibration = null;
let current = null;

// Automatic hands-free start and pause/resume counters
let stableFullBodyFrames = 0;
const REQUIRED_STABLE_FRAMES = 36; // ~1.2s to 1.5s of stable full-body detection
let inGameResumeFrames = 0;

// ============================================================================
// TEMPORAL STATE MACHINE & CONFIDENCE ENGINE
// ============================================================================

const FSMState = {
  IDLE: 'IDLE',
  SQUAT_CANDIDATE: 'SQUAT_CANDIDATE',
  SQUATTING: 'SQUATTING',
  JUMP_CANDIDATE: 'JUMP_CANDIDATE',
  JUMPING: 'JUMPING',
  RUN_CANDIDATE: 'RUN_CANDIDATE',
  RUNNING: 'RUNNING'
};

let currentState = FSMState.IDLE;
let candidateFrameCount = 0;
let stateHoldFrames = 0;
let runDropFrames = 0;

// Smoothed Action Confidences
let idleConf = 1.0;
let squatConf = 0.0;
let jumpConf = 0.0;
let runConf = 0.0;

// Rolling Frame History (24 frames max ~ 1 sec)
const HISTORY_WINDOW = 24;
const frameHistory = [];
let lowQualityFrames = 0;

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function avg(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2, visibility: Math.min(a.visibility || 0, b.visibility || 0) }; }
function visible(p) { return p && (p.visibility ?? 1) > 0.40; }

function kneeAngle(a, b, c) {
  if (!a || !b || !c) return 180;
  const ab = { x: a.x - b.x, y: a.y - b.y }, cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y, den = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  return Math.acos(Math.max(-1, Math.min(1, dot / Math.max(1e-6, den)))) * 180 / Math.PI;
}

// Main Motion & Action Classifier
function analyze(landmarks) {
  current = { poseLandmarks: landmarks };

  // 1. Landmark Quality Check (Section 9)
  const keyJoints = [11, 12, 23, 24, 25, 26, 27, 28];
  let visSum = 0;
  keyJoints.forEach(i => visSum += (landmarks[i]?.visibility ?? 1));
  const visScore = visSum / keyJoints.length;

  const upperBodyOk = visible(landmarks[11]) && visible(landmarks[12]) && visible(landmarks[23]) && visible(landmarks[24]);
  if (!upperBodyOk || visScore < 0.35) {
    lowQualityFrames++;
    if (lowQualityFrames < 10 && frameHistory.length > 0) {
      const prev = frameHistory[frameHistory.length - 1];
      return { ok: true, pos: prev.pos, act: getDisplayedAction(), screenX: prev.screenX, footY: prev.footY, bodyH: prev.bodyH };
    }
    return { ok: false };
  }
  lowQualityFrames = 0;

  // Key Coordinates
  const shoulder = avg(landmarks[11], landmarks[12]);
  const hip = avg(landmarks[23], landmarks[24]);
  const knee = (landmarks[25] && landmarks[26]) ? avg(landmarks[25], landmarks[26]) : hip;
  const ankle = (landmarks[27] && landmarks[28]) ? avg(landmarks[27], landmarks[28]) : knee;
  const foot = (landmarks[31] && landmarks[32] && visible(landmarks[31]) && visible(landmarks[32])) ? avg(landmarks[31], landmarks[32]) : ankle;

  const bodyH = Math.max(.2, dist(shoulder, foot));

  const leftKnee = (landmarks[23] && landmarks[25] && landmarks[27]) ? kneeAngle(landmarks[23], landmarks[25], landmarks[27]) : 180;
  const rightKnee = (landmarks[24] && landmarks[26] && landmarks[28]) ? kneeAngle(landmarks[24], landmarks[26], landmarks[28]) : 180;
  const kneeAvg = (leftKnee + rightKnee) / 2;

  const leftAnkleY = landmarks[27] ? landmarks[27].y : ankle.y;
  const rightAnkleY = landmarks[28] ? landmarks[28].y : ankle.y;

  const hipY = hip.y, shoulderY = shoulder.y, footY = foot.y;

  // POSITION DETECTION (STRICTLY INDEPENDENT)
  const rawX = (hip.x + shoulder.x) / 2;
  const screenX = 1 - rawX; // Mirror adjustment
  const pos = screenX < .38 ? "LEFT" : screenX > .62 ? "RIGHT" : "CENTER";

  const now = performance.now();
  const frameSample = {
    t: now,
    hipY,
    shoulderY,
    footY,
    rawX,
    screenX,
    pos,
    kneeAvg,
    leftKnee,
    rightKnee,
    kneeDiff: leftKnee - rightKnee,
    leftAnkleY,
    rightAnkleY,
    ankleYDiff: leftAnkleY - rightAnkleY,
    bodyH
  };

  frameHistory.push(frameSample);
  if (frameHistory.length > HISTORY_WINDOW) frameHistory.shift();

  // --------------------------------------------------------------------------
  // TEMPORAL FEATURE EXTRACTION
  // --------------------------------------------------------------------------
  const baseFootY = calibration?.baselineFootY ?? frameHistory[0].footY;
  const firstFrame = frameHistory[0];
  const dt = Math.max(.05, (now - firstFrame.t) / 1000);

  const hipDy = firstFrame.hipY - hipY; // Positive when body moves UP
  const shoulderDy = firstFrame.shoulderY - shoulderY;
  const footDy = firstFrame.footY - footY;
  const hipSpeed = Math.abs(hipDy) / dt;

  // Ground / Airborne check
  const feetAirborne = (baseFootY - footY) > 0.032 || footDy > 0.028;

  // --- SQUATTING FEATURE ---
  const bothKneesBent = leftKnee < 142 && rightKnee < 142;
  const deepKneeBend = kneeAvg < 135;
  const hipLowered = hipY > shoulderY + bodyH * .17;
  const isSquatAscending = (currentState === FSMState.SQUATTING || currentState === FSMState.SQUAT_CANDIDATE) && hipDy > 0.015 && kneeAvg < 162;

  let squatRaw = 0.0;
  if (bothKneesBent || deepKneeBend || (hipLowered && kneeAvg < 148)) {
    squatRaw = Math.min(1.0, 0.4 + (145 - Math.min(leftKnee, rightKnee)) / 50);
  } else if (isSquatAscending) {
    squatRaw = 0.65; // Preserve continuous squat during ASCENDING phase
  }

  // --- JUMPING FEATURE ---
  const upwardCoherence = Math.min(hipDy, shoulderDy);
  const isJumpTrajectory = upwardCoherence > 0.025 && feetAirborne;

  let jumpRaw = 0.0;
  if (isSquatAscending) {
    jumpRaw = 0.0;
  } else if (isJumpTrajectory) {
    jumpRaw = Math.min(1.0, 0.5 + (baseFootY - footY) * 10 + upwardCoherence * 5);
  } else if (feetAirborne && kneeAvg > 115 && hipDy > 0.015) {
    jumpRaw = 0.6;
  }

  // --- RUNNING FEATURE ---
  let kneeFlips = 0;
  for (let i = 1; i < frameHistory.length; i++) {
    const prevDiff = frameHistory[i - 1].kneeDiff;
    const currDiff = frameHistory[i].kneeDiff;
    if ((prevDiff > 5 && currDiff < -5) || (prevDiff < -5 && currDiff > 5)) {
      kneeFlips++;
    }
  }

  const maxKneeAsymmetry = Math.max(...frameHistory.map(f => Math.abs(f.kneeDiff)));
  const maxAnkleAsymmetry = Math.max(...frameHistory.map(f => Math.abs(f.ankleYDiff)));

  let runScore = 0.0;
  if (kneeFlips >= 2 && maxKneeAsymmetry > 22) runScore += 0.55;
  if (maxAnkleAsymmetry > 0.045) runScore += 0.30;
  if (maxKneeAsymmetry > 26 && hipSpeed > 0.03) runScore += 0.25;

  if (maxKneeAsymmetry < 16 || isJumpTrajectory || isSquatAscending) {
    runScore = 0.0;
  }

  let runRaw = Math.min(1.0, runScore);

  // --- IDLE FEATURE ---
  const activeSum = squatRaw * 1.2 + jumpRaw * 1.2 + runRaw * 1.2;
  let idleRaw = Math.max(0.0, 1.0 - activeSum);

  if (hipSpeed < 0.035 && !feetAirborne && maxKneeAsymmetry < 18 && kneeAvg > 155 && !bothKneesBent) {
    idleRaw = 1.0; squatRaw = 0.0; jumpRaw = 0.0; runRaw = 0.0;
  }

  // --------------------------------------------------------------------------
  // CONFIDENCE TEMPORAL SMOOTHING (EMA)
  // --------------------------------------------------------------------------
  const alpha = 0.22;
  idleConf  = idleConf  * (1 - alpha) + idleRaw  * alpha;
  squatConf = squatConf * (1 - alpha) + squatRaw * alpha;
  jumpConf  = jumpConf  * (1 - alpha) + jumpRaw  * alpha;
  runConf   = runConf   * (1 - alpha) + runRaw   * alpha;

  // --------------------------------------------------------------------------
  // FINITE STATE MACHINE TRANSITIONS & HYSTERESIS
  // --------------------------------------------------------------------------
  updateStateMachine(feetAirborne, kneeAvg);

  const act = getDisplayedAction();

  window.debugMotionState = {
    state: currentState,
    confidences: {
      idle: parseFloat(idleConf.toFixed(2)),
      squat: parseFloat(squatConf.toFixed(2)),
      jump: parseFloat(jumpConf.toFixed(2)),
      run: parseFloat(runConf.toFixed(2))
    },
    kneeAvg: Math.round(kneeAvg),
    leftKnee: Math.round(leftKnee),
    rightKnee: Math.round(rightKnee),
    hipSpeed: parseFloat(hipSpeed.toFixed(3)),
    ankleAsymmetry: parseFloat(maxAnkleAsymmetry.toFixed(3)),
    kneeFlips,
    feetAirborne,
    visScore: parseFloat(visScore.toFixed(2))
  };

  return { ok: true, pos, act, x: rawX, screenX, footY, bodyH, kneeA: kneeAvg };
}

function updateStateMachine(feetAirborne, kneeAvg) {
  switch (currentState) {
    case FSMState.IDLE:
      if (squatConf > 0.48) { currentState = FSMState.SQUAT_CANDIDATE; candidateFrameCount = 1; }
      else if (jumpConf > 0.52 && feetAirborne) { currentState = FSMState.JUMP_CANDIDATE; candidateFrameCount = 1; }
      else if (runConf > 0.48) { currentState = FSMState.RUN_CANDIDATE; candidateFrameCount = 1; }
      break;

    case FSMState.SQUAT_CANDIDATE:
      if (squatConf > 0.42) {
        candidateFrameCount++;
        if (candidateFrameCount >= 2) { currentState = FSMState.SQUATTING; candidateFrameCount = 0; }
      } else { currentState = FSMState.IDLE; candidateFrameCount = 0; }
      break;

    case FSMState.SQUATTING:
      if (squatConf > 0.25 || kneeAvg < 155) { /* Remain SQUATTING */ }
      else { currentState = FSMState.IDLE; }
      break;

    case FSMState.JUMP_CANDIDATE:
      if (jumpConf > 0.45) {
        candidateFrameCount++;
        if (candidateFrameCount >= 2) {
          currentState = FSMState.JUMPING; candidateFrameCount = 0; stateHoldFrames = 8;
        }
      } else { currentState = FSMState.IDLE; candidateFrameCount = 0; }
      break;

    case FSMState.JUMPING:
      stateHoldFrames--;
      if (stateHoldFrames > 0 || jumpConf > 0.28 || feetAirborne) { /* Hold JUMPING */ }
      else { currentState = (runConf > 0.50) ? FSMState.RUNNING : FSMState.IDLE; }
      break;

    case FSMState.RUN_CANDIDATE:
      if (runConf > 0.45) {
        candidateFrameCount++;
        if (candidateFrameCount >= 4) { currentState = FSMState.RUNNING; candidateFrameCount = 0; runDropFrames = 0; }
      } else { currentState = FSMState.IDLE; candidateFrameCount = 0; }
      break;

    case FSMState.RUNNING:
      if (runConf > 0.25) { runDropFrames = 0; }
      else {
        runDropFrames++;
        if (runDropFrames >= 6) {
          if (squatConf > 0.50) currentState = FSMState.SQUATTING;
          else if (jumpConf > 0.55 && feetAirborne) currentState = FSMState.JUMPING;
          else currentState = FSMState.IDLE;
          runDropFrames = 0;
        }
      }
      break;

    default:
      currentState = FSMState.IDLE;
  }
}

function getDisplayedAction() {
  if (currentState === FSMState.SQUATTING || currentState === FSMState.SQUAT_CANDIDATE) return "SQUATTING";
  if (currentState === FSMState.JUMPING || currentState === FSMState.JUMP_CANDIDATE) return "JUMPING";
  if (currentState === FSMState.RUNNING || currentState === FSMState.RUN_CANDIDATE) return "RUNNING";
  return "IDLE";
}

// Full Body Calibration Verification
function checkFullBodyVisibility(landmarks) {
  if (!landmarks) return { isFull: false, reason: "No body detected. Please stand in front of the camera." };

  const nose = landmarks[0];
  const leftShoulder = landmarks[11], rightShoulder = landmarks[12];
  const leftHip = landmarks[23], rightHip = landmarks[24];
  const leftKnee = landmarks[25], rightKnee = landmarks[26];
  const leftAnkle = landmarks[27], rightAnkle = landmarks[28];
  const leftFoot = landmarks[31], rightFoot = landmarks[32];

  const shouldersOk = visible(leftShoulder) && visible(rightShoulder);
  const hipsOk = visible(leftHip) && visible(rightHip);
  const kneesOk = visible(leftKnee) && visible(rightKnee);
  const anklesOk = visible(leftAnkle) && visible(rightAnkle);
  const feetOk = anklesOk || (leftFoot && visible(leftFoot)) || (rightFoot && visible(rightFoot));

  if (!shouldersOk || !hipsOk) {
    return { isFull: false, reason: "Torso not fully visible — step back from the camera." };
  }

  const headY = nose ? nose.y : (leftShoulder.y - 0.15);
  if (headY < 0.02) {
    return { isFull: false, reason: "Head is cut off at top — step back or tilt camera up." };
  }

  if (!kneesOk) {
    return { isFull: false, reason: "Knees and lower body not visible — step back." };
  }

  if (!feetOk) {
    return { isFull: false, reason: "Feet not visible — step back or tilt camera down." };
  }

  const footY = (leftFoot && visible(leftFoot)) ? leftFoot.y : (leftAnkle ? leftAnkle.y : 1.0);
  if (footY > 0.96) {
    return { isFull: false, reason: "Feet cut off at bottom — step back slightly." };
  }

  const screenX = 1 - (leftHip.x + rightHip.x) / 2;
  if (screenX < 0.12 || screenX > 0.88) {
    return { isFull: false, reason: "Step towards the center of the frame." };
  }

  return { isFull: true, reason: "✓ Full body detected!" };
}

function calibrateFrame(landmarks, result) {
  const bodyCheck = checkFullBodyVisibility(landmarks);

  if (!bodyCheck.isFull) {
    stableFullBodyFrames = Math.max(0, stableFullBodyFrames - 2);
    if (badgeEl) {
      badgeEl.textContent = "INCOMPLETE BODY";
      badgeEl.className = "overlay-badge";
    }
    setupMessage.textContent = bodyCheck.reason;
    return;
  }

  stableFullBodyFrames++;
  const progressPct = Math.min(100, Math.round((stableFullBodyFrames / REQUIRED_STABLE_FRAMES) * 100));

  if (stableFullBodyFrames < REQUIRED_STABLE_FRAMES) {
    if (badgeEl) {
      badgeEl.textContent = `VERIFYING STABILITY ${progressPct}%`;
      badgeEl.className = "overlay-badge verifying";
    }
    const remainingSec = (Math.ceil((REQUIRED_STABLE_FRAMES - stableFullBodyFrames) / 25 * 10) / 10).toFixed(1);
    setupMessage.textContent = `Full body detected! Hold still to auto-start game (${remainingSec}s)...`;
  } else {
    // AUTOMATIC TRANSITION TO GAME ACTIVE - NO BUTTON CLICK REQUIRED!
    if (badgeEl) {
      badgeEl.textContent = "STARTING GAME! ✓";
      badgeEl.className = "overlay-badge ready";
    }
    setupMessage.textContent = "Full body locked! Auto-starting game...";
    
    calibration = { baselineFootY: result.footY, centerX: result.screenX };

    // Auto-transition to gameplay
    appEl.className = "game-mode";
    setup.classList.remove("active");
    game.classList.add("active");
    gameVideoHolder.appendChild(cameraContainer);
    actionEl.textContent = "IDLE";
    positionEl.textContent = "CENTER";
    currentState = FSMState.IDLE;
    idleConf = 1.0; squatConf = 0.0; jumpConf = 0.0; runConf = 0.0;
    statusEl.textContent = "GAME ACTIVE";
  }
}

function onResults(results) {
  overlay.width = video.videoWidth || 640;
  overlay.height = video.videoHeight || 480;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!results.poseLandmarks) {
    statusEl.textContent = "NO BODY";
    if (setup.classList.contains("active")) {
      if (badgeEl) {
        badgeEl.textContent = "NO BODY DETECTED";
        badgeEl.className = "overlay-badge";
      }
      setupMessage.textContent = "No body detected. Step in front of the camera.";
    } else if (game.classList.contains("active")) {
      warning.classList.remove("hidden");
      warning.textContent = "MOVE INTO THE PLAY AREA";
      confidenceEl.textContent = "DETECTION: PAUSED — NO BODY DETECTED";
      actionEl.textContent = "PAUSED";
      inGameResumeFrames = 0;
    }
    return;
  }

  const r = analyze(results.poseLandmarks);

  if (setup.classList.contains("active")) {
    calibrateFrame(results.poseLandmarks, r);
    drawSkeleton(results.poseLandmarks);
    return;
  }

  // IN-GAME PLAY AREA & OUT-OF-BOUNDS AUTO PAUSE / AUTO RESUME
  const bodyCheck = checkFullBodyVisibility(results.poseLandmarks);
  const inZone = r.screenX > .08 && r.screenX < .92 && r.footY < .99;

  if (!bodyCheck.isFull || !inZone || !r.ok) {
    inGameResumeFrames = 0;
    warning.classList.remove("hidden");
    warning.textContent = "MOVE INTO THE PLAY AREA";
    confidenceEl.textContent = "DETECTION: PAUSED — MOVE INTO PLAY AREA";
    actionEl.textContent = "PAUSED";
    positionEl.textContent = r.pos || "CENTER";
    statusEl.textContent = "PAUSED";
    drawSkeleton(results.poseLandmarks);
    return;
  }

  // Player returned fully into play area
  inGameResumeFrames++;
  if (inGameResumeFrames >= 8) { // ~0.3s stability to resume tracking cleanly
    warning.classList.add("hidden");
    actionEl.textContent = r.act;
    positionEl.textContent = r.pos;
    confidenceEl.textContent = `DETECTION: LOCKED  •  ${r.act} (${r.pos})`;
    statusEl.textContent = "TRACKING ACTIVE";
  } else {
    actionEl.textContent = "RESUMING...";
    confidenceEl.textContent = "DETECTION: VERIFYING POSITION...";
  }

  drawSkeleton(results.poseLandmarks);
}

function drawSkeleton(ls) {
  const pairs = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
    [24, 26], [26, 28], [27, 31], [28, 32]
  ];
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#38bdf8";
  ctx.fillStyle = "#38bdf8";

  for (const [a, b] of pairs) {
    if (!ls[a] || !ls[b] || !visible(ls[a]) || !visible(ls[b])) continue;
    ctx.beginPath();
    ctx.moveTo(ls[a].x * overlay.width, ls[a].y * overlay.height);
    ctx.lineTo(ls[b].x * overlay.width, ls[b].y * overlay.height);
    ctx.stroke();
  }

  for (let i of [11, 12, 23, 24, 25, 26, 27, 28, 31, 32]) {
    if (ls[i] && visible(ls[i])) {
      ctx.beginPath();
      ctx.arc(ls[i].x * overlay.width, ls[i].y * overlay.height, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

async function startCamera() {
  try {
    statusEl.textContent = "CONNECTING CAMERA...";
    setupMessage.textContent = "Requesting camera permissions...";

    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    setupVideoHolder.appendChild(cameraContainer);
    statusEl.textContent = "CAMERA ACTIVE";
    setupMessage.textContent = "Camera active! Step into the play area and show your full body.";
    if (badgeEl) badgeEl.textContent = "CAMERA LIVE";

    pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`
    });
    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    pose.onResults(onResults);

    if (typeof Camera !== 'undefined') {
      camera = new Camera(video, {
        onFrame: async () => await pose.send({ image: video }),
        width: 1280,
        height: 720
      });
      camera.start();
    } else {
      const frameLoop = async () => {
        if (video.readyState >= 2 && !video.paused && !video.ended) {
          await pose.send({ image: video });
        }
        requestAnimationFrame(frameLoop);
      };
      frameLoop();
    }
  } catch (e) {
    statusEl.textContent = "CAMERA ERROR";
    setupMessage.textContent = "Camera access blocked. Please allow camera permissions in browser header and reload.";
    if (badgeEl) badgeEl.textContent = "PERMISSION DENIED";
    console.error("Camera error:", e);
  }
}

recalibrate.onclick = () => {
  appEl.className = "setup-mode";
  game.classList.remove("active");
  setup.classList.add("active");
  setupVideoHolder.appendChild(cameraContainer);
  stableFullBodyFrames = 0;
  inGameResumeFrames = 0;
  currentState = FSMState.IDLE;
  idleConf = 1.0; squatConf = 0.0; jumpConf = 0.0; runConf = 0.0;
  setupMessage.textContent = "Step into the play area and show your full body.";
  statusEl.textContent = "CALIBRATING";
};

startCamera();