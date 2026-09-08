const appEl = document.getElementById("app");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const setupOverlay = document.getElementById("setupOverlay");
const setupBadge = document.getElementById("setupBadge");
const setupMessage = document.getElementById("setupMessage");
const actionEl = document.getElementById("action");
const positionEl = document.getElementById("position");
const confidenceEl = document.getElementById("confidence");
const warning = document.getElementById("boundaryWarning");
const statusEl = document.getElementById("status");
const debugPanel = document.getElementById("debugPanel");

// Debug Panel Elements
const dbgState = document.getElementById("dbgState");
const dbgConf = document.getElementById("dbgConf");
const dbgScale = document.getElementById("dbgScale");
const dbgKnees = document.getElementById("dbgKnees");
const dbgHipVel = document.getElementById("dbgHipVel");
const dbgGait = document.getElementById("dbgGait");
const dbgQuality = document.getElementById("dbgQuality");

let stream = null, pose = null, camera = null, calibration = null;
let current = null;

// Hands-free Auto-Start & Vision Loss Grace Period Parameters
let stableFullBodyFrames = 0;
const REQUIRED_STABLE_FRAMES = 36; // ~1.2s - 1.5s stability for auto-start
let gameActive = false;

// LOST-VISION GRACE PERIOD (Section 4): Protect jumps & fast motion from instant pause
let gracePeriodCounter = 0;
const MAX_GRACE_FRAMES = 25; // ~800ms grace period before declaring lost vision
let lastValidAnalysis = null;
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

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function avg(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2, visibility: Math.min(a.visibility || 0, b.visibility || 0) }; }
function visible(p) { return p && (p.visibility ?? 1) > 0.40; }

function kneeAngle(a, b, c) {
  if (!a || !b || !c) return 180;
  const ab = { x: a.x - b.x, y: a.y - b.y }, cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y, den = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  return Math.acos(Math.max(-1, Math.min(1, dot / Math.max(1e-6, den)))) * 180 / Math.PI;
}

// Main Motion & Action Classifier using Normalized Body Geometry
function analyze(landmarks) {
  current = { poseLandmarks: landmarks };

  // 1. Landmark Quality Check (Section 9)
  const keyJoints = [11, 12, 23, 24, 25, 26, 27, 28];
  let visSum = 0;
  keyJoints.forEach(i => visSum += (landmarks[i]?.visibility ?? 1));
  const visScore = visSum / keyJoints.length;

  const upperBodyOk = visible(landmarks[11]) && visible(landmarks[12]) && visible(landmarks[23]) && visible(landmarks[24]);
  if (!upperBodyOk || visScore < 0.35) {
    return { ok: false, visScore };
  }

  // Key Joint Centers
  const shoulder = avg(landmarks[11], landmarks[12]);
  const hip = avg(landmarks[23], landmarks[24]);
  const knee = (landmarks[25] && landmarks[26]) ? avg(landmarks[25], landmarks[26]) : hip;
  const ankle = (landmarks[27] && landmarks[28]) ? avg(landmarks[27], landmarks[28]) : knee;
  const foot = (landmarks[31] && landmarks[32] && visible(landmarks[31]) && visible(landmarks[32])) ? avg(landmarks[31], landmarks[32]) : ankle;

  // NORMALIZED BODY GEOMETRY (Section 5): Normalize using full-body scale
  const bodyHeight = Math.max(0.20, dist(shoulder, foot));
  const bodyScalePct = Math.round(bodyHeight * 100);

  const leftKnee = (landmarks[23] && landmarks[25] && landmarks[27]) ? kneeAngle(landmarks[23], landmarks[25], landmarks[27]) : 180;
  const rightKnee = (landmarks[24] && landmarks[26] && landmarks[28]) ? kneeAngle(landmarks[24], landmarks[26], landmarks[28]) : 180;
  const kneeAvg = (leftKnee + rightKnee) / 2;

  const leftAnkleY = landmarks[27] ? landmarks[27].y : ankle.y;
  const rightAnkleY = landmarks[28] ? landmarks[28].y : ankle.y;

  const hipY = hip.y, shoulderY = shoulder.y, footY = foot.y;

  // POSITION DETECTION (STRICTLY INDEPENDENT - Section 13)
  const rawX = (hip.x + shoulder.x) / 2;
  const screenX = 1 - rawX; // Mirror adjustment
  const pos = screenX < .35 ? "LEFT" : screenX > .65 ? "RIGHT" : "CENTER";

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
    bodyH: bodyHeight
  };

  frameHistory.push(frameSample);
  if (frameHistory.length > HISTORY_WINDOW) frameHistory.shift();

  // --------------------------------------------------------------------------
  // TEMPORAL FEATURE EXTRACTION (NORMALIZED SCALE-INVARIANT RATIOS)
  // --------------------------------------------------------------------------
  const baseFootY = calibration?.baselineFootY ?? frameHistory[0].footY;
  const baseHipY = calibration?.baselineHipY ?? frameHistory[0].hipY;
  const firstFrame = frameHistory[0];
  const dt = Math.max(.05, (now - firstFrame.t) / 1000);

  // Normalized displacements relative to body height (Section 5)
  const normHipDy = (firstFrame.hipY - hipY) / bodyHeight; // Positive when body moves UP
  const normShoulderDy = (firstFrame.shoulderY - shoulderY) / bodyHeight;
  const normFootDy = (firstFrame.footY - footY) / bodyHeight;
  const normHipSpeed = Math.abs(normHipDy) / dt;

  // Ground / Airborne check (Strict Normalized Threshold to prevent Idle Jumping)
  const normFeetElev = (baseFootY - footY) / bodyHeight;
  const feetAirborne = normFeetElev > 0.085 && normHipDy > 0.04;

  // --- SQUATTING FEATURE (RELATIVE GEOMETRY - Section 7) ---
  const bothKneesBent = leftKnee < 145 && rightKnee < 145;
  const deepKneeBend = kneeAvg < 138;
  const normHipDescent = (hipY - baseHipY) / bodyHeight;
  const isSquatAscending = (currentState === FSMState.SQUATTING || currentState === FSMState.SQUAT_CANDIDATE) && normHipDy > 0.03 && kneeAvg < 162;

  let squatRaw = 0.0;
  if (bothKneesBent || deepKneeBend || (normHipDescent > 0.08 && kneeAvg < 150)) {
    squatRaw = Math.min(1.0, 0.4 + (145 - Math.min(leftKnee, rightKnee)) / 50);
  } else if (isSquatAscending) {
    squatRaw = 0.65; // Protect continuous squatting event during ASCENDING phase
  }

  // --- JUMPING FEATURE (COORDINATED VERTICAL EVENT - Section 8) ---
  const normUpwardCoherence = Math.min(normHipDy, normShoulderDy);
  const isJumpTrajectory = normUpwardCoherence > 0.04 && feetAirborne;

  let jumpRaw = 0.0;
  if (isSquatAscending || normFeetElev < 0.04) {
    jumpRaw = 0.0; // Idle standing & squat ascent MUST NEVER trigger jumping
  } else if (isJumpTrajectory) {
    jumpRaw = Math.min(1.0, 0.5 + normFeetElev * 8 + normUpwardCoherence * 6);
  } else if (feetAirborne && kneeAvg > 115 && normHipDy > 0.035) {
    jumpRaw = 0.6;
  }

  // --- RUNNING FEATURE (ALTERNATING GAIT ENGINE - Section 9) ---
  let kneeFlips = 0;
  for (let i = 1; i < frameHistory.length; i++) {
    const prevDiff = frameHistory[i - 1].kneeDiff;
    const currDiff = frameHistory[i].kneeDiff;
    if ((prevDiff > 8 && currDiff < -8) || (prevDiff < -8 && currDiff > 8)) {
      kneeFlips++;
    }
  }

  const maxKneeAsymmetry = Math.max(...frameHistory.map(f => Math.abs(f.kneeDiff)));
  const maxAnkleAsymmetry = Math.max(...frameHistory.map(f => Math.abs(f.ankleYDiff)));
  const normAnkleAsymmetry = maxAnkleAsymmetry / bodyHeight;

  let runScore = 0.0;
  if (kneeFlips >= 1 && maxKneeAsymmetry > 18 && !bothKneesBent) runScore += 0.55;
  if (normAnkleAsymmetry > 0.07 && maxKneeAsymmetry > 18 && !bothKneesBent) runScore += 0.35;
  if (maxKneeAsymmetry > 22 && normHipSpeed > 0.04 && !bothKneesBent) runScore += 0.25;

  // CRUCIAL DISAMBIGUATION:
  // 1. If BOTH knees are bent (Squatting), RUNNING MUST BE 0.0!
  // 2. If body is jumping or ascending from squat, RUNNING MUST BE 0.0!
  if (bothKneesBent || deepKneeBend || maxKneeAsymmetry < 14 || isJumpTrajectory || isSquatAscending) {
    runScore = 0.0;
  }

  let runRaw = Math.min(1.0, runScore);

  // --- IDLE FEATURE (NORMAL HUMAN STANDING - Section 6) ---
  const activeSum = squatRaw * 1.2 + jumpRaw * 1.2 + runRaw * 1.2;
  let idleRaw = Math.max(0.0, 1.0 - activeSum);

  // Dead-zone tolerance for breathing, small posture/arm shifts, camera jitter
  if (normHipSpeed < 0.06 && !feetAirborne && maxKneeAsymmetry < 16 && kneeAvg > 155 && !bothKneesBent) {
    idleRaw = 1.0; squatRaw = 0.0; jumpRaw = 0.0; runRaw = 0.0;
  }

  // --------------------------------------------------------------------------
  // CONFIDENCE TEMPORAL SMOOTHING (EMA - Section 6 & 11)
  // --------------------------------------------------------------------------
  const alpha = 0.22;
  idleConf  = idleConf  * (1 - alpha) + idleRaw  * alpha;
  squatConf = squatConf * (1 - alpha) + squatRaw * alpha;
  jumpConf  = jumpConf  * (1 - alpha) + jumpRaw  * alpha;
  runConf   = runConf   * (1 - alpha) + runRaw   * alpha;

  // --------------------------------------------------------------------------
  // FINITE STATE MACHINE TRANSITIONS & HYSTERESIS (Section 11 & 12)
  // --------------------------------------------------------------------------
  updateStateMachine(feetAirborne, kneeAvg);

  const act = getDisplayedAction();

  // Developer Debug Information (Section 15)
  updateDebugPanel({
    state: currentState,
    confidences: {
      idle: Math.round(idleConf * 100),
      squat: Math.round(squatConf * 100),
      jump: Math.round(jumpConf * 100),
      run: Math.round(runConf * 100)
    },
    bodyScalePct,
    leftKnee: Math.round(leftKnee),
    rightKnee: Math.round(rightKnee),
    hipSpeed: normHipSpeed.toFixed(3),
    kneeFlips,
    normAnkleAsymmetry: normAnkleAsymmetry.toFixed(2),
    grounded: feetAirborne ? "NO (AIRBORNE)" : "YES",
    visScore: Math.round(visScore * 100)
  });

  const analysisResult = { ok: true, pos, act, x: rawX, screenX, footY, hipY, bodyH: bodyHeight, bodyScalePct, visScore };
  lastValidAnalysis = analysisResult;
  return analysisResult;
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
          currentState = FSMState.JUMPING; candidateFrameCount = 0; stateHoldFrames = 10;
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

function updateDebugPanel(info) {
  if (!debugPanel) return;
  dbgState.textContent = info.state;
  dbgConf.textContent = `I:${info.confidences.idle}% S:${info.confidences.squat}% J:${info.confidences.jump}% R:${info.confidences.run}%`;
  dbgScale.textContent = `${info.bodyScalePct}%`;
  dbgKnees.textContent = `L:${info.leftKnee}° R:${info.rightKnee}°`;
  dbgHipVel.textContent = info.hipSpeed;
  dbgGait.textContent = `${info.kneeFlips} flips / ${info.normAnkleAsymmetry}`;
  dbgQuality.textContent = `${info.grounded} / ${info.visScore}%`;
}

// Full Body Calibration & Proximity Verification (Section 3 & 14)
function checkFullBodyVisibility(landmarks) {
  if (!landmarks) return { isFull: false, reason: "No body detected. Please step into view." };

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

  // 1. Proximity Check (Section 14): When Player is Too Close to Camera
  const shoulderAvgY = (leftShoulder.y + rightShoulder.y) / 2;
  const footAvgY = (leftFoot && visible(leftFoot)) ? leftFoot.y : (leftAnkle ? leftAnkle.y : 1.0);
  const bodyH = dist(avg(leftShoulder, rightShoulder), avg(leftAnkle || leftHip, rightAnkle || rightHip));

  if (bodyH > 0.84 || (shoulderAvgY < 0.06 && (!kneesOk || !feetOk))) {
    return { isFull: false, isTooClose: true, reason: "STEP BACK — Standing too close to camera." };
  }

  if (!shouldersOk || !hipsOk) {
    return { isFull: false, reason: "Torso not fully visible — step back from the camera." };
  }

  const headY = nose ? nose.y : (shoulderAvgY - 0.15);
  if (headY < 0.01) {
    return { isFull: false, reason: "Head is cut off at top — step back or tilt camera up." };
  }

  // During active gameplay, if hips and shoulders are clearly visible and player is squatting, protect from false pauses
  const isSquatActive = (currentState === FSMState.SQUATTING || currentState === FSMState.SQUAT_CANDIDATE);
  if (gameActive && shouldersOk && hipsOk && isSquatActive) {
    return { isFull: true, reason: "✓ Full body detected!" };
  }

  if (!kneesOk) {
    return { isFull: false, reason: "Knees and lower body not visible — step back." };
  }

  if (!feetOk) {
    return { isFull: false, reason: "Feet not visible — step back or tilt camera down." };
  }

  if (footAvgY > 0.98) {
    return { isFull: false, reason: "Feet cut off at bottom — step back slightly." };
  }

  const screenX = 1 - (leftHip.x + rightHip.x) / 2;
  if (screenX < 0.08 || screenX > 0.92) {
    return { isFull: false, reason: "Step towards the center of the screen." };
  }

  return { isFull: true, reason: "✓ Full body detected!" };
}

function calibrateFrame(landmarks, result) {
  const bodyCheck = checkFullBodyVisibility(landmarks);

  if (!bodyCheck.isFull) {
    stableFullBodyFrames = Math.max(0, stableFullBodyFrames - 2);
    if (setupBadge) {
      setupBadge.textContent = bodyCheck.isTooClose ? "STEP BACK" : "INCOMPLETE BODY";
      setupBadge.className = "setup-badge";
    }
    setupMessage.textContent = bodyCheck.reason;
    return;
  }

  stableFullBodyFrames++;
  const progressPct = Math.min(100, Math.round((stableFullBodyFrames / REQUIRED_STABLE_FRAMES) * 100));

  if (stableFullBodyFrames < REQUIRED_STABLE_FRAMES) {
    if (setupBadge) {
      setupBadge.textContent = `VERIFYING STABILITY ${progressPct}%`;
      setupBadge.className = "setup-badge verifying";
    }
    const remainingSec = (Math.ceil((REQUIRED_STABLE_FRAMES - stableFullBodyFrames) / 25 * 10) / 10).toFixed(1);
    setupMessage.textContent = `Full body detected! Hold still to auto-start (${remainingSec}s)...`;
  } else {
    // AUTOMATIC START TRANSITION
    if (setupBadge) {
      setupBadge.textContent = "GAME STARTING! ✓";
      setupBadge.className = "setup-badge ready";
    }
    setupMessage.textContent = "Full body locked! Auto-starting game...";
    
    calibration = { baselineFootY: result.footY, baselineHipY: result.hipY, centerX: result.screenX };
    gameActive = true;

    // Auto-transition
    setupOverlay.classList.remove("active");
    actionEl.textContent = "IDLE";
    positionEl.textContent = "CENTER";
    currentState = FSMState.IDLE;
    idleConf = 1.0; squatConf = 0.0; jumpConf = 0.0; runConf = 0.0;
    statusEl.textContent = "GAME ACTIVE";
  }
}

function onResults(results) {
  overlay.width = video.videoWidth || window.innerWidth;
  overlay.height = video.videoHeight || window.innerHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!results.poseLandmarks) {
    statusEl.textContent = "NO BODY";
    if (!gameActive) {
      if (setupBadge) {
        setupBadge.textContent = "NO BODY DETECTED";
        setupBadge.className = "setup-badge";
      }
      setupMessage.textContent = "No body detected. Step in front of the camera.";
    } else {
      // LOST-VISION GRACE PERIOD FOR GAMEPLAY
      handleLandmarkLoss("NO BODY DETECTED");
    }
    return;
  }

  const r = analyze(results.poseLandmarks);

  if (!gameActive) {
    calibrateFrame(results.poseLandmarks, r);
    drawSkeleton(results.poseLandmarks);
    return;
  }

  // GAME ACTIVE MONITORING & LOST-VISION GRACE PERIOD (Section 4)
  const bodyCheck = checkFullBodyVisibility(results.poseLandmarks);

  if (!r.ok || !bodyCheck.isFull) {
    if (bodyCheck.isTooClose) {
      warning.classList.remove("hidden");
      warning.classList.add("step-back");
      warning.textContent = "STEP BACK — STANDING TOO CLOSE";
      confidenceEl.textContent = "DETECTION: STEP BACK FOR FULL VIEW";
      actionEl.textContent = "STEP BACK";
      drawSkeleton(results.poseLandmarks);
      return;
    }

    // Short Vision Loss Grace Period (Section 4): Protect jumps & occlusion
    handleLandmarkLoss(bodyCheck.reason);
    drawSkeleton(results.poseLandmarks);
    return;
  }

  // Vision is clear! Reset grace period & present active tracking
  gracePeriodCounter = 0;
  warning.classList.add("hidden");
  warning.classList.remove("step-back");
  
  actionEl.textContent = r.act;
  positionEl.textContent = r.pos;
  confidenceEl.textContent = `DETECTION: LOCKED  •  ${r.act} (${r.pos})`;
  statusEl.textContent = "TRACKING ACTIVE";

  drawSkeleton(results.poseLandmarks);
}

function handleLandmarkLoss(reason) {
  // LOST-VISION GRACE PERIOD (Section 4):
  // Preserve previous stable state during short vision loss (e.g. jumping high)
  if (gracePeriodCounter < MAX_GRACE_FRAMES && lastValidAnalysis) {
    gracePeriodCounter++;
    actionEl.textContent = lastValidAnalysis.act;
    positionEl.textContent = lastValidAnalysis.pos;
    confidenceEl.textContent = `DETECTION: TRACKING (GRACE PERIOD ${Math.round((MAX_GRACE_FRAMES - gracePeriodCounter)/25*10)/10}s)`;
    statusEl.textContent = "JUMP / MOTION BUFFER";
  } else {
    // Grace period expired -> declare out of bounds / paused
    warning.classList.remove("hidden");
    warning.classList.remove("step-back");
    warning.textContent = "MOVE INTO THE PLAY AREA";
    confidenceEl.textContent = `DETECTION: PAUSED — ${reason}`;
    actionEl.textContent = "PAUSED";
    statusEl.textContent = "PAUSED";
  }
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

    statusEl.textContent = "CAMERA ACTIVE";
    setupMessage.textContent = "Camera active! Step into the play area and show your full body.";
    if (setupBadge) setupBadge.textContent = "CAMERA LIVE";

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
    if (setupBadge) setupBadge.textContent = "PERMISSION DENIED";
    console.error("Camera error:", e);
  }
}

startCamera();