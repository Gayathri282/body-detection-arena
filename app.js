const appEl = document.getElementById("app");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const startBtn = document.getElementById("startBtn");
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

let stream = null, pose = null, camera = null, calibration = null, history = [], lastTime = performance.now(), readyFrames = 0;
const FPS_WINDOW = 18;
let current = null;

function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }
function avg(a,b){ return {x:(a.x+b.x)/2, y:(a.y+b.y)/2, z:(a.z+b.z)/2, visibility:Math.min(a.visibility||0, b.visibility||0)}; }
function visible(p){ return p && (p.visibility ?? 1) > 0.40; }

function analyze(landmarks){
  current = { poseLandmarks: landmarks };
  const upperBody = [11, 12, 23, 24];
  if(!upperBody.every(i => visible(landmarks[i]))) return { ok: false };

  const shoulder = avg(landmarks[11], landmarks[12]);
  const hip = avg(landmarks[23], landmarks[24]);
  const knee = (landmarks[25] && landmarks[26]) ? avg(landmarks[25], landmarks[26]) : hip;
  const ankle = (landmarks[27] && landmarks[28]) ? avg(landmarks[27], landmarks[28]) : knee;
  const foot = (landmarks[31] && landmarks[32] && visible(landmarks[31]) && visible(landmarks[32])) ? avg(landmarks[31], landmarks[32]) : ankle;
  
  const bodyH = Math.max(.2, dist(shoulder, foot));

  const kneeAngle = (a, b, c) => {
    if(!a || !b || !c) return 180;
    const ab = {x: a.x - b.x, y: a.y - b.y}, cb = {x: c.x - b.x, y: c.y - b.y};
    const dot = ab.x * cb.x + ab.y * cb.y, den = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
    return Math.acos(Math.max(-1, Math.min(1, dot / Math.max(1e-6, den)))) * 180 / Math.PI;
  };

  const leftKnee = (landmarks[23] && landmarks[25] && landmarks[27]) ? kneeAngle(landmarks[23], landmarks[25], landmarks[27]) : 180;
  const rightKnee = (landmarks[24] && landmarks[26] && landmarks[28]) ? kneeAngle(landmarks[24], landmarks[26], landmarks[28]) : 180;
  const kneeA = (leftKnee + rightKnee) / 2;

  const leftAnkleY = landmarks[27] ? landmarks[27].y : ankle.y;
  const rightAnkleY = landmarks[28] ? landmarks[28].y : ankle.y;
  const ankleYDiff = Math.abs(leftAnkleY - rightAnkleY);

  const hipY = hip.y, shoulderY = shoulder.y, footY = foot.y;

  // Mirror adjustment for selfie view camera (transform: scaleX(-1))
  const rawX = (hip.x + shoulder.x) / 2;
  const screenX = 1 - rawX;
  let pos = screenX < .38 ? "LEFT" : screenX > .62 ? "RIGHT" : "CENTER";

  const now = performance.now();
  lastTime = now;
  history.push({
    t: now,
    hipY,
    shoulderY,
    footY,
    rawX,
    screenX,
    kneeA,
    leftKnee,
    rightKnee,
    leftAnkleY,
    rightAnkleY,
    ankleYDiff
  });
  if(history.length > FPS_WINDOW) history.shift();

  const baseFootY = calibration?.baselineFootY ?? history[0].footY;
  const first = history[0];
  const dt = Math.max(.05, (now - first.t) / 1000);
  const hipDy = first.hipY - hipY; // Positive when body moves UP
  const footDy = first.footY - footY; // Positive when feet move UP
  const hipSpeed = Math.abs(hipDy) / dt;

  const maxAnkleDiff = Math.max(...history.map(v => v.ankleYDiff));
  const maxKneeDiff = Math.max(...history.map(v => Math.abs(v.leftKnee - v.rightKnee)));

  let act = "IDLE";

  // 1. SQUATTING: Both knees bent significantly AND feet on/near ground
  const isSquatting = (leftKnee < 140 && rightKnee < 140) || (kneeA < 135 && hipY > shoulderY + bodyH * .18);

  // 2. JUMPING: Feet or body elevated UP off ground baseline (requires feet to actually move UP)
  const feetElevated = (baseFootY - footY) > 0.035 || footDy > 0.03;
  const isJumping = feetElevated && (hipDy > 0.02 || kneeA > 110);

  // 3. RUNNING: Alternating leg motion in place (knee/ankle asymmetry over time)
  const isRunning = (maxKneeDiff > 28 || maxAnkleDiff > 0.055) && hipSpeed > 0.035 && !isJumping && !isSquatting;

  if(isJumping) {
    act = "JUMPING";
  } else if(isSquatting) {
    act = "SQUATTING";
  } else if(isRunning) {
    act = "RUNNING";
  } else {
    act = "IDLE";
  }

  return { ok: true, pos, act, x: rawX, screenX, footY, bodyH, kneeA };
}

function calibrateFrame(result){
  if(!result.ok){
    readyFrames = Math.max(0, readyFrames - 1);
    if(badgeEl) { badgeEl.textContent = "CAMERA LIVE — STEP BACK"; badgeEl.classList.remove("ready"); }
    setupMessage.textContent = "Step back until upper body and feet are visible.";
    return;
  }

  const footY = result.footY;
  const full = result.bodyH > .28 && footY < .99;
  if(full) readyFrames++; else readyFrames = Math.max(0, readyFrames - 1);

  if(readyFrames >= 8){
    calibration = { baselineFootY: footY, centerX: result.screenX };
    if(badgeEl) { badgeEl.textContent = "BODY LOCKED ✓ READY"; badgeEl.classList.add("ready"); }
    setupMessage.textContent = "Full body detected! Click START GAME to enter the arena.";
  } else {
    if(badgeEl) { badgeEl.textContent = "DETECTING BODY..."; badgeEl.classList.remove("ready"); }
    setupMessage.textContent = "Hold still inside the play frame.";
  }
}

function onResults(results){
  overlay.width = video.videoWidth || 640;
  overlay.height = video.videoHeight || 480;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if(!results.poseLandmarks){
    statusEl.textContent = "NO BODY";
    if(setup.classList.contains("active") && badgeEl){
      badgeEl.textContent = "NO BODY DETECTED";
      badgeEl.classList.remove("ready");
    }
    return;
  }

  const r = analyze(results.poseLandmarks);
  
  if(setup.classList.contains("active")){
    calibrateFrame(r);
    drawSkeleton(results.poseLandmarks);
    return;
  }

  if(!r.ok){
    statusEl.textContent = "BODY LOST";
    confidenceEl.textContent = "DETECTION: STEP INTO CAMERA VIEW";
    return;
  }

  const inZone = r.screenX > .08 && r.screenX < .92 && r.footY < .99;
  warning.classList.toggle("hidden", inZone);
  
  if(!inZone){
    confidenceEl.textContent = "DETECTION: MOVE INTO PLAY AREA";
    return;
  }

  actionEl.textContent = r.act;
  positionEl.textContent = r.pos;
  confidenceEl.textContent = `DETECTION: LOCKED  •  ${r.act} (${r.pos})`;
  statusEl.textContent = "TRACKING";
  drawSkeleton(results.poseLandmarks);
}

function drawSkeleton(ls){
  const pairs = [
    [11,12],[11,13],[13,15],[12,14],[14,16],
    [11,23],[12,24],[23,24],[23,25],[25,27],
    [24,26],[26,28],[27,31],[28,32]
  ];
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#38bdf8";
  ctx.fillStyle = "#38bdf8";

  for(const [a,b] of pairs){
    if(!ls[a] || !ls[b] || !visible(ls[a]) || !visible(ls[b])) continue;
    ctx.beginPath();
    ctx.moveTo(ls[a].x * overlay.width, ls[a].y * overlay.height);
    ctx.lineTo(ls[b].x * overlay.width, ls[b].y * overlay.height);
    ctx.stroke();
  }

  for(let i of [11,12,23,24,25,26,27,28,31,32]){
    if(ls[i] && visible(ls[i])){
      ctx.beginPath();
      ctx.arc(ls[i].x * overlay.width, ls[i].y * overlay.height, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

async function startCamera(){
  try{
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
    setupMessage.textContent = "Camera active! Step back to align your body.";
    if(badgeEl) badgeEl.textContent = "CAMERA LIVE";
    startBtn.disabled = false;

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
  } catch(e) {
    statusEl.textContent = "CAMERA ERROR";
    setupMessage.textContent = "Camera access blocked. Please allow camera permissions in browser header and reload.";
    if(badgeEl) badgeEl.textContent = "PERMISSION DENIED";
    console.error("Camera error:", e);
  }
}

startBtn.onclick = () => {
  appEl.className = "game-mode";
  setup.classList.remove("active");
  game.classList.add("active");
  gameVideoHolder.appendChild(cameraContainer);
  actionEl.textContent = "IDLE";
  positionEl.textContent = "CENTER";
  history = [];
  statusEl.textContent = "GAME READY";
};

recalibrate.onclick = () => {
  appEl.className = "setup-mode";
  game.classList.remove("active");
  setup.classList.add("active");
  setupVideoHolder.appendChild(cameraContainer);
  readyFrames = 0;
  history = [];
  setupMessage.textContent = "Step into the play area and wait for calibration.";
  statusEl.textContent = "CALIBRATING";
};

startCamera();