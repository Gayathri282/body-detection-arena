# Motion Arena — 2D Camera Body Detection Game

## Run
Use a local web server because browser camera permissions require a secure context.

Examples:
- VS Code Live Server
- `python -m http.server 8000`

Then open:
`http://localhost:8000`

Camera access must be granted.

## Detection architecture
The application separates:
1. POSITION — LEFT / CENTER / RIGHT
2. ACTION — IDLE / JUMPING / SQUATTING / RUNNING

The final displayed state is the combination of the two.

### Position
Calculated from the torso/hip center, rather than face position.

### Squatting
Uses knee flexion and hip displacement. A squat requires both knees to bend and the hips to lower relative to the torso while the feet remain visible.

### Jumping
Uses a temporal signature: upward hip movement + sufficient vertical speed + feet becoming elevated relative to the calibrated baseline. This prevents ordinary standing/squatting from being called a jump.

### Running
Uses repeated leg/knee movement over a time window plus torso/hip movement. A single arm movement or a single frame cannot trigger running.

### Idle
The fallback state when none of the action signatures has enough evidence.

## Important
This is a browser prototype, not a clinical-grade pose classifier. Lighting, camera angle, clothing, occlusion, frame rate, and distance affect detection accuracy. For production gameplay, thresholds should be tuned against recorded examples of each action.
