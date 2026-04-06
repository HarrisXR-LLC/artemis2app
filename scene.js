/**
 * scene.js — Three.js Artemis II Realistic Mode (Phase 1)
 *
 * Renders Earth moving through a 10-day mission-aligned grid.
 * Scale: 1 unit = 1,000 km
 *
 * Camera strategy:
 *   - Camera is PURE TOP-DOWN (straight down the Z axis).
 *   - This makes pan/zoom math trivial: screen X = world X, screen Y = world Y.
 *   - Visual tilt (~10°) is achieved by rotating the scene contents group,
 *     not the camera. This gives the 3D feel without breaking the math.
 *
 * Exports an API object that index.html calls to drive the scene.
 */

import * as THREE from 'three';
import OEM_POSITIONS, { OEM_WITH_MET, getOrionPositionAtMET, getOrionFullPosition, OEM_MET_START, OEM_MET_END, MISSION_END_MET, getMoonEphemerisAtMET } from './trajectory.js';

/* ============================================================
   SHARED SCENE CONSTANTS
   ============================================================ */

const EARTH_DIAMETER = 12.756;    // 12,756 km
const EARTH_RADIUS = EARTH_DIAMETER / 2;

const DAILY_GRID_SIZE = 2600;     // 2,600,000 km per day
const HOUR_SUBDIVISION = 24;
const MISSION_DAYS = 10;
const GRID_TOTAL = DAILY_GRID_SIZE * MISSION_DAYS; // 26,000 u

const MOON_DIAMETER = 3.475;
const MOON_RADIUS = MOON_DIAMETER / 2;
const MOON_ORBIT_RADIUS = 384.4;  // average Earth-Moon distance in scene units

const VISUAL_TILT_DEG = 10;       // scene group tilt for 3D feel
const EDGE_BUFFER = 0.10;         // 10% buffer around field edges
const CAM_DIST = 50000;           // camera Z distance (arbitrary for ortho)

/* ============================================================
   EARTH MOTION MODEL — real circular arc
   ============================================================
   Earth orbits the Sun at R ≈ 149,598,000 km = 149,598 scene units.
   Over 10 days it sweeps θ ≈ 9.86° (0.1721 rad) of that circle.
   The arc chord ≈ 25,731u, sagitta ≈ 553u.

   Grid mapping:
     - The chord direction maps to the grid Y axis (top to bottom).
     - The sagitta (outward bulge away from Sun) maps to grid -X.
     - Earth orbits counterclockwise (ecliptic north view), so the
       Sun (orbit center) is 90° right of the velocity vector.

   We compute positions on the real circle, then rotate/translate
   so the chord aligns with the grid Y axis, centered in the grid.
   ============================================================ */

const MISSION_DURATION_SEC = MISSION_DAYS * 24 * 3600;

// Orbital constants
const ORBIT_RADIUS = 149598;  // 1 AU in scene units
const EARTH_ANGULAR_SPEED = (2 * Math.PI) / (365.25 * 24 * 3600); // rad/s
const ARC_ANGLE = EARTH_ANGULAR_SPEED * MISSION_DURATION_SEC; // ~0.1721 rad (~9.86°)
const HALF_ARC = ARC_ANGLE / 2;

// Derived geometry (for debug HUD)
const CHORD_LENGTH = 2 * ORBIT_RADIUS * Math.sin(HALF_ARC); // ~25,731u
const SAGITTA = ORBIT_RADIUS * (1 - Math.cos(HALF_ARC));     // ~553u

// Sun direction: center of the orbit circle relative to the arc midpoint.
// On the grid, the chord runs along Y (top to bottom). The Sun (orbit center)
// is to the +X side (right), perpendicular to the chord at the midpoint.
// Distance from chord midpoint to center = R * cos(θ/2) ≈ 149,596u (off-screen).
const SUN_OFFSET_X = ORBIT_RADIUS * Math.cos(HALF_ARC); // distance from midpoint to Sun along X

function getEarthPosition(missionFraction) {
  const f = Math.max(0, Math.min(1, missionFraction));

  // Angle along the arc: 0 at start, ARC_ANGLE at end
  // We parameterize from -HALF_ARC to +HALF_ARC so the midpoint is at angle 0
  const angle = -HALF_ARC + f * ARC_ANGLE;

  // Position on the circle (Sun at origin, orbit in XY plane):
  //   circleX = R * sin(angle)   → lateral position along chord direction
  //   circleY = R * cos(angle)   → perpendicular to chord (toward/away from Sun)
  //
  // At angle=0 (midpoint): circleX=0, circleY=R (closest to chord midpoint on Sun side)
  // We want to map this so:
  //   - circleX → grid Y (top to bottom, so negate: positive angle = further along = lower Y)
  //   - The perpendicular displacement → grid X (bow toward Sun = +X)

  // Grid Y: chord runs top (+GRID_TOTAL/2) to bottom (-GRID_TOTAL/2)
  // Map circleX from [-R*sin(HALF_ARC), +R*sin(HALF_ARC)] to [+GRID_TOTAL/2, -GRID_TOTAL/2]
  const chordPos = ORBIT_RADIUS * Math.sin(angle); // ranges from -CHORD/2 to +CHORD/2
  const gridY = -chordPos * (GRID_TOTAL / CHORD_LENGTH);

  // Grid X: perpendicular bow away from Sun.
  // The arc bulges AWAY from the orbit center (Sun is at +X, bulge goes -X).
  // perpDisp is positive at midpoint (arc is closer to Sun than chord endpoints).
  // Negate it so the bulge goes -X (away from Sun), matching real orbital geometry.
  // perpDisp = R*cos(angle) - R*cos(HALF_ARC), 0 at endpoints, SAGITTA at midpoint.
  const perpDisp = ORBIT_RADIUS * Math.cos(angle) - ORBIT_RADIUS * Math.cos(HALF_ARC);
  const gridX = -perpDisp; // bulge toward -X (away from Sun)

  return new THREE.Vector3(gridX, gridY, 0);
}

// Sample the full orbit arc for drawing the path line
function sampleOrbitArc(steps) {
  const points = [];
  for (let i = 0; i <= steps; i++) {
    const f = i / steps;
    points.push(getEarthPosition(f));
  }
  return points;
}

/* ============================================================
   SCENE STATE
   ============================================================ */

let renderer, scene, camera;
let starScene, starCamera; // separate scene/camera for fixed starfield background
let container;
let contentGroup;  // holds grid + Earth; rotated for visual tilt
let earthMesh;
let orionMarker;  // + marker for Orion position
let moonMesh;     // Moon sphere
let moonOrbitLine; // Moon orbit trace
let animFrameId = null;
let debugCanvas, debugCtx; // 2D overlay for debug info

// Camera: pure top-down, target is a world XY point
let cameraTarget = new THREE.Vector3(0, 0, 0);
let zoomLevel = 1;
let targetZoomLevel = 1; // smooth zoom target — zoomLevel eases toward this
const ZOOM_LERP_SPEED = 8; // exponential ease-out speed (higher = faster response)

// Camera animation (startup zoom-to-Earth)
let camLerp = null; // { startTarget, endTarget, startZoom, endZoom, elapsed, duration }

// Camera tracking: 'orion' | 'moon' | 'earth' | 'lunar-orbit' | null
let trackTarget = null;
let trackPanOffset = new THREE.Vector2(0, 0); // user pan offset relative to tracked target
let debugVisible = false; // debug HUD toggle

// Pan interaction
let isPanning = false, panStartX = 0, panStartY = 0;
let panStartTargetX = 0, panStartTargetY = 0;

// Mission time
let currentMET = 0;

// Earth lerp
let earthTargetPos = new THREE.Vector3(0, 0, 0);
const EARTH_LERP_SPEED = 30; // very tight lerp — almost instant

/* ============================================================
   STARFIELD (fixed background, does not move with camera)
   ============================================================ */

function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function createStarfield() {
  const STAR_COUNT = 400;
  const rng = mulberry32(42);

  starScene = new THREE.Scene();
  // Orthographic camera that never changes — fills the screen
  starCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2);
  starCamera.position.z = 1;

  const positions = [];
  const sizes = [];
  const alphas = [];

  for (let i = 0; i < STAR_COUNT; i++) {
    // Normalized -1 to 1 (fills screen)
    positions.push(
      -1 + rng() * 2,
      -1 + rng() * 2,
      0
    );
    sizes.push(0.5 + rng() * 1.0);    // 0.5-1.5 pixel equivalent
    alphas.push(0.2 + rng() * 0.5);   // 0.2-0.7 brightness
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('aSize', new THREE.Float32BufferAttribute(sizes, 1));
  geometry.setAttribute('aAlpha', new THREE.Float32BufferAttribute(alphas, 1));

  // Simple point material — white dots
  const material = new THREE.PointsMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.6,
    size: 0.004, // in NDC units for ortho camera (-1 to 1)
    sizeAttenuation: false
  });

  const points = new THREE.Points(geometry, material);
  starScene.add(points);
}

/* ============================================================
   INIT
   ============================================================ */

function init(containerEl) {
  container = containerEl;

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x0a0a0f, 1);
  renderer.autoClear = false; // we render starfield + main scene manually
  updateRendererSize();
  container.appendChild(renderer.domElement);

  // Scene
  scene = new THREE.Scene();

  // Starfield background (separate scene, rendered first)
  createStarfield();

  // Content group: everything in the scene goes here, tilted for visual effect
  contentGroup = new THREE.Group();
  contentGroup.rotation.x = -THREE.MathUtils.degToRad(VISUAL_TILT_DEG);
  scene.add(contentGroup);

  // Camera: pure top-down
  setupCamera();

  // Grid
  createGrid();

  // Earth
  createEarth();

  // Orbit arc line + Sun direction indicator
  createOrbitArc();
  createSunIndicator();

  // Artemis II trajectory (OEM backbone, child of Earth)
  createTrajectoryLine();

  // Orion marker
  createOrionMarker();

  // LIVE indicator (above Orion)
  createLiveIndicator();

  // Moon (uses JPL Horizons ephemeris for real 3D position)
  createMoon();
  createMoonOrbitTrace();

  // Axis labels at grid edges
  createAxisLabels();

  // Lighting (added to contentGroup so it tilts with the scene)
  contentGroup.add(new THREE.AmbientLight(0xffffff, 0.4));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(1, 1, 1).normalize();
  contentGroup.add(dirLight);

  // Debug HUD overlay (2D canvas on top of Three.js)
  debugCanvas = document.createElement('canvas');
  debugCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10;display:none;';
  container.appendChild(debugCanvas);
  debugCtx = debugCanvas.getContext('2d');

  // Controls
  setupControls();

  // Resize observer
  const ro = new ResizeObserver(() => {
    updateRendererSize();
    updateCameraFrustum();
  });
  ro.observe(container);

  // Snap Earth to its real position for the first frame
  updateEarthPosition();
  earthMesh.position.copy(earthTargetPos);

  // Start locked on Orion at zoom 45 — no fly-in animation
  zoomLevel = 360;
  targetZoomLevel = 360;
  setTrackTarget('orion');
  updateOrionPosition();
  // Position camera at Orion's world position immediately
  if (orionMarker && orionMarker.visible) {
    const wp = new THREE.Vector3();
    orionMarker.getWorldPosition(wp);
    moveCamera(wp.x, wp.y);
  } else {
    moveCamera(earthTargetPos.x, earthTargetPos.y);
  }
  updateCameraFrustum();

  // Start render loop
  startLoop();

  console.log('scene.js: init complete — locked on Orion');
}

function updateRendererSize() {
  const rect = container.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height);
}

/* ============================================================
   CAMERA — pure top-down orthographic
   ============================================================ */

function setupCamera() {
  const rect = container.getBoundingClientRect();
  const aspect = rect.width / rect.height;
  const viewHeight = GRID_TOTAL * (1 + EDGE_BUFFER);
  const viewWidth = viewHeight * aspect;

  camera = new THREE.OrthographicCamera(
    -viewWidth / 2, viewWidth / 2,
    viewHeight / 2, -viewHeight / 2,
    1, CAM_DIST * 2
  );

  // Straight down: camera at (x, y, dist) looking at (x, y, 0)
  camera.position.set(cameraTarget.x, cameraTarget.y, CAM_DIST);
  camera.lookAt(cameraTarget.x, cameraTarget.y, 0);
  camera.updateProjectionMatrix();
}

// Camera vertical offset: shift view up by 25% to account for info panel at bottom
const CAMERA_Y_OFFSET_FRACTION = 0.25;

function updateCameraFrustum() {
  if (!camera) return;
  const rect = container.getBoundingClientRect();
  const aspect = rect.width / rect.height;
  const viewHeight = (GRID_TOTAL * (1 + EDGE_BUFFER)) / zoomLevel;
  const viewWidth = viewHeight * aspect;

  // Shift the frustum down so the focal point appears in the upper 75% of the screen
  const offsetY = viewHeight * CAMERA_Y_OFFSET_FRACTION;
  camera.left = -viewWidth / 2;
  camera.right = viewWidth / 2;
  camera.top = viewHeight / 2 - offsetY;
  camera.bottom = -viewHeight / 2 - offsetY;
  camera.updateProjectionMatrix();
}

function moveCamera(worldX, worldY) {
  cameraTarget.set(worldX, worldY, 0);
  camera.position.set(worldX, worldY, CAM_DIST);
  camera.lookAt(worldX, worldY, 0);
}

function clampCameraTarget() {
  const maxPan = GRID_TOTAL / 2;
  cameraTarget.x = Math.max(-maxPan, Math.min(maxPan, cameraTarget.x));
  cameraTarget.y = Math.max(-maxPan, Math.min(maxPan, cameraTarget.y));
  camera.position.set(cameraTarget.x, cameraTarget.y, CAM_DIST);
  camera.lookAt(cameraTarget.x, cameraTarget.y, 0);
}

/* ============================================================
   CAMERA TRACKING
   ============================================================ */

const LUNAR_ORBIT_ZOOM = (GRID_TOTAL * (1 + EDGE_BUFFER)) / (MOON_ORBIT_RADIUS * 3);
// Zoom level that fits the Moon orbit with padding

function setTrackTarget(target) {
  trackTarget = target;
  trackPanOffset.set(0, 0); // reset pan offset when switching targets
  if (target === 'lunar-orbit') {
    zoomLevel = LUNAR_ORBIT_ZOOM;
    updateCameraFrustum();
  }
}

function updateTracking() {
  if (!trackTarget) return;

  const worldPos = new THREE.Vector3();

  if (trackTarget === 'orion' && orionMarker && orionMarker.visible) {
    orionMarker.getWorldPosition(worldPos);
  } else if (trackTarget === 'moon' && moonMesh) {
    moonMesh.getWorldPosition(worldPos);
  } else if (trackTarget === 'earth' && earthMesh) {
    earthMesh.getWorldPosition(worldPos);
  } else if (trackTarget === 'lunar-orbit' && earthMesh) {
    earthMesh.getWorldPosition(worldPos);
  } else {
    return;
  }

  // Apply user pan offset relative to tracked target
  moveCamera(worldPos.x + trackPanOffset.x, worldPos.y + trackPanOffset.y);
  updateCameraFrustum();
}

function setDebugVisible(visible) {
  debugVisible = visible;
  if (debugCanvas) debugCanvas.style.display = visible ? 'block' : 'none';
}

/* ============================================================
   GRID
   ============================================================ */

function createGrid() {
  const gridGroup = new THREE.Group();
  const halfGrid = GRID_TOTAL / 2;

  const majorMat = new THREE.LineBasicMaterial({
    color: 0x2266cc, transparent: true, opacity: 0.25
  });
  const minorMat = new THREE.LineBasicMaterial({
    color: 0x888888, transparent: true, opacity: 0.06
  });

  for (let day = 0; day <= MISSION_DAYS; day++) {
    const pos = -halfGrid + day * DAILY_GRID_SIZE;
    addLine(gridGroup, pos, -halfGrid, pos, halfGrid, majorMat);
    addLine(gridGroup, -halfGrid, pos, halfGrid, pos, majorMat);

    if (day < MISSION_DAYS) {
      for (let h = 1; h < HOUR_SUBDIVISION; h++) {
        const subPos = pos + (h / HOUR_SUBDIVISION) * DAILY_GRID_SIZE;
        addLine(gridGroup, subPos, -halfGrid, subPos, halfGrid, minorMat);
        addLine(gridGroup, -halfGrid, subPos, halfGrid, subPos, minorMat);
      }
    }
  }

  gridGroup.position.z = -5; // below all other scene elements
  contentGroup.add(gridGroup);
}

function addLine(group, x1, y1, x2, y2, material) {
  const geo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(x1, y1, 0),
    new THREE.Vector3(x2, y2, 0)
  ]);
  group.add(new THREE.Line(geo, material));
}

/* ============================================================
   EARTH
   ============================================================ */

function createEarth() {
  const geo = new THREE.SphereGeometry(EARTH_RADIUS, 32, 32);
  const mat = new THREE.MeshPhongMaterial({
    color: 0x2196f3,
    emissive: 0x0d3b6e,
    emissiveIntensity: 0.3,
    shininess: 30
  });
  earthMesh = new THREE.Mesh(geo, mat);

  // Earth SOI ring (925u radius — Hill sphere relative to Sun)
  const soiGeo = new THREE.RingGeometry(924, 926, 128);
  const soiMat = new THREE.MeshBasicMaterial({ color: 0x4488ff, side: THREE.DoubleSide, transparent: true, opacity: 0.3 });
  earthMesh.add(new THREE.Mesh(soiGeo, soiMat));

  contentGroup.add(earthMesh);
}

function updateEarthPosition() {
  const missionFrac = currentMET / MISSION_DURATION_SEC;
  earthTargetPos = getEarthPosition(missionFrac);
}

function lerpEarth(dt) {
  const t = 1 - Math.exp(-EARTH_LERP_SPEED * dt);
  earthMesh.position.lerp(earthTargetPos, t);
}

/* ============================================================
   ORBIT ARC LINE + SUN INDICATOR
   ============================================================ */

function createOrbitArc() {
  // Sample the arc and draw it as a dashed line
  const points = sampleOrbitArc(200);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineDashedMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.25,
    dashSize: 54.17,
    gapSize: 54.17
  });
  const line = new THREE.Line(geometry, material);
  line.computeLineDistances(); // required for dashed lines
  contentGroup.add(line);
}

function createSunIndicator() {
  // The Sun is to the +X side of the arc midpoint, at distance SUN_OFFSET_X (~149,596u).
  // We can't show the Sun itself (way off grid), so we draw an arrow at the grid edge
  // pointing in the +X direction with a "☀ Sun" label.

  // Arrow line from grid edge toward +X
  const midY = 0; // arc midpoint is at grid center vertically
  const arrowStart = GRID_TOTAL / 2 + 200; // just outside the grid right edge
  const arrowEnd = arrowStart + 800;

  const arrowGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(arrowStart, midY, 0),
    new THREE.Vector3(arrowEnd, midY, 0)
  ]);
  const arrowMat = new THREE.LineBasicMaterial({
    color: 0xffcc00, transparent: true, opacity: 0.5
  });
  const arrowLine = new THREE.Line(arrowGeo, arrowMat);
  contentGroup.add(arrowLine);

  // Arrowhead
  const headSize = 200;
  const headGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(arrowEnd, midY, 0),
    new THREE.Vector3(arrowEnd - headSize, midY + headSize * 0.4, 0),
    new THREE.Vector3(arrowEnd - headSize, midY - headSize * 0.4, 0),
    new THREE.Vector3(arrowEnd, midY, 0)
  ]);
  const headLine = new THREE.Line(headGeo, arrowMat);
  contentGroup.add(headLine);

  // "SUN →" label: we'll draw this in the debug HUD instead (2D text is cleaner)
}

/* ============================================================
   ARTEMIS II TRAJECTORY (Earth-centered OEM, child of earthMesh)
   ============================================================
   OEM data is Earth-centered (EME2000), so making the trajectory
   a child of earthMesh means it automatically moves with Earth
   through the grid. P_orion_world = P_earth + R_orion_from_earth.

   Color-coded by stage and time position:
   - Past stages: solid dark azure
   - Current stage behind Orion: solid lime green
   - Current stage ahead of Orion: dashed lime green
   - Future stages: dashed cornflower
   ============================================================ */

// --- Archived original for reference ---
function archived_createTrajectoryLine() {
  const points = OEM_POSITIONS.map(p => new THREE.Vector3(p[1], p[2], p[3]));
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0x00ffaa, transparent: true, opacity: 0.8 });
  const line = new THREE.Line(geometry, material);
  earthMesh.add(line);
  const startGeo = new THREE.SphereGeometry(1, 8, 8);
  const startMat = new THREE.MeshBasicMaterial({ color: 0x00ffaa });
  const startMarker = new THREE.Mesh(startGeo, startMat);
  startMarker.position.copy(points[0]);
  earthMesh.add(startMarker);
  const endGeo = new THREE.SphereGeometry(1, 8, 8);
  const endMat = new THREE.MeshBasicMaterial({ color: 0xff4444 });
  const endMarker = new THREE.Mesh(endGeo, endMat);
  endMarker.position.copy(points[points.length - 1]);
  earthMesh.add(endMarker);
}

// --- Trajectory color constants ---
const TRAJ_COLOR_PAST = 0x4477cc;      // dark azure — completed stages
const TRAJ_COLOR_CURRENT = 0x39ff14;   // lime green — current stage
const TRAJ_COLOR_FUTURE = 0x6699ee;    // cornflower — future stages

// --- Materials (reused across segments) ---
const matPastSolid = new THREE.LineBasicMaterial({ color: TRAJ_COLOR_PAST, transparent: true, opacity: 0.8, linewidth: 6 });
const matCurrentSolid = new THREE.LineBasicMaterial({ color: TRAJ_COLOR_CURRENT, transparent: true, opacity: 0.9, linewidth: 6 });
const matCurrentDashed = new THREE.LineDashedMaterial({ color: TRAJ_COLOR_CURRENT, transparent: true, opacity: 0.6, dashSize: 0.75, gapSize: 0.5, linewidth: 6 });
const matFutureDashed = new THREE.LineDashedMaterial({ color: TRAJ_COLOR_FUTURE, transparent: true, opacity: 0.5, dashSize: 0.75, gapSize: 0.5, linewidth: 6 });

// Per-stage trajectory segments
let stageSegments = []; // { metStart, metEnd, points[], line, stageId }
// Current-stage split lines (dynamic geometry)
let currentBehindLine = null; // solid lime — path already traveled in current stage
let currentAheadLine = null;  // dashed lime — path remaining in current stage
let lastTrajectoryUpdateMET = -1; // throttle updates to ~1/sec
let currentTrajectoryStageIdx = 0; // which stage index is "current" for trajectory coloring

/**
 * Get trajectory points between two MET values.
 * Samples from getOrionFullPosition at OEM data density.
 */
function getTrajectoryPointsBetweenMET(metStart, metEnd) {
  const points = [];

  // Add start point
  const startPos = getOrionFullPosition(metStart);
  if (startPos) points.push(new THREE.Vector3(startPos.x, startPos.y, startPos.z));

  // Add all OEM points that fall within this MET range
  for (const oem of OEM_WITH_MET) {
    if (oem.met > metStart && oem.met < metEnd) {
      points.push(new THREE.Vector3(oem.x, oem.y, oem.z));
    }
  }

  // Add end point
  const endPos = getOrionFullPosition(metEnd);
  if (endPos) points.push(new THREE.Vector3(endPos.x, endPos.y, endPos.z));

  return points;
}

function createTrajectoryLine() {
  // This will be called from init. The actual per-stage segments are created
  // in createStageTrajectorySegments() which is called after stages are passed in.

  // Create the two dynamic split lines for the current stage
  // (geometry will be set dynamically in updateTrajectoryColors)
  const emptyGeo = new THREE.BufferGeometry();

  currentBehindLine = new THREE.Line(emptyGeo.clone(), matCurrentSolid);
  currentBehindLine.frustumCulled = false;
  earthMesh.add(currentBehindLine);

  currentAheadLine = new THREE.Line(emptyGeo.clone(), matCurrentDashed);
  currentAheadLine.frustumCulled = false;
  earthMesh.add(currentAheadLine);
}

function createStageTrajectorySegments(stages) {
  // Build one line per stage with pre-computed points
  stageSegments = stages.map(stage => {
    const points = getTrajectoryPointsBetweenMET(stage.metSeconds, stage.metEndSeconds);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    // Start with future dashed material (will be updated in updateTrajectoryColors)
    const line = new THREE.Line(geometry, matFutureDashed);
    line.computeLineDistances(); // needed for dashed materials
    line.frustumCulled = false;
    earthMesh.add(line);
    return {
      metStart: stage.metSeconds,
      metEnd: stage.metEndSeconds,
      stageId: stage.id,
      points,
      line
    };
  });
}

function updateTrajectoryColors() {
  if (!stageSegments.length) return;

  // Throttle: only update if MET changed by at least 1 second
  if (Math.abs(currentMET - lastTrajectoryUpdateMET) < 1) return;
  lastTrajectoryUpdateMET = currentMET;

  // Find which stage the current MET falls in
  let currentStageIdx = -1;
  for (let i = stageSegments.length - 1; i >= 0; i--) {
    if (currentMET >= stageSegments[i].metStart) {
      currentStageIdx = i;
      break;
    }
  }
  if (currentStageIdx === -1) currentStageIdx = 0;
  currentTrajectoryStageIdx = currentStageIdx;

  // Update each stage segment's material
  for (let i = 0; i < stageSegments.length; i++) {
    const seg = stageSegments[i];
    if (i < currentStageIdx) {
      // Past stage: solid dark azure
      seg.line.material = matPastSolid;
      seg.line.visible = true;
    } else if (i === currentStageIdx) {
      // Current stage: hide the full-stage line, use split lines instead
      seg.line.visible = false;
    } else {
      // Future stage: dashed cornflower
      seg.line.material = matFutureDashed;
      seg.line.visible = true;
    }
  }

  // Update current-stage split lines
  const currentSeg = stageSegments[currentStageIdx];
  if (!currentSeg) return;

  // Build split points from MET values
  // Behind: stage start → current MET
  const behindPts = getTrajectoryPointsBetweenMET(currentSeg.metStart, currentMET);
  // Ahead: current MET → stage end
  const aheadPts = getTrajectoryPointsBetweenMET(currentMET, currentSeg.metEnd);

  // Set materials based on live mode: lime only when live, otherwise azure/cornflower
  currentBehindLine.material = isLiveMode ? matCurrentSolid : matPastSolid;
  currentAheadLine.material = isLiveMode ? matCurrentDashed : matFutureDashed;

  // Update behind line geometry
  if (behindPts.length >= 2) {
    const geo = new THREE.BufferGeometry().setFromPoints(behindPts);
    currentBehindLine.geometry.dispose();
    currentBehindLine.geometry = geo;
    currentBehindLine.visible = true;
  } else {
    currentBehindLine.visible = false;
  }

  // Update ahead line geometry
  if (aheadPts.length >= 2) {
    const geo = new THREE.BufferGeometry().setFromPoints(aheadPts);
    geo.computeBoundingSphere(); // needed for dashed
    currentAheadLine.geometry.dispose();
    currentAheadLine.geometry = geo;
    currentAheadLine.computeLineDistances(); // required for dashed material
    currentAheadLine.visible = true;
  } else {
    currentAheadLine.visible = false;
  }

  // Update ghost circle colors to match trajectory state
  updateGhostCircleColors(currentStageIdx);

  // Update Orion glow ball color based on temporal state
  if (isLiveMode) {
    updateOrionMarkerColor(TRAJ_COLOR_CURRENT); // lime green
  } else {
    // Determine if scrubbed time is past or future relative to real live time
    // currentMET is the effective MET — compare to what live MET would be
    const realLiveMET = (Date.now() - Date.UTC(2026, 3, 1, 22, 35, 12)) / 1000;
    if (currentMET < realLiveMET) {
      updateOrionMarkerColor(TRAJ_COLOR_PAST); // dark azure
    } else {
      updateOrionMarkerColor(TRAJ_COLOR_FUTURE); // cornflower
    }
  }
}

function updateGhostCircleColors(currentStageIdx) {
  if (!ghostCircles.length || !stageSegments.length) return;

  const currentSeg = stageSegments[currentStageIdx];
  if (!currentSeg) return;

  for (const gc of ghostCircles) {
    if (gc.metSeconds < currentSeg.metStart) {
      // Past: dark azure
      gc.ringMat.color.setHex(TRAJ_COLOR_PAST);
    } else if (gc.metSeconds >= currentSeg.metStart && gc.metSeconds <= currentSeg.metEnd) {
      // Current stage: lime only when live, otherwise based on position relative to real time
      gc.ringMat.color.setHex(isLiveMode ? TRAJ_COLOR_CURRENT : TRAJ_COLOR_PAST);
    } else {
      // Future: cornflower
      gc.ringMat.color.setHex(TRAJ_COLOR_FUTURE);
    }
  }
}

/* ============================================================
   ORION MARKER (glow ball sprite at interpolated OEM position)
   ============================================================ */

const ORION_SCREEN_PX = 18; // target screen-space diameter
let orionSpriteMaterial = null; // stored for color updates
let orionGlowTextures = {}; // cache: hex color → CanvasTexture

function createOrionGlowTexture(hexColor) {
  const key = hexColor.toString(16);
  if (orionGlowTextures[key]) return orionGlowTextures[key];

  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const cx = size / 2, cy = size / 2;

  // Extract RGB from hex
  const r = (hexColor >> 16) & 0xff;
  const g = (hexColor >> 8) & 0xff;
  const b = hexColor & 0xff;
  const rgb = `${r},${g},${b}`;

  // Outer glow (soft, wide)
  const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, cx);
  glow.addColorStop(0, `rgba(${rgb},0.9)`);
  glow.addColorStop(0.25, `rgba(${rgb},0.6)`);
  glow.addColorStop(0.5, `rgba(${rgb},0.2)`);
  glow.addColorStop(1, `rgba(${rgb},0)`);
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, size, size);

  // Outer ring
  ctx.beginPath();
  ctx.arc(cx, cy, 22, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(${rgb},0.7)`;
  ctx.lineWidth = 3;
  ctx.stroke();

  // Inner solid circle
  ctx.beginPath();
  ctx.arc(cx, cy, 12, 0, Math.PI * 2);
  ctx.fillStyle = `rgba(${rgb},0.95)`;
  ctx.fill();

  const texture = new THREE.CanvasTexture(canvas);
  orionGlowTextures[key] = texture;
  return texture;
}

function createOrionMarker() {
  const texture = createOrionGlowTexture(TRAJ_COLOR_CURRENT);
  orionSpriteMaterial = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    opacity: 1.0,
    depthTest: false
  });
  orionMarker = new THREE.Sprite(orionSpriteMaterial);
  orionMarker.scale.set(6, 6, 1); // base size, will be scaled per frame
  orionMarker.visible = false;
  earthMesh.add(orionMarker);
}

function updateOrionMarkerColor(hexColor) {
  if (!orionSpriteMaterial) return;
  const texture = createOrionGlowTexture(hexColor);
  orionSpriteMaterial.map = texture;
  orionSpriteMaterial.needsUpdate = true;
}

function updateOrionMarkerScale() {
  if (!orionMarker || !orionMarker.visible || !container) return;
  const rect = container.getBoundingClientRect();
  const viewHeight = (GRID_TOTAL * (1 + EDGE_BUFFER)) / zoomLevel;
  const worldUnitsPerPixel = viewHeight / rect.height;
  const desiredSize = ORION_SCREEN_PX * worldUnitsPerPixel;
  orionMarker.scale.set(desiredSize, desiredSize, 1);
}

function updateOrionPosition() {
  const pos = getOrionFullPosition(currentMET);
  if (pos) {
    orionMarker.position.set(pos.x, pos.y, pos.z);
    orionMarker.visible = true;
  } else {
    orionMarker.visible = false;
  }
}

/* ============================================================
   ORION INDICATOR SYSTEM (triangle + text above Orion)
   ============================================================
   Three states:
   - Live: lime green solid triangle + "LIVE" text, pulsating
   - Past: dark azure solid triangle + "ORION" text
   - Future: cornflower solid triangle + "ORION" text

   Ghost marker: when scrubbed away, a dashed triangle outline
   + faint "LIVE" text remains at the real-time live position.
   ============================================================ */

let orionIndicatorGroup = null;  // follows scrubbed Orion position
let orionIndicatorTriMat = null;
let orionIndicatorSprite = null;
let orionIndicatorSpriteMat = null;

let ghostLiveGroup = null;       // stays at real-time live position
let ghostLiveTriMat = null;
let ghostLiveSpriteMat = null;

let isLiveMode = true;

// Pre-rendered text sprites (cached)
let textTextures = {};

function createTextTexture(text, color) {
  const key = text + color;
  if (textTextures[key]) return textTextures[key];
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 48;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 32px monospace';
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 64, 24);
  const texture = new THREE.CanvasTexture(canvas);
  textTextures[key] = texture;
  return texture;
}

function createTriangleShape() {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.lineTo(2, 0);
  shape.lineTo(1, -1.5);
  shape.closePath();
  return shape;
}

function createLiveIndicator() {
  // --- Main indicator (follows scrubbed Orion position) ---
  orionIndicatorGroup = new THREE.Group();
  orionIndicatorGroup.visible = true;

  // Solid triangle
  const triGeo = new THREE.ShapeGeometry(createTriangleShape());
  triGeo.translate(-1, 0, 0);
  orionIndicatorTriMat = new THREE.MeshBasicMaterial({
    color: TRAJ_COLOR_CURRENT,
    transparent: true, opacity: 0.9, side: THREE.DoubleSide
  });
  orionIndicatorGroup.add(new THREE.Mesh(triGeo, orionIndicatorTriMat));

  // Text sprite (starts as "LIVE")
  orionIndicatorSpriteMat = new THREE.SpriteMaterial({
    map: createTextTexture('LIVE', '#39ff14'),
    transparent: true, opacity: 0.9
  });
  orionIndicatorSprite = new THREE.Sprite(orionIndicatorSpriteMat);
  orionIndicatorSprite.scale.set(6, 2.25, 1);
  orionIndicatorSprite.position.set(0, 1.125, 0);
  orionIndicatorGroup.add(orionIndicatorSprite);

  earthMesh.add(orionIndicatorGroup);

  // --- Ghost LIVE marker (stays at real-time position) ---
  ghostLiveGroup = new THREE.Group();
  ghostLiveGroup.visible = false; // only shown when scrubbed away

  // Dashed triangle outline
  const triPoints = [
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, -1.5, 0),
    new THREE.Vector3(-1, 0, 0) // close the loop
  ];
  const triLineGeo = new THREE.BufferGeometry().setFromPoints(triPoints);
  ghostLiveTriMat = new THREE.LineDashedMaterial({
    color: TRAJ_COLOR_CURRENT,
    transparent: true, opacity: 0.25,
    dashSize: 0.3, gapSize: 0.2
  });
  const triLine = new THREE.Line(triLineGeo, ghostLiveTriMat);
  triLine.computeLineDistances();
  ghostLiveGroup.add(triLine);

  // Faint "LIVE" text
  ghostLiveSpriteMat = new THREE.SpriteMaterial({
    map: createTextTexture('LIVE', '#39ff14'),
    transparent: true, opacity: 0.25
  });
  const ghostSprite = new THREE.Sprite(ghostLiveSpriteMat);
  ghostSprite.scale.set(6, 2.25, 1);
  ghostSprite.position.set(0, 1.125, 0);
  ghostLiveGroup.add(ghostSprite);

  earthMesh.add(ghostLiveGroup);
}

function positionIndicatorAboveOrion(group, targetPos, targetScale) {
  if (!group || !container) return;
  const rect = container.getBoundingClientRect();
  const viewHeight = (GRID_TOTAL * (1 + EDGE_BUFFER)) / zoomLevel;
  const worldUnitsPerPixel = viewHeight / rect.height;
  const desiredWorldSize = 15 * worldUnitsPerPixel;
  const scaleFactor = desiredWorldSize / 1.5;
  group.scale.setScalar(scaleFactor);

  const topY = targetPos.y + (targetScale ? targetScale.y / 2 : 0);
  const groupY = topY + 1.5 * scaleFactor;
  group.position.set(targetPos.x, groupY, targetPos.z);
}

function updateLiveIndicator() {
  if (!orionIndicatorGroup || !container || !orionMarker) return;

  const orionVisible = orionMarker.visible;

  // --- Main indicator (always visible above current/scrubbed Orion) ---
  orionIndicatorGroup.visible = orionVisible;
  if (orionVisible) {
    positionIndicatorAboveOrion(orionIndicatorGroup, orionMarker.position, orionMarker.scale);

    if (isLiveMode) {
      // Live: lime green, "LIVE", pulsating
      orionIndicatorTriMat.color.setHex(TRAJ_COLOR_CURRENT);
      orionIndicatorSpriteMat.map = createTextTexture('LIVE', '#39ff14');
      const pulse = 0.8 + 0.2 * Math.sin(Date.now() * 0.003);
      orionIndicatorTriMat.opacity = pulse;
      orionIndicatorSpriteMat.opacity = pulse;
    } else {
      const realLiveMET = (Date.now() - Date.UTC(2026, 3, 1, 22, 35, 12)) / 1000;
      if (currentMET < realLiveMET) {
        // Past: dark azure, "ORION"
        orionIndicatorTriMat.color.setHex(TRAJ_COLOR_PAST);
        orionIndicatorSpriteMat.map = createTextTexture('ORION', '#4477cc');
      } else {
        // Future: cornflower, "ORION"
        orionIndicatorTriMat.color.setHex(TRAJ_COLOR_FUTURE);
        orionIndicatorSpriteMat.map = createTextTexture('ORION', '#6699ee');
      }
      orionIndicatorTriMat.opacity = 0.9;
      orionIndicatorSpriteMat.opacity = 0.9;
    }
  }

  // --- Ghost LIVE marker at real-time position ---
  ghostLiveGroup.visible = !isLiveMode && orionVisible;
  if (ghostLiveGroup.visible) {
    // Get real-time Orion position
    const realLiveMET = (Date.now() - Date.UTC(2026, 3, 1, 22, 35, 12)) / 1000;
    const realPos = getOrionFullPosition(realLiveMET);
    if (realPos) {
      const realVec = new THREE.Vector3(realPos.x, realPos.y, realPos.z);
      // Use Orion's current scale for consistent sizing
      positionIndicatorAboveOrion(ghostLiveGroup, realVec, orionMarker.scale);
    } else {
      ghostLiveGroup.visible = false;
    }
  }
}

function setLiveMode(isLive) {
  isLiveMode = isLive;
}

/* ============================================================
   MOON (sphere + orbit trace + motion model)
   ============================================================
   Simplified circular orbit, tuned so the Moon is at the correct
   position during the Artemis II lunar flyby (~MET 5d 00:29:59).

   The Moon's orbital parameters:
   - Average distance: 384,400 km = 384.4 u
   - Orbital period: ~27.322 days = 2,360,621 seconds
   - Flyby timing: Moon should be near Orion's closest approach point

   We derive the Moon's initial phase angle from the OEM: at flyby
   time, Orion is at max distance from Earth (~413u). The Moon should
   be approximately at that same angle from Earth.
   ============================================================ */

/**
 * Get Moon's real 3D position at a given MET using JPL Horizons ephemeris.
 * Returns THREE.Vector3 in scene units (Earth-relative, EME2000).
 */
function getMoonPositionAtMET(metSec) {
  const pos = getMoonEphemerisAtMET(metSec);
  if (pos) return new THREE.Vector3(pos.x, pos.y, pos.z);
  // Fallback: origin
  return new THREE.Vector3(0, 0, 0);
}

// --- Moon settings (adjustable via MOONSET debug modal) ---
const moonSettings = {
  color: 0xcccccc,
  emissive: 0x111111,
  emissiveIntensity: 0.05,
  shininess: 5,
  darkSideBrightness: 0.03,  // ambient light intensity on Moon
  sunLightIntensity: 1.8,    // directional light from Sun
  sunDirX: 1.0,
  sunDirY: 0.0,
  sunDirZ: 0.3
};

let moonMaterial = null;
let moonSunLight = null;
let moonAmbientLight = null;

function createMoon() {
  // Moon sphere with sharp terminator material
  const geo = new THREE.SphereGeometry(MOON_RADIUS, 32, 32);
  moonMaterial = new THREE.MeshPhongMaterial({
    color: moonSettings.color,
    emissive: moonSettings.emissive,
    emissiveIntensity: moonSettings.emissiveIntensity,
    shininess: moonSettings.shininess
  });
  moonMesh = new THREE.Mesh(geo, moonMaterial);
  earthMesh.add(moonMesh);

  // Dedicated ambient light for Moon dark side (very dim)
  moonAmbientLight = new THREE.AmbientLight(0xffffff, moonSettings.darkSideBrightness);
  moonMesh.add(moonAmbientLight);

  // Dedicated directional light from Sun direction for sharp terminator
  moonSunLight = new THREE.DirectionalLight(0xffffff, moonSettings.sunLightIntensity);
  moonSunLight.position.set(moonSettings.sunDirX, moonSettings.sunDirY, moonSettings.sunDirZ).normalize();
  // Light as child of earthMesh so it stays in Earth-relative coords
  // (Sun direction is constant relative to Earth since Sun is so far away)
  earthMesh.add(moonSunLight);

  // Moon SOI ring (66.2u radius — gravitational sphere of influence)
  const moonSoiGeo = new THREE.RingGeometry(65.7, 66.7, 64);
  const moonSoiMat = new THREE.MeshBasicMaterial({ color: 0xaaaaaa, side: THREE.DoubleSide, transparent: true, opacity: 0.3 });
  moonMesh.add(new THREE.Mesh(moonSoiGeo, moonSoiMat));
}

function applyMoonSettings() {
  if (moonMaterial) {
    moonMaterial.color.setHex(moonSettings.color);
    moonMaterial.emissive.setHex(moonSettings.emissive);
    moonMaterial.emissiveIntensity = moonSettings.emissiveIntensity;
    moonMaterial.shininess = moonSettings.shininess;
    moonMaterial.needsUpdate = true;
  }
  if (moonAmbientLight) {
    moonAmbientLight.intensity = moonSettings.darkSideBrightness;
  }
  if (moonSunLight) {
    moonSunLight.intensity = moonSettings.sunLightIntensity;
    moonSunLight.position.set(moonSettings.sunDirX, moonSettings.sunDirY, moonSettings.sunDirZ).normalize();
  }
}

function createMoonOrbitTrace() {
  // Full circle orbit trace
  const points = [];
  const steps = 128;
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * Math.PI * 2;
    points.push(new THREE.Vector3(
      MOON_ORBIT_RADIUS * Math.cos(angle),
      MOON_ORBIT_RADIUS * Math.sin(angle),
      0
    ));
  }
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineDashedMaterial({
    color: 0x888888,
    transparent: true,
    opacity: 0.2,
    dashSize: 10,
    gapSize: 5
  });
  moonOrbitLine = new THREE.Line(geo, mat);
  moonOrbitLine.computeLineDistances();
  earthMesh.add(moonOrbitLine); // child of Earth
}

function updateMoonPosition() {
  const pos = getMoonPositionAtMET(currentMET);
  moonMesh.position.copy(pos);
}

/* ============================================================
   STAGE TICKS AND LABELS
   ============================================================
   Placed along the trajectory at each stage's start/end MET.
   Labels only visible when that stage is selected.
   ============================================================ */

let stageTicks = []; // { metStart, metEnd, startMarker, endMarker, label, activityPips[] }
let selectedStageIndex = -1;

function createStageTicks(stages) {
  // stages: array of { name, metSeconds, metEndSeconds, ticks[] } from index.html
  stageTicks = stages.map((stage, i) => {
    const color = 0xffffff;

    // Start pip — hollow ring for stage boundary
    const startMarker = createTickMark(color, 0.6);
    const startPos = getOrionFullPosition(stage.metSeconds);
    if (startPos) startMarker.position.set(startPos.x, startPos.y, startPos.z + 0.05);
    earthMesh.add(startMarker);

    // End pip — hollow ring for stage boundary
    const endMarker = createTickMark(color, 0.6);
    const endPos = getOrionFullPosition(stage.metEndSeconds);
    if (endPos) endMarker.position.set(endPos.x, endPos.y, endPos.z + 0.05);
    earthMesh.add(endMarker);

    // Sub-stage activity pips — smaller solid circles along trajectory
    const activityPips = [];
    if (stage.ticks && stage.ticks.length > 0) {
      for (const tick of stage.ticks) {
        const pip = createTickMark(0xaaaaaa, 0.3); // smaller hollow ring, gray
        const pipPos = getOrionFullPosition(tick.metSeconds);
        if (pipPos) pip.position.set(pipPos.x, pipPos.y, pipPos.z + 0.05);
        earthMesh.add(pip);
        activityPips.push(pip);
      }
    }

    // Label sprite (only visible when selected)
    const label = createTextSprite(stage.name, '#ffffff');
    const midMET = (stage.metSeconds + stage.metEndSeconds) / 2;
    const midPos = getOrionFullPosition(midMET);
    if (midPos) label.position.set(midPos.x, midPos.y + 15, midPos.z);
    label.visible = false;
    earthMesh.add(label);

    return { metStart: stage.metSeconds, metEnd: stage.metEndSeconds, startMarker, endMarker, label, activityPips };
  });
}

function createTickMark(color, radius = 0.6) {
  // Hollow ring pip
  const innerR = radius * 0.6;
  const outerR = radius;
  const geo = new THREE.RingGeometry(innerR, outerR, 16);
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6, side: THREE.DoubleSide });
  return new THREE.Mesh(geo, mat);
}

function createTextSprite(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 32px monospace';
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 32);
  const texture = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.8 });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(30, 7.5, 1);
  return sprite;
}

function setSelectedStage(index) {
  selectedStageIndex = index;
  stageTicks.forEach((tick, i) => {
    tick.label.visible = (i === index);
  });
}

/* ============================================================
   ACTIVITY PIP FLASH EFFECT
   ============================================================ */

let activeFlashMesh = null;
let flashStartTime = 0;
const FLASH_DURATION = 800; // ms

function flashActivityPip(metSeconds) {
  // Find the pip closest to this MET
  const pos = getOrionFullPosition(metSeconds);
  if (!pos) return;

  // Remove existing flash
  if (activeFlashMesh) {
    earthMesh.remove(activeFlashMesh);
    activeFlashMesh.geometry.dispose();
    activeFlashMesh.material.dispose();
    activeFlashMesh = null;
  }

  // Create a slightly larger ring overlay at the pip position
  const geo = new THREE.RingGeometry(0.6, 1.0, 24);
  const mat = new THREE.MeshBasicMaterial({
    color: TRAJ_COLOR_CURRENT,
    transparent: true,
    opacity: 1.0,
    side: THREE.DoubleSide
  });
  activeFlashMesh = new THREE.Mesh(geo, mat);
  activeFlashMesh.position.set(pos.x, pos.y, pos.z + 0.15);
  earthMesh.add(activeFlashMesh);
  flashStartTime = Date.now();
}

function updateFlashEffect() {
  if (!activeFlashMesh) return;

  const elapsed = Date.now() - flashStartTime;
  if (elapsed > FLASH_DURATION) {
    earthMesh.remove(activeFlashMesh);
    activeFlashMesh.geometry.dispose();
    activeFlashMesh.material.dispose();
    activeFlashMesh = null;
    return;
  }

  // Fade out + scale up
  const t = elapsed / FLASH_DURATION;
  activeFlashMesh.material.opacity = 1.0 - t;
  const scale = 1.0 + t * 1.5; // grows 1x to 2.5x
  activeFlashMesh.scale.setScalar(scale);
}

/* ============================================================
   GHOST CIRCLES (clickable stage boundary markers on trajectory)
   ============================================================
   Ring/outline circles at each stage boundary along the trajectory.
   - Each circle belongs to the stage that STARTS at that MET
   - Plus one at the final stage's end MET
   - Zoom-adaptive: visible as clickable circles at zoom <= 20,
     hidden at zoom > 20 (existing tick marks shown instead)
   - Fixed 24px screen-space size for consistent mobile/desktop taps
   - Raycasting for click/tap hit detection
   ============================================================ */

const GHOST_CIRCLE_BASE_RADIUS = 1; // geometry base radius (scaled per frame)
const GHOST_CIRCLE_SCREEN_PX = 12;  // target screen-space diameter in pixels
const GHOST_CIRCLE_ZOOM_THRESHOLD = 20; // show circles ABOVE this zoom, ticks below

let ghostCircles = []; // { mesh, metSeconds, stageId, hitMesh }
let ghostCircleClickCallback = null;

// Raycaster for ghost circle hit detection
const raycaster = new THREE.Raycaster();
const raycastNDC = new THREE.Vector2();

function createGhostCircles(stages) {
  // Build boundary list: each stage's start, plus the last stage's end
  const boundaries = [];

  for (let i = 0; i < stages.length; i++) {
    boundaries.push({
      metSeconds: stages[i].metSeconds,
      stageId: stages[i].id
    });
  }
  // Add final stage end point
  const lastStage = stages[stages.length - 1];
  boundaries.push({
    metSeconds: lastStage.metEndSeconds,
    stageId: lastStage.id
  });

  // Remove duplicate MET boundaries (keep the one belonging to the LATER stage)
  const seen = new Map();
  for (const b of boundaries) {
    seen.set(b.metSeconds, b); // later entries overwrite earlier ones
  }
  const uniqueBoundaries = Array.from(seen.values());

  ghostCircles = uniqueBoundaries.map(b => {
    // Ring geometry: outline circle
    const innerR = GHOST_CIRCLE_BASE_RADIUS * 0.7;
    const outerR = GHOST_CIRCLE_BASE_RADIUS;
    const ringGeo = new THREE.RingGeometry(innerR, outerR, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00ffaa,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85
    });
    const mesh = new THREE.Mesh(ringGeo, ringMat);

    // Position along trajectory
    const pos = getOrionFullPosition(b.metSeconds);
    if (pos) mesh.position.set(pos.x, pos.y, pos.z + 0.1); // slight Z offset above trajectory

    // Invisible hit target (larger, for easier tapping)
    const hitGeo = new THREE.CircleGeometry(GHOST_CIRCLE_BASE_RADIUS * 1.5, 16);
    const hitMat = new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide });
    const hitMesh = new THREE.Mesh(hitGeo, hitMat);
    mesh.add(hitMesh);

    mesh.visible = false; // starts hidden, controlled by zoom
    earthMesh.add(mesh);

    return { mesh, hitMesh, ringMat, metSeconds: b.metSeconds, stageId: b.stageId };
  });
}

function updateGhostCircles() {
  if (!ghostCircles.length || !container) return;

  const rect = container.getBoundingClientRect();
  const viewHeight = (GRID_TOTAL * (1 + EDGE_BUFFER)) / zoomLevel;
  const worldUnitsPerPixel = viewHeight / rect.height;

  // Target world-space size for 24px screen diameter
  const desiredWorldRadius = (GHOST_CIRCLE_SCREEN_PX / 2) * worldUnitsPerPixel;
  const scaleFactor = desiredWorldRadius / GHOST_CIRCLE_BASE_RADIUS;

  const zoomedIn = zoomLevel >= GHOST_CIRCLE_ZOOM_THRESHOLD;

  // Get current stage's MET boundaries
  const currentSeg = stageSegments[currentTrajectoryStageIdx];
  const currentStart = currentSeg ? currentSeg.metStart : -1;
  const currentEnd = currentSeg ? currentSeg.metEnd : -1;

  // Only show ghost circles for current stage's start and end boundaries
  ghostCircles.forEach(gc => {
    const isCurrentStageBoundary = (gc.metSeconds === currentStart || gc.metSeconds === currentEnd);
    gc.mesh.visible = zoomedIn && isCurrentStageBoundary;
    if (gc.mesh.visible) {
      gc.mesh.scale.setScalar(scaleFactor);
    }
  });

  // Show + ticks for all non-current-stage boundaries (and all when zoomed out)
  stageTicks.forEach((tick, i) => {
    const isCurrentStage = (i === currentTrajectoryStageIdx);
    if (zoomedIn && isCurrentStage) {
      // Current stage uses circles, hide its ticks
      tick.startMarker.visible = false;
      tick.endMarker.visible = false;
    } else {
      // Other stages or zoomed out: show ticks
      tick.startMarker.visible = true;
      tick.endMarker.visible = true;
    }
  });
}

function onGhostCircleClick(callback) {
  ghostCircleClickCallback = callback;
}

function handleGhostCircleRaycast(screenX, screenY) {
  if (!ghostCircleClickCallback || !ghostCircles.length) return false;
  if (zoomLevel < GHOST_CIRCLE_ZOOM_THRESHOLD) return false;

  const rect = container.getBoundingClientRect();
  raycastNDC.x = ((screenX - rect.left) / rect.width) * 2 - 1;
  raycastNDC.y = -((screenY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(raycastNDC, camera);

  // Collect all ghost circle meshes (recursive catches child hit targets too)
  const hitTargets = ghostCircles.map(gc => gc.mesh);
  const intersects = raycaster.intersectObjects(hitTargets, true);

  if (intersects.length > 0) {
    // Find which ghost circle was hit
    const hitObj = intersects[0].object;
    const gc = ghostCircles.find(g =>
      g.mesh === hitObj || g.hitMesh === hitObj || hitObj.parent === g.mesh
    );
    if (gc) {
      ghostCircleClickCallback(gc.metSeconds, gc.stageId);
      return true;
    }
  }
  return false;
}

/* ============================================================
   AXIS LABELS (floating text at grid edges)
   ============================================================ */

function createAxisLabels() {
  // We use sprite-based text for axis labels so they always face camera
  const labels = [
    { text: '+X', x: GRID_TOTAL / 2 + 600, y: 0, color: '#ff4444' },
    { text: '-X', x: -GRID_TOTAL / 2 - 600, y: 0, color: '#ff4444' },
    { text: '+Y', x: 0, y: GRID_TOTAL / 2 + 600, color: '#44ff44' },
    { text: '-Y', x: 0, y: -GRID_TOTAL / 2 - 600, color: '#44ff44' },
  ];

  labels.forEach(({ text, x, y, color }) => {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.font = 'bold 48px monospace';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.6 });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.position.set(x, y, 0);
    sprite.scale.set(800, 400, 1);
    contentGroup.add(sprite);
  });
}

/* ============================================================
   PAN / ZOOM CONTROLS
   ============================================================ */

function pixelsToWorldUnits(pixelDelta) {
  const rect = container.getBoundingClientRect();
  const viewHeight = (GRID_TOTAL * (1 + EDGE_BUFFER)) / zoomLevel;
  return pixelDelta * (viewHeight / rect.height);
}

function setupControls() {
  const el = renderer.domElement;
  const CLICK_THRESHOLD = 5; // pixels — below this, treat as click not pan
  let mouseDownX = 0, mouseDownY = 0; // track for click detection

  // Mouse pan — modifies trackPanOffset if tracking, or moves camera directly if free
  el.addEventListener('mousedown', (e) => {
    camLerp = null;
    isPanning = true;
    mouseDownX = e.clientX;
    mouseDownY = e.clientY;
    panStartX = e.clientX;
    panStartY = e.clientY;
    if (trackTarget) {
      panStartTargetX = trackPanOffset.x;
      panStartTargetY = trackPanOffset.y;
    } else {
      panStartTargetX = cameraTarget.x;
      panStartTargetY = cameraTarget.y;
    }
    el.style.cursor = 'grabbing';
  });

  window.addEventListener('mousemove', (e) => {
    if (!isPanning) return;
    const dx = pixelsToWorldUnits(e.clientX - panStartX);
    const dy = pixelsToWorldUnits(e.clientY - panStartY);
    if (trackTarget) {
      trackPanOffset.set(panStartTargetX - dx, panStartTargetY + dy);
    } else {
      moveCamera(panStartTargetX - dx, panStartTargetY + dy);
      clampCameraTarget();
      updateCameraFrustum();
    }
  });

  window.addEventListener('mouseup', (e) => {
    // Check if this was a click (minimal movement) rather than a pan
    const dx = Math.abs(e.clientX - mouseDownX);
    const dy = Math.abs(e.clientY - mouseDownY);
    if (isPanning && dx < CLICK_THRESHOLD && dy < CLICK_THRESHOLD) {
      handleGhostCircleRaycast(e.clientX, e.clientY);
    }
    isPanning = false;
    el.style.cursor = 'default';
  });

  // Mouse wheel zoom — sets target for smooth interpolation
  el.addEventListener('wheel', (e) => {
    e.preventDefault();
    camLerp = null;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    targetZoomLevel = Math.max(0.5, Math.min(420, targetZoomLevel * factor));
  }, { passive: false });

  // Touch pan + pinch zoom
  let touchStartDist = 0, touchStartZoom = 1;
  let touchDownX = 0, touchDownY = 0; // track for tap detection

  el.addEventListener('touchstart', (e) => {
    e.preventDefault();
    camLerp = null;
    if (e.touches.length === 1) {
      isPanning = true;
      touchDownX = e.touches[0].clientX;
      touchDownY = e.touches[0].clientY;
      panStartX = e.touches[0].clientX;
      panStartY = e.touches[0].clientY;
      if (trackTarget) {
        panStartTargetX = trackPanOffset.x;
        panStartTargetY = trackPanOffset.y;
      } else {
        panStartTargetX = cameraTarget.x;
        panStartTargetY = cameraTarget.y;
      }
    } else if (e.touches.length === 2) {
      isPanning = false;
      const dx = e.touches[1].clientX - e.touches[0].clientX;
      const dy = e.touches[1].clientY - e.touches[0].clientY;
      touchStartDist = Math.sqrt(dx * dx + dy * dy);
      touchStartZoom = zoomLevel;
    }
  }, { passive: false });

  el.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (e.touches.length === 1 && isPanning) {
      const dx = pixelsToWorldUnits(e.touches[0].clientX - panStartX);
      const dy = pixelsToWorldUnits(e.touches[0].clientY - panStartY);
      if (trackTarget) {
        trackPanOffset.set(panStartTargetX - dx, panStartTargetY + dy);
      } else {
        moveCamera(panStartTargetX - dx, panStartTargetY + dy);
        clampCameraTarget();
        updateCameraFrustum();
      }
    } else if (e.touches.length === 2 && touchStartDist > 0) {
      const dx = e.touches[1].clientX - e.touches[0].clientX;
      const dy = e.touches[1].clientY - e.touches[0].clientY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      // Pinch sets both target and actual for responsive feel
      const newZoom = Math.max(0.5, Math.min(420, touchStartZoom * (dist / touchStartDist)));
      targetZoomLevel = newZoom;
      zoomLevel = newZoom; // direct for pinch responsiveness
      clampCameraTarget();
      updateCameraFrustum();
    }
  }, { passive: false });

  el.addEventListener('touchend', (e) => {
    // Check if this was a tap (minimal movement, single finger) rather than a pan
    if (isPanning && e.changedTouches.length === 1) {
      const touch = e.changedTouches[0];
      const dx = Math.abs(touch.clientX - touchDownX);
      const dy = Math.abs(touch.clientY - touchDownY);
      if (dx < CLICK_THRESHOLD && dy < CLICK_THRESHOLD) {
        handleGhostCircleRaycast(touch.clientX, touch.clientY);
      }
    }
    isPanning = false;
    touchStartDist = 0;
  }, { passive: false });
}

/* ============================================================
   CAMERA ANIMATION (smooth zoom-to-target)
   ============================================================ */

function updateCamLerp(dt) {
  if (!camLerp) return;

  camLerp.elapsed += dt;
  // Ease-out cubic: fast start, gentle arrival
  const raw = Math.min(1, camLerp.elapsed / camLerp.duration);
  const t = 1 - Math.pow(1 - raw, 3);

  // Lerp zoom (keep targetZoomLevel in sync)
  zoomLevel = camLerp.startZoom + (camLerp.endZoom - camLerp.startZoom) * t;
  targetZoomLevel = zoomLevel;

  // Lerp camera target position
  const lx = camLerp.startTarget.x + (camLerp.endTarget.x - camLerp.startTarget.x) * t;
  const ly = camLerp.startTarget.y + (camLerp.endTarget.y - camLerp.startTarget.y) * t;
  moveCamera(lx, ly);
  updateCameraFrustum();

  if (raw >= 1) {
    if (camLerp.onComplete) camLerp.onComplete();
    camLerp = null;
  }
}

/* ============================================================
   SMOOTH MET TRANSITION (ease-out lerp between time positions)
   ============================================================ */

let metLerp = null; // { startMET, endMET, elapsed, duration, onUpdate, onComplete }

function lerpToMET(targetMET, duration = 0.5, onUpdate = null, onComplete = null) {
  metLerp = {
    startMET: currentMET,
    endMET: targetMET,
    elapsed: 0,
    duration,
    onUpdate,
    onComplete
  };
}

function updateMetLerp(dt) {
  if (!metLerp) return;

  metLerp.elapsed += dt;
  const raw = Math.min(1, metLerp.elapsed / metLerp.duration);
  // Ease-out cubic
  const t = 1 - Math.pow(1 - raw, 3);

  const met = metLerp.startMET + (metLerp.endMET - metLerp.startMET) * t;
  setMissionTime(met, true); // snap Earth during lerp

  if (metLerp.onUpdate) metLerp.onUpdate(met);

  if (raw >= 1) {
    if (metLerp.onComplete) metLerp.onComplete();
    metLerp = null;
  }
}

/* ============================================================
   SMOOTH ZOOM (exponential ease-out toward targetZoomLevel)
   ============================================================ */

function lerpZoom(dt) {
  // Don't fight with camLerp — it controls zoom when active
  if (camLerp) return;

  const diff = targetZoomLevel - zoomLevel;
  if (Math.abs(diff) < 0.01) {
    zoomLevel = targetZoomLevel;
    return;
  }
  // Exponential ease-out: fast initial response, smooth deceleration
  const t = 1 - Math.exp(-ZOOM_LERP_SPEED * dt);
  zoomLevel += diff * t;
  clampCameraTarget();
  updateCameraFrustum();
}

/* ============================================================
   ANIMATION LOOP
   ============================================================ */

let lastFrameTime = 0;

function drawDebugHUD() {
  if (!debugVisible) return;
  const rect = container.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  debugCanvas.width = rect.width * dpr;
  debugCanvas.height = rect.height * dpr;
  const ctx = debugCtx;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.font = '12px monospace';
  ctx.fillStyle = '#39ff14';
  ctx.textBaseline = 'top';

  const ep = earthMesh.position;
  const et = earthTargetPos;
  const ct = cameraTarget;
  const cp = camera.position;
  const vH = ((GRID_TOTAL * (1 + EDGE_BUFFER)) / zoomLevel).toFixed(0);
  const half = (GRID_TOTAL / 2).toFixed(0);
  const mf = (currentMET / MISSION_DURATION_SEC).toFixed(4);

  const arcAngleDeg = (ARC_ANGLE * 180 / Math.PI).toFixed(2);
  const sunDir = `+X (right)`;

  const lines = [
    `cam pos:     (${cp.x.toFixed(1)}, ${cp.y.toFixed(1)}, ${cp.z.toFixed(1)})`,
    `cam target:  (${ct.x.toFixed(1)}, ${ct.y.toFixed(1)}, ${ct.z.toFixed(1)})`,
    `earth pos:   (${ep.x.toFixed(1)}, ${ep.y.toFixed(1)}, ${ep.z.toFixed(1)})`,
    `earth tgt:   (${et.x.toFixed(1)}, ${et.y.toFixed(1)}, ${et.z.toFixed(1)})`,
    `zoom:        ${zoomLevel.toFixed(2)}`,
    `viewHeight:  ${vH}u`,
    `grid:        -${half} to +${half}`,
    `MET frac:    ${mf}`,
    `MET sec:     ${currentMET.toFixed(0)}`,
    `camLerp:     ${camLerp ? (camLerp.elapsed / camLerp.duration * 100).toFixed(0) + '%' : 'none'}`,
    `--- ORBIT ---`,
    `orbit R:     ${ORBIT_RADIUS.toLocaleString()}u (1 AU)`,
    `arc angle:   ${arcAngleDeg}° (${ARC_ANGLE.toFixed(4)} rad)`,
    `chord:       ${CHORD_LENGTH.toFixed(0)}u`,
    `sagitta:     ${SAGITTA.toFixed(1)}u (max X bulge)`,
    `sun dir:     ${sunDir}`,
    `bulge dir:   -X (left, away from Sun)`,
    `earth X:     ${ep.x.toFixed(1)}u (expect ~-${SAGITTA.toFixed(0)} at mid)`,
    `--- AXES ---`,
    `+Y = up (screen top), +X = right, +Z = toward camera`,
    `velocity = -Y (down), Sun = +X (right)`,
  ];

  const x = 10, startY = 10, lineH = 16;
  // Background
  ctx.fillStyle = 'rgba(0,0,0,0.7)';
  ctx.fillRect(x - 4, startY - 4, 340, lines.length * lineH + 8);
  // Text
  ctx.fillStyle = '#39ff14';
  lines.forEach((line, i) => {
    ctx.fillText(line, x, startY + i * lineH);
  });
}

function renderFrame(timestamp) {
  const dt = lastFrameTime ? (timestamp - lastFrameTime) / 1000 : 0.016;
  lastFrameTime = timestamp;

  updateCamLerp(dt);
  updateMetLerp(dt);
  lerpZoom(dt);
  lerpEarth(dt);
  updateTracking();
  updateOrionMarkerScale();
  updateGhostCircles();
  updateLiveIndicator();
  updateFlashEffect();
  // Render starfield background first, then main scene on top
  renderer.clear();
  if (starScene && starCamera) renderer.render(starScene, starCamera);
  renderer.render(scene, camera);
  drawDebugHUD();
}

function startLoop() {
  function frame(timestamp) {
    animFrameId = requestAnimationFrame(frame);
    renderFrame(timestamp);
  }
  requestAnimationFrame(frame);
}

/* ============================================================
   PUBLIC API
   ============================================================ */

function setMissionTime(metSeconds, snapEarth = false) {
  currentMET = metSeconds;
  updateEarthPosition();
  // Snap Earth instantly when scrubbing, otherwise let it lerp
  if (snapEarth) {
    earthMesh.position.copy(earthTargetPos);
  }
  updateOrionPosition();
  updateMoonPosition();
  updateTrajectoryColors();
  // If camera lerp is still running, update its destination to follow Orion
  if (camLerp) {
    const orionWorld = getOrionWorldPosition();
    if (orionWorld) {
      camLerp.endTarget.copy(orionWorld);
    } else {
      camLerp.endTarget.copy(earthTargetPos);
    }
  }
}

/**
 * Get Orion's world position (for camera follow).
 * Returns the Earth position + Orion's Earth-relative position.
 */
function getOrionWorldPosition() {
  if (!orionMarker || !orionMarker.visible) return null;
  const worldPos = new THREE.Vector3();
  orionMarker.getWorldPosition(worldPos);
  return worldPos;
}

function getMissionFraction() {
  return currentMET / MISSION_DURATION_SEC;
}

function returnCameraToOrion() {
  // Smoothly animate camera back to Orion at zoom 45, then lock tracking
  const orionWorld = getOrionWorldPosition();
  if (!orionWorld) return;

  trackPanOffset.set(0, 0); // reset any user pan offset

  camLerp = {
    startTarget: cameraTarget.clone(),
    endTarget: orionWorld,
    startZoom: zoomLevel,
    endZoom: 360,
    elapsed: 0,
    duration: 1.4,
    onComplete: () => {
      setTrackTarget('orion');
      targetZoomLevel = 360;
    }
  };
}

function dispose() {
  if (animFrameId) cancelAnimationFrame(animFrameId);
  renderer.dispose();
  if (container && renderer.domElement.parentElement === container) {
    container.removeChild(renderer.domElement);
  }
}

export default {
  init,
  setMissionTime,
  getMissionFraction,
  createStageTicks,
  setSelectedStage,
  setTrackTarget,
  setDebugVisible,
  createGhostCircles,
  onGhostCircleClick,
  createStageTrajectorySegments,
  setLiveMode,
  returnCameraToOrion,
  lerpToMET,
  flashActivityPip,
  moonSettings,
  applyMoonSettings,
  dispose,
  MISSION_DURATION_SEC,
  GRID_TOTAL,
  EARTH_RADIUS
};
