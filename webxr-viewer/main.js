import * as THREE from 'three';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadSplatFile } from './splat-loader.js';

// ---- Config: which memory to load -----------------------------------------
// Read from the URL so a query call can eventually deep-link straight to a
// specific memory: index.html?scene=/scenes/park.ply&note=Park+visit
const params = new URLSearchParams(window.location.search);
const SCENE_URL = params.get('scene') || './scenes/placeholder.splat';
const MEMORY_NOTE = params.get('note') || '(no memory loaded)';
const MEMORY_SOURCE = params.get('source') || '';

// ---- DOM refs --------------------------------------------------------------
const canvas = document.getElementById('scene-canvas');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingBarFill = document.getElementById('loading-bar-fill');
const loadingPercent = document.getElementById('loading-percent');
const enterVrButton = document.getElementById('enter-vr');
const statusEl = document.getElementById('status');
const noWebxrNotice = document.getElementById('no-webxr-notice');
const memoryNoteEl = document.getElementById('memory-note');
const memorySourceEl = document.getElementById('memory-source');

memoryNoteEl.textContent = MEMORY_NOTE;
memorySourceEl.textContent = MEMORY_SOURCE;

// ---- Renderer / scene / camera --------------------------------------------
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x16141c);

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 100);

// A dolly rig lets us reposition the whole camera/controller group inside
// the scene. This matters specifically because WebXR takes over
// camera.position directly once a VR session starts (driven by the
// headset's tracked pose) — setting camera.position manually has no
// effect in VR. To place the user somewhere sensible in world space, we
// instead position the dolly, and WebXR's head-tracking becomes a local
// offset from wherever the dolly sits.
const dolly = new THREE.Group();
dolly.add(camera);
scene.add(dolly);

// Desktop-only orbit controls for the flythrough preview — WebXR sessions
// use the headset's own tracked pose instead, so these get disabled the
// moment a VR session starts. For the desktop preview, camera.position
// DOES work directly (no XR session overriding it), so we set it here.
camera.position.set(0, 1.6, 3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.2, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---- Splat rendering --------------------------------------------------------
// Real production splat renderers use a custom sorted-billboard shader with
// covariance-based screen-space ellipse projection. For a hackathon-scope
// viewer, we approximate each splat as a camera-facing sprite sized by its
// average scale and colored/opacity from the trained data — visually close
// for typical indoor room scans, dramatically simpler to implement and
// debug, and still genuinely reads as a walkable 3D point cloud in VR.
function buildSplatPoints(splatData) {
  const { count: rawCount, data: rawData } = splatData;

  // 5.8M individual splats, rendered as one GPU point each with no culling
  // or LOD, is far beyond what a mobile-class GPU (Quest 3) or even most
  // desktop browsers can sustain at a usable framerate — this is what was
  // actually causing the freeze/extreme slowdown, not a bug. Decimate to a
  // fixed cap by evenly sampling every Nth splat, which is fast, simple,
  // and preserves the overall shape/density distribution of the scan
  // reasonably well for a walkable preview (real production splat viewers
  // do proper LOD; this is the hackathon-scope equivalent).
  const MAX_RENDERED_SPLATS = 400000;
  const stride = rawCount > MAX_RENDERED_SPLATS ? Math.ceil(rawCount / MAX_RENDERED_SPLATS) : 1;
  const count = Math.ceil(rawCount / stride);

  if (stride > 1) {
    console.log(
      `[buildSplatPoints] ${rawCount.toLocaleString()} splats exceeds render cap — ` +
      `sampling every ${stride}th splat (~${count.toLocaleString()} rendered)`
    );
  }

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 4);
  const sizes = new Float32Array(count);

  let outIdx = 0;
  for (let i = 0; i < rawCount; i += stride) {
    const o = i * 14;
    positions[outIdx * 3 + 0] = rawData[o + 0];
    positions[outIdx * 3 + 1] = rawData[o + 1];
    positions[outIdx * 3 + 2] = rawData[o + 2];

    colors[outIdx * 4 + 0] = rawData[o + 3];
    colors[outIdx * 4 + 1] = rawData[o + 4];
    colors[outIdx * 4 + 2] = rawData[o + 5];
    colors[outIdx * 4 + 3] = rawData[o + 6];

    const avgScale = (rawData[o + 7] + rawData[o + 8] + rawData[o + 9]) / 3;
    // Slight size compensation for decimation, but much more conservative
    // than sqrt(stride) — that was making points so large they overlapped
    // into indistinct blobs rather than reading as a room shape. A small
    // fixed multiplier keeps points visually distinct while still closing
    // some of the gaps left by sampling fewer of them.
    sizes[outIdx] = Math.max(0.005, avgScale) * Math.min(1.5, Math.sqrt(stride) * 0.4);

    outIdx++;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('splatColor', new THREE.BufferAttribute(colors, 4));
  geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

  const material = new THREE.ShaderMaterial({
    uniforms: {
      pointScale: { value: 250.0 }, // reduced from 800 — was making points overlap into indistinct blobs
    },
    vertexShader: `
      attribute float size;
      attribute vec4 splatColor;
      varying vec4 vColor;
      uniform float pointScale;
      void main() {
        vColor = splatColor;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * pointScale / -mvPosition.z;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      varying vec4 vColor;
      void main() {
        vec2 uv = gl_PointCoord - 0.5;
        float d = length(uv);
        if (d > 0.5) discard;
        float falloff = 1.0 - smoothstep(0.3, 0.5, d);
        gl_FragColor = vec4(vColor.rgb, vColor.a * falloff);
      }
    `,
    transparent: true,
    depthWrite: false,
    // Note: NOT setting vertexColors:true here — that flag makes Three.js
    // auto-inject its own "attribute vec3 color" into the shader, which
    // collided with (and shadowed) our custom color attribute above. We
    // handle color entirely ourselves via splatColor/vColor instead.
  });

  return new THREE.Points(geometry, material);
}

function recenterSceneOnLoad(points) {
  // Room scans aren't guaranteed to be centered at the world origin, and a
  // splat cloud that's off to one side is disorienting the moment you put
  // a headset on. Compute the bounding box and shift the geometry so its
  // horizontal centre sits at the origin, with the floor near y=0.
  points.geometry.computeBoundingBox();
  const box = points.geometry.boundingBox;

  const widthX = box.max.x - box.min.x;
  const heightY = box.max.y - box.min.y;
  const depthZ = box.max.z - box.min.z;

  console.log('Scene bounding box (before recenter):', {
    x: [box.min.x.toFixed(3), box.max.x.toFixed(3)],
    y: [box.min.y.toFixed(3), box.max.y.toFixed(3)],
    z: [box.min.z.toFixed(3), box.max.z.toFixed(3)],
    width_x: widthX.toFixed(3),
    height_y: heightY.toFixed(3),
    depth_z: depthZ.toFixed(3),
  });

  // The source reconstruction's Y axis points the opposite way from what
  // Three.js/WebXR expect (confirmed visually: scene rendered upside down
  // before this flip). Rather than trust an assumed convention, we flip Y
  // directly on the geometry's scale, which is simpler and more reliable
  // than trying to reason about rotation matrices for a point-cloud that
  // has no inherent "front" direction anyway.
  points.scale.y *= -1;

  // Different reconstruction pipelines output in different, often
  // arbitrary units — not necessarily real-world meters. A splat cloud
  // spanning 100+ units is either a huge space, or (far more likely for a
  // single-room scan) the source units just aren't meters. Rather than
  // assume, normalize based on the measured horizontal extent so a typical
  // room-scale scan ends up roughly human-scale (~4-6m across) regardless
  // of what units the source file actually used.
  const largestHorizontal = Math.max(widthX, depthZ);
  const TARGET_ROOM_SIZE = 5; // meters — reasonable single-room scan width
  const scaleFactor = largestHorizontal > 0 ? TARGET_ROOM_SIZE / largestHorizontal : 1;

  if (Math.abs(scaleFactor - 1) > 0.01) {
    console.log(
      `[recenterSceneOnLoad] scene extent (${largestHorizontal.toFixed(1)} units) doesn't look ` +
      `room-scale — applying scale factor ${scaleFactor.toFixed(4)} to normalize to ~${TARGET_ROOM_SIZE}m`
    );
    points.scale.setScalar(scaleFactor);
  }

  const centerX = (box.min.x + box.max.x) / 2;
  const centerZ = (box.min.z + box.max.z) / 2;
  const floorY = box.min.y;

  // Position offset must also be scaled, since points.position is applied
  // in the parent's (unscaled) space, but we're centering the scaled child.
  points.position.set(-centerX * scaleFactor, -floorY * scaleFactor, -centerZ * scaleFactor);
}

// ---- Loading -----------------------------------------------------------
async function loadScene() {
  try {
    statusEl.textContent = `loading ${SCENE_URL}…`;
    console.log('[loadScene] starting fetch...');
    const splatData = await loadSplatFile(SCENE_URL, (fraction) => {
      const pct = Math.round(fraction * 100);
      loadingBarFill.style.width = `${pct}%`;
      loadingPercent.textContent = `${pct}%`;
    });
    console.log(`[loadScene] fetch + parse complete: ${splatData.count.toLocaleString()} splats`);

    statusEl.textContent = `building geometry for ${splatData.count.toLocaleString()} splats…`;
    console.log('[loadScene] building geometry...');
    const points = buildSplatPoints(splatData);
    console.log('[loadScene] geometry built, recentering...');

    recenterSceneOnLoad(points);
    console.log('[loadScene] recentered, adding to scene...');
    scene.add(points);

    // Place the VR starting position a few steps back from room-center and
    // at standing eye-height, so the headset wearer starts with an actual
    // view of the room rather than standing at the exact origin (which,
    // after recentering, is room-center — likely inside/adjacent to
    // whatever object sits in the middle of the scan).
    dolly.position.set(0, 0, 2);
    console.log('[loadScene] dolly positioned for VR start point');
    console.log('[loadScene] done — scene should now be visible');

    loadingPercent.textContent = `${splatData.count.toLocaleString()} splats`;
    loadingOverlay.classList.add('hidden');
    statusEl.textContent = 'flythrough preview — drag to orbit, scroll to zoom';
  } catch (err) {
    console.error(err);
    loadingOverlay.classList.add('hidden');
    statusEl.textContent = `failed to load scene: ${err.message}`;
    statusEl.classList.add('error');
  }
}

// ---- WebXR session setup ------------------------------------------------
function setupWebXR() {
  if (!('xr' in navigator)) {
    statusEl.style.display = 'none';
    noWebxrNotice.style.display = 'block';
    enterVrButton.remove();
    return;
  }

  navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
    if (!supported) {
      statusEl.style.display = 'none';
      noWebxrNotice.style.display = 'block';
      enterVrButton.remove();
      return;
    }

    enterVrButton.disabled = false;
    enterVrButton.textContent = 'Enter VR';

    // Three.js's VRButton handles the session request/end lifecycle, but we
    // want our own styled button, so we borrow its click handler rather
    // than inserting its default DOM element.
    const vrButtonProxy = VRButton.createButton(renderer);
    enterVrButton.addEventListener('click', () => vrButtonProxy.click());

    renderer.xr.addEventListener('sessionstart', () => {
      controls.enabled = false;
      statusEl.style.display = 'none';
    });
    renderer.xr.addEventListener('sessionend', () => {
      controls.enabled = true;
      statusEl.style.display = 'block';
    });
  });
}

// ---- Render loop ------------------------------------------------------
renderer.setAnimationLoop(() => {
  if (!renderer.xr.isPresenting) {
    controls.update();
  }
  renderer.render(scene, camera);
});

loadScene();
setupWebXR();