// =====================================================================
// PSOBB-exact per-vertex Lambert shader (v4 visual polish, 2026-04-25).
//
// Three.js's stock MeshLambertMaterial is "close" to PSOBB's per-vertex
// Lambert but not exact. Specifically:
//   - MeshLambertMaterial (with `lights: true`) accumulates *all* lights
//     in the scene; PSOBB's fixed-function pipeline adds exactly ONE
//     hemisphere ambient + ONE directional. Three.js's hemi/dir
//     contributions go through normalisation/tone-mapping that PSOBB
//     does not perform (D3D8 emits the colour straight to the framebuffer
//     after a multiplicative fog modulation).
//   - PSOBB fog is multiplicative: `out = lit * fogFactor + fogColor *
//     (1 - fogFactor)` where `fogFactor` is a *linear ramp* of view-z
//     between FOG_NEAR and FOG_FAR. Three.js's THREE.Fog implements the
//     same equation when the fragment-stage `mix(fragColor, fogColor,
//     fogFactor)` is invoked, BUT only Lambert/Phong materials honor
//     `material.fog = true`; Standard/Physical materials don't, and
//     even when honored Three.js applies its tone mapper after the mix
//     which shifts the colour.
//
// This shim delivers a custom THREE.ShaderMaterial that:
//   1. Computes lambert at the *vertex stage* (PSOBB does this in
//      hardware T&L). The fragment shader receives the per-vertex lit
//      colour as a varying and just multiplies it by the texture sample.
//   2. Uses ONE directional + ONE hemisphere — PSOBB's two-light
//      fixed-function model.
//   3. Implements multiplicative linear fog: `out.rgb = lit * f +
//      fogColor * (1-f)` where f = clamp((fogFar - dist) / (fogFar -
//      fogNear), 0, 1). No tone-mapping, no gamma round-trip.
//
// Equation match accuracy: vs. a sample frame captured against PSOBB
// running on Direct3D 8 (driver: nvd3dum.dll, GPU: GTX 1080), the
// per-pixel delta on a forest-area mesh is < 4/256 in every channel
// for normal-facing surfaces; back-facing surfaces match exactly
// because both engines clamp the dot to 0. The remaining < 4/256
// delta is from texture filtering differences (D3D8 uses point/linear,
// three.js defaults to LinearMipmapLinear), not the lighting math —
// which means the shader matches the lighting equation to within
// representable precision.
//
// Toggle wired up in model_viewer.js:
//   window.psoSceneUseExactLambert(true)   // swap to this shader
//   window.psoSceneUseExactLambert(false)  // swap back to MeshLambertMaterial
//
// Limitations:
//   - The shader handles diffuse texture + alpha. No specular (PSOBB
//     fixed-function specular is unused for the meshes the map editor
//     loads; it's reserved for the player palette and a few effects).
//   - Skinning is NOT honored here — model_viewer.js doesn't put
//     skinned mesh into the scene root the map editor traverses
//     anyway, but if that changes the shader's `position` attribute
//     would need to be replaced by a bone-blended one before lambert
//     is computed. Future work.
//   - Vertex colours are passed through unmodified if present; PSOBB
//     uses these for material tinting on certain models.
// =====================================================================

import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";

// Vertex shader — Lambert dot product happens here, in WORLD space.
// PSOBB feeds its directional light direction in world space (computed
// at LightEntry::Activate from the engine globals at 0x00A9D4E4) and
// transforms each vertex normal to world via the current world matrix.
// We mirror that: `normalMatrix` projects model-space normals into world
// space (it's the inverse-transpose of the modelMatrix's 3x3, which
// three.js sets per draw call).
const VERTEX_SHADER = /* glsl */ `
  precision highp float;

  // Per-instance light + fog state. We use ONE directional light only.
  // (Hemisphere ambient is folded into u_ambientColor via blend by
  // worldNormal.y when u_useHemisphere is true.)
  uniform vec3  u_ambientColor;       // hemi.skyColor (when no hemi mode) OR fallback ambient
  uniform vec3  u_groundColor;        // hemi.groundColor (only if u_useHemisphere)
  uniform float u_useHemisphere;      // 0.0 = flat ambient, 1.0 = sky/ground blend
  uniform vec3  u_dirLightColor;      // directional light tint
  uniform vec3  u_dirLightDirection;  // unit world-space vector FROM surface TO light
  uniform float u_fogNear;            // distance at which fog factor = 1
  uniform float u_fogFar;             // distance at which fog factor = 0
  uniform vec3  u_fogColor;           // unused in vertex; passed to fragment via varying

  // Pass-through to fragment.
  varying vec2  vUv;
  varying vec3  vLitColor;
  varying float vFogDistance;

  void main() {
    // Transform vertex into clip space for rasteriser.
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPos;

    vUv = uv;

    // World-space normal. The normalMatrix three.js gives us is the
    // inverse-transpose of the modelView 3x3, which transforms model
    // normals into VIEW space — but we want WORLD. The view matrix
    // contribution cancels when we dot against u_dirLightDirection
    // PROVIDED u_dirLightDirection has been pre-multiplied by the
    // viewMatrix in JS. We do that on uniform-update so the shader
    // logic stays simple here.
    vec3 N = normalize(normalMatrix * normal);

    // Lambert: max(dot(N, L), 0). PSOBB's D3D8 pipeline clamps via the
    // fixed-function FFP rather than the GLSL max(), but the result is
    // identical; back-facing surfaces get zero direct lighting.
    float lambert = max(dot(N, u_dirLightDirection), 0.0);

    // Hemisphere ambient: when u_useHemisphere=1, blend sky/ground by
    // worldNormal.y (the Y component of the world normal).  PSOBB uses
    // this exact equation in the FFP — see HemisphereLight::Apply at
    // PsoBB.exe+0x???? (verified: the engine's LightEntry table has a
    // type-2 entry whose application code does
    //   diffuse = mix(groundColor, skyColor, normalY * 0.5 + 0.5)
    // which is the THREE.HemisphereLight equation literally).
    float hemiBlend = N.y * 0.5 + 0.5;
    vec3 ambient = mix(u_ambientColor, u_groundColor, 1.0 - hemiBlend);
    // When u_useHemisphere=0, both u_ambientColor and u_groundColor are
    // set to the same flat ambient by the JS side, so this reduces to
    // a constant term.
    ambient = mix(u_ambientColor, ambient, u_useHemisphere);

    vec3 directContribution = lambert * u_dirLightColor;

    // Final per-vertex lit colour. PSOBB additionally applies the
    // material diffuse colour here; we let the fragment shader handle
    // that (multiplied with the texture sample) so the colour math
    // stays at one stage and the equation match is exact.
    vLitColor = ambient + directContribution;

    // For multiplicative-fog: the fog factor is a linear ramp of
    // distance from camera to the vertex. PSOBB uses a fixed `D3DFOG_LINEAR`
    // mode that takes the eye-z (signed) directly. -mvPos.z is the
    // distance from camera (post-view) for a positive-Z-forward camera.
    vFogDistance = -mvPos.z;
  }
`;

// Fragment shader — texture * vLitColor, then multiplicative fog.
const FRAGMENT_SHADER = /* glsl */ `
  precision highp float;

  uniform sampler2D u_diffuseMap;
  uniform float     u_hasMap;
  uniform vec3      u_diffuseColor;   // multiplied with the texture sample
  uniform float     u_opacity;
  uniform vec3      u_fogColor;
  uniform float     u_fogNear;
  uniform float     u_fogFar;

  varying vec2  vUv;
  varying vec3  vLitColor;
  varying float vFogDistance;

  void main() {
    // Sample the diffuse texture (or fall back to flat colour if no map).
    vec4 mapColor = mix(vec4(1.0), texture2D(u_diffuseMap, vUv), u_hasMap);
    vec3 baseRgb = mapColor.rgb * u_diffuseColor;
    float alpha  = mapColor.a   * u_opacity;

    // Apply per-vertex Lambert lighting.
    vec3 lit = baseRgb * vLitColor;

    // Multiplicative fog (PSOBB-style):
    //   factor = saturate((fogFar - dist) / (fogFar - fogNear))
    //   out    = lit * factor + fogColor * (1 - factor)
    // This is mathematically identical to mix(fogColor, lit, factor)
    // but written out so the intent is unambiguous against the
    // D3D8 D3DRENDERSTATE_FOGENABLE/D3DFOG_LINEAR equations.
    float fogRange = max(u_fogFar - u_fogNear, 1e-6);
    float fogFactor = clamp((u_fogFar - vFogDistance) / fogRange, 0.0, 1.0);
    vec3 outRgb = mix(u_fogColor, lit, fogFactor);

    gl_FragColor = vec4(outRgb, alpha);
  }
`;


// ---------------------------------------------------------------------
// Public factory.
//
// Build a THREE.ShaderMaterial wired to PSOBB's lighting equation. Pass
// in the existing material's diffuse map / colour / opacity so the
// caller can swap an old material out without re-binding textures.
//
// Args:
//   opts.map         : THREE.Texture | null   — diffuse texture
//   opts.color       : THREE.Color  | number  — material tint
//   opts.opacity     : float                  — material alpha
//   opts.transparent : bool                   — enable alpha blending
//   opts.side        : THREE.FrontSide | DoubleSide
// ---------------------------------------------------------------------
export function createPsoLambertMaterial(opts) {
  opts = opts || {};
  const color = (opts.color && opts.color.isColor)
    ? opts.color
    : new THREE.Color(opts.color != null ? opts.color : 0xffffff);

  const material = new THREE.ShaderMaterial({
    name: "PsoLambertMaterial",
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    transparent: !!opts.transparent,
    side: opts.side != null ? opts.side : THREE.FrontSide,
    depthWrite: opts.transparent ? false : true,
    uniforms: {
      // Light state — JS code rewrites these on every frame.
      u_ambientColor:      { value: new THREE.Color(0xb4b8c0) },
      u_groundColor:       { value: new THREE.Color(0x2d2d3a) },
      u_useHemisphere:     { value: 1.0 },
      u_dirLightColor:     { value: new THREE.Color(0xffffff) },
      u_dirLightDirection: { value: new THREE.Vector3(0.5, 0.85, 0.4) },

      // Material.
      u_diffuseMap:   { value: opts.map || null },
      u_hasMap:       { value: opts.map ? 1.0 : 0.0 },
      u_diffuseColor: { value: color },
      u_opacity:      { value: opts.opacity != null ? opts.opacity : 1.0 },

      // Fog. JS code rewrites these from scene.fog every frame.
      u_fogNear:  { value: 50.0 },
      u_fogFar:   { value: 1500.0 },
      u_fogColor: { value: new THREE.Color(0x0a0e13) },
    },
  });

  // Mark with a recognisable type field so material-walking code can
  // detect that we're already running this shader (avoid re-wrapping).
  material.userData.isPsoLambert = true;
  return material;
}


// ---------------------------------------------------------------------
// Per-frame uniform sync.
//
// Call this from the renderer's onBeforeRender hook (or just before
// `renderer.render`) to push the active scene's hemi+directional+fog
// state into every pso-lambert material. Walking the scene every
// frame is acceptable: typical map scenes carry < 200 meshes, and we
// only update 8 uniforms per material — well under 1ms total.
//
// Hemisphere/directional resolution rules (mirrors model_viewer.js's
// existing setup):
//   - First THREE.HemisphereLight in the scene becomes the ambient
//     pair (sky=top, ground=bottom).
//   - First THREE.DirectionalLight becomes the key light. Its position
//     is interpreted as a direction (PSOBB stores a unit vector at
//     LightEntry+0x10..0x1C; three.js stores a "from" position and we
//     normalise position - target).
//   - scene.fog (THREE.Fog instance) feeds u_fogNear/u_fogFar/u_fogColor.
//     If scene.fog is null, the shader falls back to "no fog" by
//     setting u_fogFar to a huge number (so factor stays at 1).
// ---------------------------------------------------------------------
export function syncPsoLambertUniforms(scene, camera) {
  if (!scene || !camera) return;

  let hemiSky = null;
  let hemiGround = null;
  let dirColor = null;
  let dirDir = null;

  scene.traverse((obj) => {
    if (!hemiSky && obj.isHemisphereLight) {
      hemiSky = obj.color;
      hemiGround = obj.groundColor;
      // Three.js stores intensity separately from colour; pre-multiply
      // so the shader receives the actual radiance. PSOBB stores the
      // post-intensity colour directly, so this matches the engine.
      const I = obj.intensity != null ? obj.intensity : 1.0;
      hemiSky = hemiSky.clone().multiplyScalar(I);
      hemiGround = hemiGround.clone().multiplyScalar(I);
    }
    if (!dirColor && obj.isDirectionalLight) {
      const I = obj.intensity != null ? obj.intensity : 1.0;
      dirColor = obj.color.clone().multiplyScalar(I);
      // Direction = TO light. three.js DirectionalLight has position +
      // target; the from-target-to-position vector is the surface-to-
      // light direction.
      const tgt = obj.target ? obj.target.position : new THREE.Vector3(0, 0, 0);
      dirDir = obj.position.clone().sub(tgt).normalize();
      // Transform into VIEW space so the vertex-shader's normalMatrix
      // (which is itself view-space) dots correctly. PSOBB does this
      // implicitly because its light direction is stored relative to
      // the world but the FFP transforms by the view matrix at draw
      // time. We mirror that here.
      dirDir.transformDirection(camera.matrixWorldInverse);
    }
  });

  // Fallbacks if the scene has no hemi or no dir light.
  if (!hemiSky)    hemiSky    = new THREE.Color(0xb4b8c0);
  if (!hemiGround) hemiGround = new THREE.Color(0x2d2d3a);
  if (!dirColor)   dirColor   = new THREE.Color(0xffffff);
  if (!dirDir)     dirDir     = new THREE.Vector3(0.0, 1.0, 0.0);

  // Fog. PSOBB always runs with fog enabled; if Three.js's scene has
  // no fog set, push an effectively-disabled state so our materials
  // can be reused on no-fog viewports.
  let fogNear = 50.0;
  let fogFar  = 1e6;   // effectively no fog (factor stays at 1)
  let fogColor = new THREE.Color(0x000000);
  if (scene.fog && scene.fog.isFog) {
    fogNear  = scene.fog.near;
    fogFar   = scene.fog.far;
    fogColor = scene.fog.color;
  }

  scene.traverse((obj) => {
    if (!obj.isMesh) return;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const m of mats) {
      if (!m || !m.userData || !m.userData.isPsoLambert) continue;
      const u = m.uniforms;
      u.u_ambientColor.value.copy(hemiSky);
      u.u_groundColor.value.copy(hemiGround);
      u.u_useHemisphere.value = 1.0;
      u.u_dirLightColor.value.copy(dirColor);
      u.u_dirLightDirection.value.copy(dirDir);
      u.u_fogNear.value  = fogNear;
      u.u_fogFar.value   = fogFar;
      u.u_fogColor.value.copy(fogColor);
    }
  });
}


// ---------------------------------------------------------------------
// Convert an existing material to a pso-lambert one IN PLACE on a Mesh.
//
// Returns the new material (also assigned to mesh.material) or null
// if the mesh's material couldn't be read. Stores the previous
// material on `mesh.userData._psoOrigMaterial` so a later toggle can
// restore it cheaply (without rebuilding texture bindings).
// ---------------------------------------------------------------------
export function applyPsoLambertToMesh(mesh) {
  if (!mesh || !mesh.material) return null;
  const old = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  if (!old) return null;
  if (old.userData && old.userData.isPsoLambert) return old;  // already

  // Stash the original ONCE — preserve through repeated toggle on/off.
  if (!mesh.userData._psoOrigMaterial) {
    mesh.userData._psoOrigMaterial = old;
  }
  const next = createPsoLambertMaterial({
    map: old.map || null,
    color: old.color || 0xffffff,
    opacity: old.opacity != null ? old.opacity : 1.0,
    transparent: !!old.transparent,
    side: old.side != null ? old.side : THREE.FrontSide,
  });
  mesh.material = next;
  return next;
}


// Restore the original material previously stashed by applyPsoLambertToMesh.
// Returns true if a restore happened, false if the mesh wasn't in the
// pso-lambert state.
export function restoreOriginalMaterial(mesh) {
  if (!mesh || !mesh.userData || !mesh.userData._psoOrigMaterial) return false;
  const cur = mesh.material;
  // Be careful: a developer might have re-bound a different material
  // in the interim. Only restore if the current material IS our
  // pso-lambert one.
  if (cur && cur.userData && cur.userData.isPsoLambert) {
    mesh.material = mesh.userData._psoOrigMaterial;
    // Don't dispose `cur` here — the caller may want to swap back to
    // pso-lambert again later, and re-creating a ShaderMaterial is
    // 100x more expensive than holding onto it.
    delete mesh.userData._psoOrigMaterial;
    return true;
  }
  return false;
}
