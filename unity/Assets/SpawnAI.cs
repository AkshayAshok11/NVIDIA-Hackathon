using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Condensed, particle-based "AI atom" hologram matching the classic
/// glowing-sphere-with-orbit-rings look:
///  - Dense particle shell forming the core (with a soft additive glow blob behind it)
///  - Several tilted rings (React-logo style angles) made entirely of particles:
///      a thin faint arc + orbiting "dust" + a few bright orbiting "node" particles
///  - Everything moves: core shimmers, dust + nodes orbit, and the ring pivots
///    themselves slowly tumble.
///
/// WHY YOUR GLOW WASN'T WORKING (fixed here):
///  1. The old core was an OPAQUE lit sphere. Emission on an opaque Standard/Lit
///     material does nothing visually unless Bloom post-processing is active.
///     Here the core is additive/transparent particles + a soft radial-gradient
///     billboard, so it visibly glows even with zero post-processing.
///  2. Additive blending is what actually "glows" — overlapping bright particles
///     sum together. This script forces additive blend on every material it makes.
///  3. For the strongest look, still add Bloom (URP: Volume -> Bloom, Threshold ~0.8,
///     Intensity ~1.5–3; also enable HDR on the URP asset & camera). But it will
///     look correct without it now.
///
/// USAGE: Attach to an empty GameObject and press Play.
/// </summary>
[DisallowMultipleComponent]
public class JarvisAtomHologram : MonoBehaviour
{
    [Header("Overall")]
    [Tooltip("Scales the entire hologram. Keep it tight/condensed like the reference.")]
    public float overallScale = 1f;
    public Color glowColor = new Color(0.35f, 0.75f, 1f);
    [Range(1f, 10f)] public float emissionIntensity = 5f;

    [Header("Core")]
    public float coreRadius = 0.28f;
    [Range(200, 4000)] public int coreParticleCount = 1600;
    public float coreParticleSize = 0.012f;
    [Range(0f, 0.5f)] public float coreShellThickness = 0.15f; // 0 = perfect shell, higher = fuzzier
    public float coreGlowBlobSize = 1.6f; // relative to coreRadius
    [Range(0.1f, 3f)] public float pulseSpeed = 1.1f;
    [Range(0f, 1f)] public float pulseAmount = 0.35f;

    [Header("Rings")]
    [Range(1, 4)] public int ringCount = 3;
    public float minRingRadius = 0.45f;
    public float maxRingRadius = 0.62f;
    public bool useClassicAtomAngles = true; // fan rings like the reference / React-logo style
    public float classicTiltDegrees = 72f;
    public Vector2 ringPivotSpinRange = new Vector2(8f, 22f); // deg/sec, whole ring tumbling

    [Header("Ring Dust (thin orbiting particle trail)")]
    [Range(0, 120)] public int dustParticlesPerRing = 55;
    public float dustParticleSize = 0.01f;
    public Vector2 dustOrbitalSpeedRange = new Vector2(0.6f, 1.4f);
    public bool showFaintArcLine = true;
    public float arcLineWidth = 0.006f;

    [Header("Ring Nodes (bright accent particles)")]
    [Range(0, 12)] public int nodesPerRing = 4;
    public float nodeParticleSize = 0.035f;
    public Vector2 nodeOrbitalSpeedRange = new Vector2(0.2f, 0.5f);

    // internal
    private Transform root;
    private ParticleSystem coreParticles;
    private Material coreGlowBlobMat;
    private Transform coreGlowBlob;
    private readonly List<Transform> ringPivots = new List<Transform>();
    private readonly List<float> ringSpinSpeeds = new List<float>();
    private Light coreLight;
    private Texture2D radialGradientTex;

    void Awake()
    {
        root = transform;
        radialGradientTex = CreateRadialGradientTexture(64);

        BuildCore();
        BuildRings();
        BuildCoreLight();

        root.localScale = Vector3.one * overallScale;
    }

    void Update()
    {
        float pulse = 1f + Mathf.Sin(Time.time * pulseSpeed) * pulseAmount;

        if (coreGlowBlobMat != null)
        {
            coreGlowBlobMat.color = glowColor * emissionIntensity * 0.5f * pulse;
        }
        if (coreGlowBlob != null)
        {
            coreGlowBlob.localScale = Vector3.one * coreRadius * coreGlowBlobSize * pulse;
            coreGlowBlob.rotation = Quaternion.LookRotation(
                coreGlowBlob.position - (Camera.main != null ? Camera.main.transform.position : coreGlowBlob.position + Vector3.forward));
        }
        if (coreLight != null)
        {
            coreLight.intensity = emissionIntensity * 0.5f * pulse;
        }

        for (int i = 0; i < ringPivots.Count; i++)
        {
            ringPivots[i].Rotate(Vector3.up, ringSpinSpeeds[i] * Time.deltaTime, Space.Self);
        }
    }

    // ---------------- CORE ----------------

    void BuildCore()
    {
        // Soft radial glow blob behind the particle shell — this is what reads as
        // "glow" even with no Bloom post-processing.
        GameObject blobGO = new GameObject("CoreGlowBlob");
        blobGO.transform.SetParent(root, false);
        MeshFilter mf = blobGO.AddComponent<MeshFilter>();
        mf.mesh = CreateQuadMesh();
        MeshRenderer mr = blobGO.AddComponent<MeshRenderer>();
        coreGlowBlobMat = CreateAdditiveMaterial(glowColor, emissionIntensity * 0.5f, radialGradientTex);
        mr.material = coreGlowBlobMat;
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        mr.receiveShadows = false;
        coreGlowBlob = blobGO.transform;
        coreGlowBlob.localScale = Vector3.one * coreRadius * coreGlowBlobSize;

        // Dense particle shell = the speckled sphere surface in the reference image
        GameObject psGO = new GameObject("CoreParticles");
        psGO.transform.SetParent(root, false);
        coreParticles = psGO.AddComponent<ParticleSystem>();

        var main = coreParticles.main;
        main.loop = true;
        main.startLifetime = Mathf.Infinity;
        main.startSpeed = 0f;
        main.startSize = new ParticleSystem.MinMaxCurve(coreParticleSize * 0.5f, coreParticleSize * 1.5f);
        main.startColor = glowColor;
        main.maxParticles = coreParticleCount;
        main.simulationSpace = ParticleSystemSimulationSpace.Local;

        var emission = coreParticles.emission;
        emission.rateOverTime = 0f;

        var shape = coreParticles.shape;
        shape.shapeType = ParticleSystemShapeType.Sphere;
        shape.radius = coreRadius;
        shape.radiusThickness = coreShellThickness; // 0 = shell only, matches speckled sphere look

        // Subtle shimmer so the core doesn't look static
        var noise = coreParticles.noise;
        noise.enabled = true;
        noise.strength = coreRadius * 0.15f;
        noise.frequency = 0.4f;
        noise.scrollSpeed = 0.2f;

        var renderer = coreParticles.GetComponent<ParticleSystemRenderer>();
        renderer.material = CreateAdditiveMaterial(glowColor, emissionIntensity);
        renderer.renderMode = ParticleSystemRenderMode.Billboard;

        coreParticles.Emit(coreParticleCount);
    }

    void BuildCoreLight()
    {
        GameObject lightGO = new GameObject("CoreLight");
        lightGO.transform.SetParent(root, false);
        coreLight = lightGO.AddComponent<Light>();
        coreLight.type = LightType.Point;
        coreLight.color = glowColor;
        coreLight.range = (maxRingRadius + 0.5f) * 3f;
        coreLight.intensity = emissionIntensity * 0.5f;
    }

    // ---------------- RINGS ----------------

    void BuildRings()
    {
        for (int i = 0; i < ringCount; i++)
        {
            float t = ringCount > 1 ? (float)i / (ringCount - 1) : 0f;
            float radius = Mathf.Lerp(minRingRadius, maxRingRadius, t);

            GameObject pivot = new GameObject($"RingPivot_{i}");
            pivot.transform.SetParent(root, false);

            if (useClassicAtomAngles)
            {
                float zAngle = i * (180f / Mathf.Max(ringCount, 1));
                pivot.transform.localRotation = Quaternion.Euler(classicTiltDegrees, 0f, zAngle);
            }
            else
            {
                pivot.transform.localRotation = Quaternion.Euler(
                    Random.Range(-40f, 40f), Random.Range(0f, 360f), Random.Range(-40f, 40f));
            }

            if (showFaintArcLine)
            {
                BuildRingArcLine(pivot.transform, radius);
            }
            if (dustParticlesPerRing > 0)
            {
                BuildOrbitingParticles(pivot.transform, radius, dustParticlesPerRing,
                    dustParticleSize, dustOrbitalSpeedRange, emissionIntensity * 0.6f, "Dust");
            }
            if (nodesPerRing > 0)
            {
                BuildOrbitingParticles(pivot.transform, radius, nodesPerRing,
                    nodeParticleSize, nodeOrbitalSpeedRange, emissionIntensity * 1.2f, "Nodes");
            }

            ringPivots.Add(pivot.transform);
            float spin = Random.Range(ringPivotSpinRange.x, ringPivotSpinRange.y);
            if (Random.value < 0.5f) spin *= -1f;
            ringSpinSpeeds.Add(spin);
        }
    }

    void BuildRingArcLine(Transform parent, float radius)
    {
        GameObject lineGO = new GameObject("ArcLine");
        lineGO.transform.SetParent(parent, false);
        LineRenderer lr = lineGO.AddComponent<LineRenderer>();
        lr.useWorldSpace = false;
        lr.loop = true;
        int segments = 80;
        lr.positionCount = segments;
        lr.widthMultiplier = arcLineWidth;
        lr.material = CreateAdditiveMaterial(glowColor, emissionIntensity * 0.35f);
        lr.startColor = new Color(glowColor.r, glowColor.g, glowColor.b, 0.5f);
        lr.endColor = lr.startColor;
        lr.numCapVertices = 2;

        for (int s = 0; s < segments; s++)
        {
            float angle = (float)s / segments * Mathf.PI * 2f;
            lr.SetPosition(s, new Vector3(Mathf.Cos(angle) * radius, 0f, Mathf.Sin(angle) * radius));
        }
    }

    void BuildOrbitingParticles(Transform parent, float radius, int count, float size,
        Vector2 speedRange, float intensity, string label)
    {
        GameObject psGO = new GameObject($"Ring{label}");
        psGO.transform.SetParent(parent, false);

        ParticleSystem ps = psGO.AddComponent<ParticleSystem>();
        var main = ps.main;
        main.loop = true;
        main.startLifetime = Mathf.Infinity;
        main.startSpeed = 0f;
        main.startSize = new ParticleSystem.MinMaxCurve(size * 0.7f, size * 1.3f);
        main.startColor = glowColor;
        main.maxParticles = Mathf.Max(2, count);
        main.simulationSpace = ParticleSystemSimulationSpace.Local;

        var emission = ps.emission;
        emission.rateOverTime = 0f;

        var shape = ps.shape;
        shape.shapeType = ParticleSystemShapeType.Circle;
        shape.radius = radius;
        shape.arc = 360f;
        shape.radiusThickness = 0f; // sit exactly on the ring path

        var velocity = ps.velocityOverLifetime;
        velocity.enabled = true;
        float orbitalSpeed = Random.Range(speedRange.x, speedRange.y);
        if (Random.value < 0.5f) orbitalSpeed *= -1f;
        velocity.orbitalY = orbitalSpeed; // moves particles continuously around the true circle

        var renderer = ps.GetComponent<ParticleSystemRenderer>();
        renderer.material = CreateAdditiveMaterial(glowColor, intensity);
        renderer.renderMode = ParticleSystemRenderMode.Billboard;

        ps.Emit(count);
    }

    // ---------------- HELPERS ----------------

    Mesh CreateQuadMesh()
    {
        Mesh mesh = new Mesh();
        mesh.vertices = new Vector3[]
        {
            new Vector3(-0.5f, -0.5f, 0f),
            new Vector3(0.5f, -0.5f, 0f),
            new Vector3(0.5f, 0.5f, 0f),
            new Vector3(-0.5f, 0.5f, 0f),
        };
        mesh.uv = new Vector2[] { new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1), new Vector2(0, 1) };
        mesh.triangles = new int[] { 0, 2, 1, 0, 3, 2 };
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    Texture2D CreateRadialGradientTexture(int resolution)
    {
        Texture2D tex = new Texture2D(resolution, resolution, TextureFormat.RGBA32, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        Vector2 center = new Vector2(resolution * 0.5f, resolution * 0.5f);
        float maxDist = resolution * 0.5f;

        for (int y = 0; y < resolution; y++)
        {
            for (int x = 0; x < resolution; x++)
            {
                float dist = Vector2.Distance(new Vector2(x, y), center) / maxDist;
                float alpha = Mathf.Clamp01(1f - dist);
                alpha = Mathf.Pow(alpha, 2f); // soften falloff
                tex.SetPixel(x, y, new Color(1f, 1f, 1f, alpha));
            }
        }
        tex.Apply();
        return tex;
    }

    Material CreateAdditiveMaterial(Color color, float intensity, Texture2D texture = null)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Particles/Unlit")
                      ?? Shader.Find("Particles/Standard Unlit")
                      ?? Shader.Find("Legacy Shaders/Particles/Additive")
                      ?? Shader.Find("Sprites/Default");

        Material mat = new Material(shader);
        Color hdrColor = color * intensity;
        hdrColor.a = 1f;

        // Try common color property names across pipelines
        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", hdrColor);
        if (mat.HasProperty("_Color")) mat.SetColor("_Color", hdrColor);
        if (mat.HasProperty("_TintColor")) mat.SetColor("_TintColor", hdrColor);

        if (texture != null)
        {
            if (mat.HasProperty("_BaseMap")) mat.SetTexture("_BaseMap", texture);
            if (mat.HasProperty("_MainTex")) mat.SetTexture("_MainTex", texture);
        }

        // Force additive/transparent blending where the shader exposes those controls
        if (mat.HasProperty("_Surface")) mat.SetFloat("_Surface", 1f); // 0 = Opaque, 1 = Transparent (URP)
        if (mat.HasProperty("_Blend")) mat.SetFloat("_Blend", 1f);     // URP: 1 = Additive
        if (mat.HasProperty("_BlendOp")) mat.SetFloat("_BlendOp", (float)UnityEngine.Rendering.BlendOp.Add);
        if (mat.HasProperty("_SrcBlend")) mat.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
        if (mat.HasProperty("_DstBlend")) mat.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.One);
        if (mat.HasProperty("_ZWrite")) mat.SetFloat("_ZWrite", 0f);

        mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        mat.EnableKeyword("_ALPHAPREMULTIPLY_ON");
        return mat;
    }
}