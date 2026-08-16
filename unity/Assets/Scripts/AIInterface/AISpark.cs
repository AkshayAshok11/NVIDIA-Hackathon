using UnityEngine;

// AI assistant visual: the imported "Effect_08" particle effect (Creepy Cat
// 3D Effects pack) as the glowing core, plus procedural orbiting ember
// trails (real particles, not line strips) looping around it. Effect_08's
// scale breathes in sync with the trails' orbit pulse, with a per-axis
// wobble layered on top so its silhouette reads as an organic blob rather
// than a perfect circle. Effect_08 is a single persistent particle
// (maxNumParticles: 1) animated via a flipbook texture, not a swarm - so its
// state changes are conveyed through scale / animation speed / light
// intensity rather than emission rate. Recolors it from its native green to
// light blue at runtime since the color is baked into the texture.
public class AISpark : MonoBehaviour
{
    public enum SparkState { Idle, Listening, Responding }

    [Header("Source")]
    public string effectObjectName = "Effect_08";

    [Header("Recolor")]
    public Color tint = new Color(0.45f, 0.8f, 1f);

    [Header("Core State Tuning")]
    public float idleScale = 0.6f;
    public float listeningScale = 1f;
    public float respondingScale = 1.5f;
    public float idleSimSpeed = 0.5f;
    public float listeningSimSpeed = 1f;
    public float respondingSimSpeed = 1.8f;
    public float idleLightIntensity = 1f;
    public float listeningLightIntensity = 2f;
    public float respondingLightIntensity = 3.5f;

    [Header("Core Breathing / Imperfection")]
    public float pulseAmount = 0.15f;
    public float wobbleAmount = 0.12f;
    public float wobbleFrequencyX = 2.3f;
    public float wobbleFrequencyY = 3.1f;

    [Header("Trails (particles)")]
    public Color trailColor = new Color(0.35f, 0.7f, 1f);
    public int trailCount = 4;
    public int pointsPerTrail = 60;
    public float trailPointSpacing = 0.035f;
    public float trailParticleSize = 0.018f;

    [Header("Trail State Tuning")]
    public float idleTrailRadius = 0.1f;
    public float listeningTrailRadius = 0.18f;
    public float respondingTrailRadius = 0.28f;
    public float idleTrailSpeed = 0.6f;
    public float listeningTrailSpeed = 1.4f;
    public float respondingTrailSpeed = 2.4f;

    [Header("Audio Reactivity")]
    public bool audioReactive = true;
    public float audioScaleAmount = 0.7f;

    SparkState _state = SparkState.Idle;
    float _targetScale, _currentScale;
    float _targetSimSpeed, _currentSimSpeed;
    float _targetLightIntensity, _currentLightIntensity;
    float _targetTrailRadius, _currentTrailRadius;
    float _targetTrailSpeed, _currentTrailSpeed;
    float _trailTime;

    Transform _effectRoot;
    ParticleSystem _particles;
    Light _light;
    ParticleSystem[] _trails;
    ParticleSystem.Particle[][] _trailBuffers;
    ParticleSystem.Particle[] _coreParticleBuffer = new ParticleSystem.Particle[32];
    float _baseCoreStartSize = -1f;
    static readonly float[] FreqA = { 2f, 3f, 4f, 5f };
    static readonly float[] FreqB = { 3f, 2f, 5f, 4f };

    void Awake()
    {
        if (audioReactive && MicrophoneInputLevel.Instance == null)
            new GameObject("Microphone Input").AddComponent<MicrophoneInputLevel>();

        var effectGo = GameObject.Find(effectObjectName);
        if (effectGo == null)
        {
            Debug.LogWarning($"AISpark: could not find '{effectObjectName}' in the scene.");
        }
        else
        {
            _effectRoot = effectGo.transform;
            _particles = effectGo.GetComponentInChildren<ParticleSystem>();
            _light = effectGo.GetComponentInChildren<Light>();
            var renderer = effectGo.GetComponentInChildren<ParticleSystemRenderer>();

            Recolor(renderer);
            if (_light != null) _light.color = tint;
        }

        BuildTrails();
        ApplyState(true);
    }

    public void SetState(SparkState state)
    {
        _state = state;
        ApplyState(false);
    }

    public void SetVisible(bool visible)
    {
        if (_effectRoot != null)
            _effectRoot.gameObject.SetActive(visible);

        foreach (var trail in _trails)
            if (trail != null) trail.gameObject.SetActive(visible);
    }

    void ApplyState(bool immediate)
    {
        switch (_state)
        {
            case SparkState.Listening:
                _targetScale = listeningScale;
                _targetSimSpeed = listeningSimSpeed;
                _targetLightIntensity = listeningLightIntensity;
                _targetTrailRadius = listeningTrailRadius;
                _targetTrailSpeed = listeningTrailSpeed;
                break;
            case SparkState.Responding:
                _targetScale = respondingScale;
                _targetSimSpeed = respondingSimSpeed;
                _targetLightIntensity = respondingLightIntensity;
                _targetTrailRadius = respondingTrailRadius;
                _targetTrailSpeed = respondingTrailSpeed;
                break;
            default:
                _targetScale = idleScale;
                _targetSimSpeed = idleSimSpeed;
                _targetLightIntensity = idleLightIntensity;
                _targetTrailRadius = idleTrailRadius;
                _targetTrailSpeed = idleTrailSpeed;
                break;
        }

        if (immediate)
        {
            _currentScale = _targetScale;
            _currentSimSpeed = _targetSimSpeed;
            _currentLightIntensity = _targetLightIntensity;
            _currentTrailRadius = _targetTrailRadius;
            _currentTrailSpeed = _targetTrailSpeed;
        }
    }

    void Update()
    {
        _currentTrailRadius = Mathf.Lerp(_currentTrailRadius, _targetTrailRadius, Time.deltaTime * 2f);
        _currentTrailSpeed = Mathf.Lerp(_currentTrailSpeed, _targetTrailSpeed, Time.deltaTime * 2f);
        _trailTime += Time.deltaTime * _currentTrailSpeed;

        // Shared pulse - trail radius and core scale breathe together on the same clock.
        float pulse = (Mathf.Sin(_trailTime * 2f) + 1f) * 0.5f;

        // Live mic loudness grows the whole animation on top of the state/pulse sizing.
        float audioLevel = audioReactive && MicrophoneInputLevel.Instance != null
            ? MicrophoneInputLevel.Instance.Level
            : 0f;
        float audioBoost = 1f + audioLevel * audioScaleAmount;

        float pulsedTrailRadius = _currentTrailRadius * (1f + pulse * pulseAmount) * audioBoost;

        for (int i = 0; i < _trails.Length; i++)
            UpdateTrail(i, pulsedTrailRadius, audioBoost);

        if (_effectRoot == null) return;

        _currentScale = Mathf.Lerp(_currentScale, _targetScale, Time.deltaTime * 2f);
        _currentSimSpeed = Mathf.Lerp(_currentSimSpeed, _targetSimSpeed, Time.deltaTime * 2f);
        _currentLightIntensity = Mathf.Lerp(_currentLightIntensity, _targetLightIntensity, Time.deltaTime * 2f);

        float breathingScale = _currentScale * (1f + pulse * pulseAmount) * audioBoost;
        float wobbleX = 1f + Mathf.Sin(Time.time * wobbleFrequencyX) * wobbleAmount;
        float wobbleY = 1f + Mathf.Sin(Time.time * wobbleFrequencyY + 1.3f) * wobbleAmount;
        _effectRoot.localScale = new Vector3(breathingScale * wobbleX, breathingScale * wobbleY, breathingScale);

        if (_particles != null)
        {
            var main = _particles.main;
            main.simulationSpeed = _currentSimSpeed;

            // Effect_08's particle is long-lived (rarely respawns), so Start Size
            // module changes wouldn't retroactively resize it - write directly
            // into the live particle buffer instead, same technique as the trails.
            int count = _particles.GetParticles(_coreParticleBuffer);
            if (count > 0)
            {
                if (_baseCoreStartSize < 0f)
                    _baseCoreStartSize = _coreParticleBuffer[0].startSize;

                for (int i = 0; i < count; i++)
                    _coreParticleBuffer[i].startSize = _baseCoreStartSize * audioBoost;

                _particles.SetParticles(_coreParticleBuffer, count);
            }
        }

        if (_light != null)
            _light.intensity = _currentLightIntensity;
    }

    void UpdateTrail(int index, float radius, float audioBoost)
    {
        float fa = FreqA[index % FreqA.Length];
        float fb = FreqB[index % FreqB.Length];
        float phase = index * 1.7f;

        var buffer = _trailBuffers[index];
        for (int i = 0; i < pointsPerTrail; i++)
        {
            float t = _trailTime - i * trailPointSpacing;
            float x = Mathf.Cos(t * fa + phase) * radius;
            float y = Mathf.Sin(t * fb + phase) * radius;
            float z = Mathf.Sin(t * (fa + fb) * 0.5f + phase) * radius * 0.3f;

            float tailFactor = 1f - (float)i / (pointsPerTrail - 1);
            var color = trailColor;
            color.a = tailFactor;

            buffer[i].position = new Vector3(x, y, z);
            buffer[i].startColor = color;
            buffer[i].startSize = trailParticleSize * Mathf.Lerp(0.3f, 1f, tailFactor) * audioBoost;
            buffer[i].startLifetime = 10f;
            buffer[i].remainingLifetime = 10f;
            buffer[i].velocity = Vector3.zero;
            buffer[i].rotation = 0f;
        }

        _trails[index].SetParticles(buffer, pointsPerTrail);
    }

    void BuildTrails()
    {
        _trails = new ParticleSystem[trailCount];
        _trailBuffers = new ParticleSystem.Particle[trailCount][];
        var dotTexture = GlowTextureUtility.BuildRadialGradient(Color.white, 32);
        var material = GlowTextureUtility.BuildAdditiveMaterial(dotTexture);

        for (int i = 0; i < trailCount; i++)
        {
            var trailObj = new GameObject("Trail " + i);
            trailObj.transform.SetParent(transform, false);

            var ps = trailObj.AddComponent<ParticleSystem>();
            var main = ps.main;
            main.loop = true;
            main.playOnAwake = false;
            main.simulationSpace = ParticleSystemSimulationSpace.Local;
            main.maxParticles = pointsPerTrail;
            main.startLifetime = 10f;
            main.startSize = trailParticleSize;
            main.startColor = trailColor;
            main.startSpeed = 0f;

            var emission = ps.emission;
            emission.enabled = false;

            var shape = ps.shape;
            shape.enabled = false;

            var renderer = trailObj.GetComponent<ParticleSystemRenderer>();
            renderer.renderMode = ParticleSystemRenderMode.Billboard;
            renderer.material = material;

            ps.Play();

            _trails[i] = ps;
            _trailBuffers[i] = new ParticleSystem.Particle[pointsPerTrail];
        }
    }

    void Recolor(ParticleSystemRenderer renderer)
    {
        if (renderer == null) return;

        var mat = renderer.material;
        var source = mat.GetTexture("_BaseMap") as Texture2D;
        if (source == null) return;

        var recolored = GlowTextureUtility.Recolor(source, tint);
        mat.SetTexture("_BaseMap", recolored);
        mat.SetTexture("_EmissionMap", recolored);
        mat.SetColor("_EmissionColor", tint * 1.5f);
        mat.SetColor("_TintColor", Color.white);
    }
}
