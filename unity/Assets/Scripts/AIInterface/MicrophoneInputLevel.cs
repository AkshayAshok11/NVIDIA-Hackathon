using UnityEngine;
#if PLATFORM_ANDROID
using UnityEngine.Android;
#endif

// Captures live microphone input and exposes a smoothed 0-1 volume level via
// MicrophoneInputLevel.Instance.Level. Drives audio-reactive visuals (e.g.
// AISpark growing louder when you speak). No transcription - just loudness.
public class MicrophoneInputLevel : MonoBehaviour
{
    public static MicrophoneInputLevel Instance { get; private set; }

    [Header("Mic")]
    public int sampleRate = 16000;
    public int clipLengthSeconds = 1;

    [Header("Level Smoothing")]
    public float smoothingSpeed = 8f;
    public float silenceThreshold = 0.01f;
    public float sensitivity = 8f;

    public float Level { get; private set; }

    AudioClip _clip;
    string _device;
    float[] _sampleBuffer;
    const int SampleWindow = 512;

    void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);

        _sampleBuffer = new float[SampleWindow];

#if PLATFORM_ANDROID
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
            Permission.RequestUserPermission(Permission.Microphone);
#endif

        StartMic();
    }

    void StartMic()
    {
        if (Microphone.devices.Length == 0)
        {
            Debug.LogWarning("MicrophoneInputLevel: no microphone devices found.");
            return;
        }

        _device = Microphone.devices[0];
        _clip = Microphone.Start(_device, true, clipLengthSeconds, sampleRate);
    }

    void Update()
    {
        float raw = 0f;

        if (_clip != null)
        {
            int micPosition = Microphone.GetPosition(_device);
            int start = micPosition - SampleWindow;
            if (start >= 0)
            {
                _clip.GetData(_sampleBuffer, start);

                float sum = 0f;
                for (int i = 0; i < SampleWindow; i++)
                    sum += _sampleBuffer[i] * _sampleBuffer[i];

                float rms = Mathf.Sqrt(sum / SampleWindow);
                raw = Mathf.Clamp01(rms * sensitivity);
                if (raw < silenceThreshold) raw = 0f;
            }
        }

        Level = Mathf.Lerp(Level, raw, Time.deltaTime * smoothingSpeed);
    }

    void OnDestroy()
    {
        if (_device != null && Microphone.IsRecording(_device))
            Microphone.End(_device);
    }
}
