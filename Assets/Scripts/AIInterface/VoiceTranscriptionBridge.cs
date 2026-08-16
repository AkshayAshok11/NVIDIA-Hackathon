using UnityEngine;
using Meta.WitAi;

// Bridges Meta Voice SDK (Wit.ai) transcription events into the existing
// TranscriptPanel UI. Requires a configured AppVoiceExperience component in
// the scene - set one up via Meta > Voice SDK > Settings first (creates/links
// a WitConfiguration asset holding your Client Access Token), then drag that
// AppVoiceExperience into this component's Voice Service field.
//
// This runs independently of AISpark/MicrophoneInputLevel's volume-based
// reactivity and Idle/Listening state switching - those keep driving the
// visuals off raw mic loudness, while this drives the actual displayed text
// off Wit.ai's real speech recognition.
public class VoiceTranscriptionBridge : MonoBehaviour
{
    [Header("References")]
    public VoiceService voiceService;
    public TranscriptPanel transcript;

    [Header("Behavior")]
    [Tooltip("Automatically start a new listening activation after each one completes, so it keeps transcribing continuously.")]
    public bool continuousListening = true;

    void Reset()
    {
        if (voiceService == null)
            voiceService = Object.FindFirstObjectByType<VoiceService>();
    }

    void OnEnable()
    {
        if (voiceService == null)
        {
            Debug.LogWarning("VoiceTranscriptionBridge: no VoiceService assigned.");
            return;
        }

        voiceService.VoiceEvents.OnPartialTranscription.AddListener(HandlePartialTranscription);
        voiceService.VoiceEvents.OnFullTranscription.AddListener(HandleFullTranscription);
        voiceService.VoiceEvents.OnRequestCompleted.AddListener(HandleRequestCompleted);
        voiceService.VoiceEvents.OnError.AddListener(HandleError);

        voiceService.Activate();
    }

    void OnDisable()
    {
        if (voiceService == null) return;

        voiceService.VoiceEvents.OnPartialTranscription.RemoveListener(HandlePartialTranscription);
        voiceService.VoiceEvents.OnFullTranscription.RemoveListener(HandleFullTranscription);
        voiceService.VoiceEvents.OnRequestCompleted.RemoveListener(HandleRequestCompleted);
        voiceService.VoiceEvents.OnError.RemoveListener(HandleError);
    }

    void HandlePartialTranscription(string text)
    {
        if (string.IsNullOrEmpty(text)) return;
        transcript.ShowUser(text);
    }

    void HandleFullTranscription(string text)
    {
        if (!string.IsNullOrEmpty(text))
            transcript.ShowUser(text);
    }

    void HandleRequestCompleted()
    {
        if (continuousListening)
            voiceService.Activate();
    }

    void HandleError(string error, string message)
    {
        Debug.LogWarning($"VoiceTranscriptionBridge: Wit.ai error ({error}): {message}");
        if (continuousListening)
            voiceService.Activate();
    }
}
