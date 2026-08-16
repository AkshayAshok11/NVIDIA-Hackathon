using System.Collections;
using UnityEngine;

// Drives the AI assistant flow off real microphone presence instead of a
// scripted timer: Idle while quiet, Listening the moment you speak. Once you
// go quiet again for a bit, it treats that as "done talking" and plays the
// Responding -> memory card -> portal beat (still placeholder timing/content
// there, since a real reply needs actual speech-to-text + an AI response,
// not just mic loudness). transcript is intentionally not shown yet - wire
// transcript.ShowUser(...)/ShowAI(...) up to real STT output when that's in.
public class TimeboxSequenceController : MonoBehaviour
{
    [Header("References")]
    public AISpark spark;
    public TranscriptPanel transcript;
    public MemoryCard memoryCard;
    public PortalRing portalRing;
    public MemorySpaceReveal memorySpaceReveal;

    [Header("Voice Detection")]
    public float speakingThreshold = 0.08f;
    public float silenceToRespondDelay = 1.2f;

    [Header("Testing")]
    [Tooltip("Uncheck to stay in the Idle/Listening loop indefinitely - useful for testing mic reactivity without advancing to the memory card/portal.")]
    public bool advanceToMemoryCard = true;

    [Header("Timing (seconds) - placeholder until real AI response exists")]
    public float respondingDuration = 3f;
    public float memoryDuration = 3.5f;

    bool _isListening;
    float _silenceTimer;
    bool _responding;

    void Start()
    {
        transcript.Hide();
        memoryCard.Hide();
        portalRing.gameObject.SetActive(false);
        portalRing.OnPlayerEntered += HandlePlayerEntered;

        spark.SetState(AISpark.SparkState.Idle);
    }

    void Update()
    {
        if (_responding) return;

        float level = MicrophoneInputLevel.Instance != null ? MicrophoneInputLevel.Instance.Level : 0f;
        bool speaking = level > speakingThreshold;

        if (speaking)
        {
            _silenceTimer = 0f;
            if (!_isListening)
            {
                _isListening = true;
                spark.SetState(AISpark.SparkState.Listening);
            }
        }
        else if (_isListening)
        {
            _silenceTimer += Time.deltaTime;
            if (_silenceTimer >= silenceToRespondDelay)
            {
                _isListening = false;

                if (advanceToMemoryCard)
                    StartCoroutine(RunResponseSequence());
                else
                    spark.SetState(AISpark.SparkState.Idle);
            }
        }
    }

    IEnumerator RunResponseSequence()
    {
        _responding = true;

        spark.SetState(AISpark.SparkState.Responding);
        yield return new WaitForSeconds(respondingDuration);

        transcript.Hide();
        spark.SetVisible(false);
        memoryCard.Show();
        yield return new WaitForSeconds(memoryDuration);

        memoryCard.Hide();
        portalRing.gameObject.SetActive(true);

        _responding = false;
    }

    void HandlePlayerEntered()
    {
        portalRing.gameObject.SetActive(false);
        spark.SetVisible(false);

        if (memorySpaceReveal != null)
            memorySpaceReveal.Reveal();
    }
}
