using UnityEngine;
using UnityEngine.UI;
using TMPro;

// Floating world-space caption bubble for user speech / AI response text.
// Reused for both speakers - call ShowUser/ShowAI to swap accent color and text.
public class TranscriptPanel : MonoBehaviour
{
    [Header("Layout")]
    public Vector2 panelSize = new Vector2(900f, 220f);
    public float worldScale = 0.0022f;

    [Header("Colors")]
    public Color userAccent = new Color(0.3f, 0.75f, 1f);
    public Color aiAccent = new Color(1f, 0.65f, 0.2f);

    Image _iconDot;
    TextMeshProUGUI _label;

    void Awake()
    {
        Build();
        Hide();
    }

    public void ShowUser(string text) => Show(text, userAccent);
    public void ShowAI(string text) => Show(text, aiAccent);

    public void Show(string text, Color accent)
    {
        gameObject.SetActive(true);
        _label.text = text;
        _iconDot.color = accent;
    }

    public void Hide()
    {
        gameObject.SetActive(false);
    }

    void Build()
    {
        var canvasObj = new GameObject("Canvas");
        canvasObj.transform.SetParent(transform, false);
        var canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        var rect = canvasObj.GetComponent<RectTransform>();
        rect.sizeDelta = panelSize;
        canvasObj.transform.localScale = Vector3.one * worldScale;
        canvasObj.transform.localPosition = Vector3.zero;
        canvasObj.transform.localRotation = Quaternion.identity;
        canvasObj.AddComponent<CanvasRenderer>();

        var bgObj = new GameObject("Background");
        bgObj.transform.SetParent(canvasObj.transform, false);
        var background = bgObj.AddComponent<Image>();
        background.color = new Color(0.03f, 0.03f, 0.05f, 0.85f);
        var bgRect = bgObj.GetComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.offsetMin = Vector2.zero;
        bgRect.offsetMax = Vector2.zero;

        var iconObj = new GameObject("Icon");
        iconObj.transform.SetParent(canvasObj.transform, false);
        _iconDot = iconObj.AddComponent<Image>();
        var iconRect = iconObj.GetComponent<RectTransform>();
        iconRect.anchorMin = new Vector2(0f, 0.5f);
        iconRect.anchorMax = new Vector2(0f, 0.5f);
        iconRect.pivot = new Vector2(0f, 0.5f);
        iconRect.sizeDelta = new Vector2(50f, 50f);
        iconRect.anchoredPosition = new Vector2(40f, 0f);

        var textObj = new GameObject("Label");
        textObj.transform.SetParent(canvasObj.transform, false);
        _label = textObj.AddComponent<TextMeshProUGUI>();
        _label.fontSize = 42f;
        _label.color = Color.white;
        _label.alignment = TextAlignmentOptions.MidlineLeft;
        _label.enableWordWrapping = true;
        var textRect = textObj.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = new Vector2(120f, 20f);
        textRect.offsetMax = new Vector2(-30f, -20f);
    }
}
