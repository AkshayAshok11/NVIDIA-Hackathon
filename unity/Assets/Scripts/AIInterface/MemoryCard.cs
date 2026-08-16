using UnityEngine;
using UnityEngine.UI;
using TMPro;

// Floating photo card shown after the AI "finds" a memory. The photo is a
// procedurally generated placeholder gradient - swap BuildPlaceholderSprite's
// result for a real texture once actual memory photos are wired up.
public class MemoryCard : MonoBehaviour
{
    [Header("Mock Data")]
    public string memoryTitle = "My 18th Birthday";
    public string memoryDate = "May 14, 2019";

    [Header("Layout")]
    public Vector2 cardSize = new Vector2(700f, 500f);
    public float worldScale = 0.0022f;

    void Awake()
    {
        Build();
        Hide();
    }

    public void Show() => gameObject.SetActive(true);
    public void Hide() => gameObject.SetActive(false);

    void Build()
    {
        var canvasObj = new GameObject("Canvas");
        canvasObj.transform.SetParent(transform, false);
        var canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        var rect = canvasObj.GetComponent<RectTransform>();
        rect.sizeDelta = cardSize;
        canvasObj.transform.localScale = Vector3.one * worldScale;
        canvasObj.AddComponent<CanvasRenderer>();

        var frameObj = new GameObject("Frame");
        frameObj.transform.SetParent(canvasObj.transform, false);
        var frame = frameObj.AddComponent<Image>();
        frame.color = new Color(1f, 0.7f, 0.25f);
        var frameRect = frameObj.GetComponent<RectTransform>();
        frameRect.anchorMin = Vector2.zero;
        frameRect.anchorMax = Vector2.one;
        frameRect.offsetMin = Vector2.zero;
        frameRect.offsetMax = Vector2.zero;

        var photoObj = new GameObject("Photo");
        photoObj.transform.SetParent(canvasObj.transform, false);
        var photo = photoObj.AddComponent<Image>();
        photo.sprite = BuildPlaceholderSprite();
        var photoRect = photoObj.GetComponent<RectTransform>();
        photoRect.anchorMin = Vector2.zero;
        photoRect.anchorMax = Vector2.one;
        photoRect.offsetMin = new Vector2(10f, 90f);
        photoRect.offsetMax = new Vector2(-10f, -10f);

        var titleObj = new GameObject("Title");
        titleObj.transform.SetParent(canvasObj.transform, false);
        var title = titleObj.AddComponent<TextMeshProUGUI>();
        title.text = memoryTitle;
        title.fontSize = 34f;
        title.fontStyle = FontStyles.Bold;
        title.color = Color.white;
        title.alignment = TextAlignmentOptions.BottomLeft;
        var titleRect = titleObj.GetComponent<RectTransform>();
        titleRect.anchorMin = new Vector2(0f, 0f);
        titleRect.anchorMax = new Vector2(1f, 0f);
        titleRect.offsetMin = new Vector2(20f, 40f);
        titleRect.offsetMax = new Vector2(-20f, 90f);

        var dateObj = new GameObject("Date");
        dateObj.transform.SetParent(canvasObj.transform, false);
        var date = dateObj.AddComponent<TextMeshProUGUI>();
        date.text = memoryDate;
        date.fontSize = 24f;
        date.color = new Color(0.85f, 0.85f, 0.85f);
        date.alignment = TextAlignmentOptions.TopLeft;
        var dateRect = dateObj.GetComponent<RectTransform>();
        dateRect.anchorMin = new Vector2(0f, 0f);
        dateRect.anchorMax = new Vector2(1f, 0f);
        dateRect.offsetMin = new Vector2(20f, 8f);
        dateRect.offsetMax = new Vector2(-20f, 38f);
    }

    static Sprite BuildPlaceholderSprite()
    {
        const int size = 256;
        var tex = new Texture2D(size, size);
        var topColor = new Color(0.9f, 0.55f, 0.2f);
        var bottomColor = new Color(0.25f, 0.1f, 0.15f);

        for (int y = 0; y < size; y++)
        {
            var rowColor = Color.Lerp(bottomColor, topColor, (float)y / size);
            for (int x = 0; x < size; x++)
                tex.SetPixel(x, y, rowColor);
        }
        tex.Apply();

        return Sprite.Create(tex, new Rect(0f, 0f, size, size), new Vector2(0.5f, 0.5f));
    }
}
