using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using TMPro; // Requires TextMeshPro (Window > TextMeshPro > Import TMP Essential Resources if prompted)

/// <summary>
/// Modern / minimalist / futuristic HUD dialogue sequence, world-space anchored
/// to the rig's starting position (captured once, at Start).
///
/// Flow:
///   1. The user's line types out one word at a time, centered in a rounded pill.
///   2. The pill fades out; a memory card fades in: a horizontal thumbnail
///      (fully rounded corners) with a separate rounded caption panel beneath it.
///   3. Below the card: "Yes" (white text, blue rounded button) and
///      "No" (white text, gray rounded button).
///   4. On Yes, everything fades out and a glowing circle appears in front of the camera.
///
/// Attach to an empty GameObject and press Play — everything is built at runtime.
/// Assign `rig` to your XR Origin / player root (falls back to Camera.main).
/// </summary>
[DisallowMultipleComponent]
public class MemoryDialogueSequence : MonoBehaviour
{
    [Header("World Placement (relative to rig's start pose)")]
    public Transform rig;
    public float forwardDistance = 2.2f;
    [Tooltip("Vertical offset from the rig's start height. Negative moves the panel down.")]
    public float heightOffset = -0.15f;
    [Tooltip("World units per UI unit. Smaller = physically smaller HUD.")]
    public float canvasScale = 0.0016f;

    [Header("Dialogue Content")]
    [TextArea] public string userLine = "I want to revisit my grandma's garden.";
    public string captionTitle = "Grandma's Garden";
    public Sprite thumbnailSprite;

    [Header("Typing Speed")]
    public float wordTypeInterval = 0.18f;
    public float pauseBeforeConfirm = 0.5f;

    [Header("Voice Bar Style (rounded pill, centered text)")]
    public float voiceBarWidth = 820f;
    public float voiceBarHeight = 96f;
    public Color userAccentColor = new Color(0.35f, 0.75f, 1f, 1f);
    public Color voiceBarFill = new Color(0.04f, 0.06f, 0.09f, 0.75f);

    [Header("Memory Card Style (horizontal thumbnail + caption panel)")]
    public float cardWidth = 480f;
    public float cardImageHeight = 260f;
    public float cardCaptionHeight = 100f;
    public float panelGap = 14f;
    public float cardCornerRadius = 24f;
    public float cardBorderThickness = 2f;
    public float cardGlowRange = 20f;
    public Color aiAccentColor = new Color(0.95f, 0.72f, 0.35f, 1f);
    public Color cardBackgroundFill = new Color(0.06f, 0.05f, 0.03f, 0.85f);
    public Color captionBackgroundFill = new Color(0.05f, 0.04f, 0.03f, 0.92f);

    [Header("Buttons")]
    public Color yesButtonColor = new Color(0.25f, 0.55f, 0.95f, 1f);
    public Color noButtonColor = new Color(0.4f, 0.4f, 0.42f, 1f);
    public Color buttonTextColor = Color.white;
    public float buttonWidth = 140f;
    public float buttonHeight = 52f;
    public float buttonGapBelowCard = 30f;

    [Header("Typography")]
    public Color primaryTextColor = new Color(0.95f, 0.97f, 1f, 1f);
    public Color mutedTextColor = new Color(0.7f, 0.72f, 0.75f, 0.85f);

    [Header("Glowing Circle (spawned after Yes)")]
    public float circleDistanceFromCamera = 2.5f;
    public float circleWorldRadius = 0.4f;
    public Color circleColor = new Color(0.4f, 0.85f, 1f);
    [Range(1f, 10f)] public float circleIntensity = 5f;

    [Header("Behavior")]
    public bool autoStartOnPlay = true;
    public Action onYes;
    public Action onNo;

    private CanvasGroup voiceBarGroup;
    private CanvasGroup cardGroup;
    private TMP_Text userText;
    private bool yesClicked;
    private bool noClicked;

    void Start()
    {
        Transform rigT = rig != null ? rig : (Camera.main != null ? Camera.main.transform : transform);
        Vector3 startPos = rigT.position;
        Vector3 flatForward = rigT.forward;
        flatForward.y = 0f;
        if (flatForward.sqrMagnitude < 0.0001f) flatForward = Vector3.forward;
        flatForward.Normalize();

        Vector3 canvasPos = startPos + flatForward * forwardDistance + Vector3.up * heightOffset;
        Quaternion canvasRot = Quaternion.LookRotation(flatForward, Vector3.up);

        BuildCanvas(canvasPos, canvasRot);
        EnsureEventSystem();

        if (autoStartOnPlay) StartCoroutine(RunSequence());
    }

    public void StartSequence() => StartCoroutine(RunSequence());

    IEnumerator RunSequence()
    {
        voiceBarGroup.gameObject.SetActive(true);
        yield return StartCoroutine(FadeCanvasGroup(voiceBarGroup, 0f, 1f, 0.3f));

        userText.text = "";
        yield return StartCoroutine(TypeWordByWord(userText, userLine, wordTypeInterval));

        yield return new WaitForSeconds(pauseBeforeConfirm);

        yield return StartCoroutine(FadeCanvasGroup(voiceBarGroup, 1f, 0f, 0.3f));
        voiceBarGroup.gameObject.SetActive(false);

        cardGroup.gameObject.SetActive(true);
        yield return StartCoroutine(FadeCanvasGroup(cardGroup, 0f, 1f, 0.35f));

        yesClicked = false;
        noClicked = false;
        yield return new WaitUntil(() => yesClicked || noClicked);

        yield return StartCoroutine(FadeCanvasGroup(cardGroup, 1f, 0f, 0.3f));
        cardGroup.gameObject.SetActive(false);

        if (yesClicked)
        {
            onYes?.Invoke();
            SpawnGlowingCircle();
        }
        else
        {
            onNo?.Invoke();
        }
    }

    IEnumerator TypeWordByWord(TMP_Text target, string content, float interval)
    {
        target.text = "";
        string[] words = content.Split(' ');
        for (int i = 0; i < words.Length; i++)
        {
            target.text += (i == 0 ? "" : " ") + words[i];
            yield return new WaitForSeconds(interval);
        }
    }

    IEnumerator FadeCanvasGroup(CanvasGroup cg, float from, float to, float duration)
    {
        float t = 0f;
        cg.alpha = from;
        while (t < duration)
        {
            t += Time.deltaTime;
            cg.alpha = Mathf.Lerp(from, to, t / duration);
            yield return null;
        }
        cg.alpha = to;
    }

    void BuildCanvas(Vector3 worldPos, Quaternion worldRot)
    {
        GameObject canvasGO = new GameObject("MemoryDialogueCanvas_World");
        Canvas canvas = canvasGO.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.worldCamera = Camera.main;
        canvasGO.AddComponent<GraphicRaycaster>();

        RectTransform canvasRt = canvasGO.GetComponent<RectTransform>();
        canvasRt.sizeDelta = new Vector2(1200f, 700f);
        canvasGO.transform.position = worldPos;
        canvasGO.transform.rotation = worldRot;
        canvasGO.transform.localScale = Vector3.one * canvasScale;

        BuildVoiceBar(canvasGO.transform);
        BuildMemoryCard(canvasGO.transform);
    }

    void BuildVoiceBar(Transform parent)
    {
        GameObject bar = new GameObject("VoiceBar");
        bar.transform.SetParent(parent, false);
        RectTransform rt = bar.AddComponent<RectTransform>();
        rt.sizeDelta = new Vector2(voiceBarWidth, voiceBarHeight);
        rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 0.5f);
        rt.anchoredPosition = Vector2.zero;

        Image bg = bar.AddComponent<Image>();
        bg.sprite = CreateRoundedRectSprite(
            Mathf.RoundToInt(voiceBarWidth * 0.5f), Mathf.RoundToInt(voiceBarHeight * 0.5f),
            voiceBarHeight / 2f, 1.5f, 14f, voiceBarFill,
            new Color(userAccentColor.r, userAccentColor.g, userAccentColor.b, 0.6f));

        voiceBarGroup = bar.AddComponent<CanvasGroup>();
        voiceBarGroup.alpha = 0f;

        userText = CreateLabel(bar.transform, "UserText", primaryTextColor, 28, FontStyles.Normal,
            TextAlignmentOptions.Center, Vector2.zero, new Vector2(voiceBarWidth - 80f, voiceBarHeight - 20f));

        bar.SetActive(false);
    }

    void BuildMemoryCard(Transform parent)
    {
        float stackHeight = cardImageHeight + panelGap + cardCaptionHeight;
        float buttonsBlockHeight = buttonGapBelowCard + buttonHeight;
        float totalHeight = stackHeight + buttonsBlockHeight;

        GameObject root = new GameObject("MemoryCardRoot");
        root.transform.SetParent(parent, false);
        RectTransform rootRt = root.AddComponent<RectTransform>();
        rootRt.sizeDelta = new Vector2(cardWidth, totalHeight);
        rootRt.anchorMin = rootRt.anchorMax = new Vector2(0.5f, 0.5f);
        rootRt.anchoredPosition = Vector2.zero;

        cardGroup = root.AddComponent<CanvasGroup>();
        cardGroup.alpha = 0f;

        BuildMaskedRoundedImage(root.transform,
            thumbnailSprite != null ? thumbnailSprite : CreatePlaceholderThumb(),
            new Vector2(cardWidth, cardImageHeight),
            new Vector2(0f, -buttonsBlockHeight),
            cardCornerRadius, aiAccentColor, cardBorderThickness, cardGlowRange);

        GameObject captionPanel = new GameObject("CaptionPanel");
        captionPanel.transform.SetParent(root.transform, false);
        Image captionImg = captionPanel.AddComponent<Image>();
        captionImg.sprite = CreateRoundedRectSprite(
            Mathf.RoundToInt(cardWidth * 0.6f), Mathf.RoundToInt(cardCaptionHeight * 0.6f),
            cardCornerRadius, cardBorderThickness, cardGlowRange, captionBackgroundFill, aiAccentColor);
        RectTransform captionRt = captionPanel.GetComponent<RectTransform>();
        captionRt.sizeDelta = new Vector2(cardWidth, cardCaptionHeight);
        captionRt.anchorMin = captionRt.anchorMax = new Vector2(0.5f, 1f);
        captionRt.pivot = new Vector2(0.5f, 1f);
        captionRt.anchoredPosition = new Vector2(0f, -buttonsBlockHeight - cardImageHeight - panelGap);

        TMP_Text title = CreateLabel(captionPanel.transform, "CaptionTitle", primaryTextColor, 26, FontStyles.Bold,
            TextAlignmentOptions.TopLeft, new Vector2(24f, -16f), new Vector2(cardWidth - 48f, 34f));
        title.text = captionTitle;

        GameObject yesBtn = CreatePillButton(root.transform, "YesButton", "Yes", yesButtonColor,
            new Vector2(-(buttonWidth / 2f + 12f), buttonHeight / 2f), new Vector2(buttonWidth, buttonHeight));
        GameObject noBtn = CreatePillButton(root.transform, "NoButton", "No", noButtonColor,
            new Vector2(buttonWidth / 2f + 12f, buttonHeight / 2f), new Vector2(buttonWidth, buttonHeight));

        SetAnchorBottom(yesBtn.GetComponent<RectTransform>(), new Vector2(-(buttonWidth / 2f + 12f), 0f));
        SetAnchorBottom(noBtn.GetComponent<RectTransform>(), new Vector2(buttonWidth / 2f + 12f, 0f));

        yesBtn.GetComponent<Button>().onClick.AddListener(() => yesClicked = true);
        noBtn.GetComponent<Button>().onClick.AddListener(() => noClicked = true);

        root.SetActive(false);
    }

    void SetAnchorBottom(RectTransform rt, Vector2 anchoredPos)
    {
        rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 0f);
        rt.pivot = new Vector2(0.5f, 0f);
        rt.anchoredPosition = anchoredPos;
    }

    void BuildMaskedRoundedImage(Transform parent, Sprite sprite, Vector2 size, Vector2 anchoredPos,
        float radius, Color borderColor, float borderThickness, float glowRange)
    {
        GameObject borderGO = new GameObject("PhotoBorder");
        borderGO.transform.SetParent(parent, false);
        Image borderImg = borderGO.AddComponent<Image>();
        borderImg.sprite = CreateRoundedRectSprite(
            Mathf.RoundToInt(size.x * 0.5f), Mathf.RoundToInt(size.y * 0.5f),
            radius, borderThickness, glowRange, new Color(0f, 0f, 0f, 0f), borderColor);
        RectTransform borderRt = borderGO.GetComponent<RectTransform>();
        borderRt.sizeDelta = size;
        borderRt.anchorMin = borderRt.anchorMax = new Vector2(0.5f, 1f);
        borderRt.pivot = new Vector2(0.5f, 1f);
        borderRt.anchoredPosition = anchoredPos;

        GameObject maskGO = new GameObject("PhotoMask");
        maskGO.transform.SetParent(borderGO.transform, false);
        Image maskImg = maskGO.AddComponent<Image>();
        maskImg.sprite = CreateRoundedRectSprite(
            Mathf.RoundToInt(size.x * 0.5f), Mathf.RoundToInt(size.y * 0.5f),
            radius, 0f, 0f, Color.white, Color.white);
        maskImg.color = Color.white;
        Mask mask = maskGO.AddComponent<Mask>();
        mask.showMaskGraphic = false;
        RectTransform maskRt = maskGO.GetComponent<RectTransform>();
        maskRt.anchorMin = Vector2.zero;
        maskRt.anchorMax = Vector2.one;
        maskRt.offsetMin = Vector2.zero;
        maskRt.offsetMax = Vector2.zero;

        GameObject photo = new GameObject("Photo");
        photo.transform.SetParent(maskGO.transform, false);
        Image photoImg = photo.AddComponent<Image>();
        photoImg.sprite = sprite;
        photoImg.type = Image.Type.Simple;
        photoImg.preserveAspect = false;
        RectTransform photoRt = photo.GetComponent<RectTransform>();
        photoRt.anchorMin = Vector2.zero;
        photoRt.anchorMax = Vector2.one;
        photoRt.offsetMin = Vector2.zero;
        photoRt.offsetMax = Vector2.zero;
    }

    TMP_Text CreateLabel(Transform parent, string name, Color color, float fontSize, FontStyles style,
        TextAlignmentOptions alignment, Vector2 anchoredPos, Vector2 size)
    {
        GameObject go = new GameObject(name);
        go.transform.SetParent(parent, false);
        TextMeshProUGUI txt = go.AddComponent<TextMeshProUGUI>();
        txt.text = "";
        txt.fontSize = fontSize;
        txt.fontStyle = style;
        txt.color = color;
        txt.alignment = alignment;
        txt.enableWordWrapping = true;
        txt.characterSpacing = 1.5f;

        RectTransform rt = go.GetComponent<RectTransform>();
        rt.sizeDelta = size;
        rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 0.5f);
        rt.pivot = new Vector2(0.5f, 0.5f);
        rt.anchoredPosition = anchoredPos;

        return txt;
    }

    GameObject CreatePillButton(Transform parent, string name, string label, Color fillColor,
        Vector2 anchoredPos, Vector2 size)
    {
        GameObject btnGO = new GameObject(name);
        btnGO.transform.SetParent(parent, false);
        RectTransform rt = btnGO.AddComponent<RectTransform>();
        rt.sizeDelta = size;
        rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 0.5f);
        rt.pivot = new Vector2(0.5f, 0.5f);
        rt.anchoredPosition = anchoredPos;

        Image fill = btnGO.AddComponent<Image>();
        fill.sprite = CreateRoundedRectSprite(
            Mathf.RoundToInt(size.x * 0.5f), Mathf.RoundToInt(size.y * 0.5f),
            size.y / 2f, 0f, 6f, fillColor, fillColor);

        Button btn = btnGO.AddComponent<Button>();
        btn.transition = Selectable.Transition.None;

        GameObject overlayGO = new GameObject("HoverOverlay");
        overlayGO.transform.SetParent(btnGO.transform, false);
        Image overlay = overlayGO.AddComponent<Image>();
        overlay.sprite = CreateRoundedRectSprite(
            Mathf.RoundToInt(size.x * 0.5f), Mathf.RoundToInt(size.y * 0.5f),
            size.y / 2f, 0f, 0f, Color.white, Color.white);
        Color oc = Color.white; oc.a = 0f;
        overlay.color = oc;
        RectTransform overlayRt = overlayGO.GetComponent<RectTransform>();
        overlayRt.anchorMin = Vector2.zero; overlayRt.anchorMax = Vector2.one;
        overlayRt.offsetMin = Vector2.zero; overlayRt.offsetMax = Vector2.zero;

        HoverGlow hover = btnGO.AddComponent<HoverGlow>();
        hover.overlay = overlay;
        hover.targetAlpha = 0.18f;

        TMP_Text labelTxt = CreateLabel(btnGO.transform, "Label", buttonTextColor, 22, FontStyles.Bold,
            TextAlignmentOptions.Center, Vector2.zero, size);
        labelTxt.text = label;
        labelTxt.characterSpacing = 2f;

        return btnGO;
    }

    void EnsureEventSystem()
    {
        if (FindObjectOfType<EventSystem>() == null)
        {
            GameObject es = new GameObject("EventSystem");
            es.AddComponent<EventSystem>();
            es.AddComponent<StandaloneInputModule>();
        }
    }

    Sprite CreateRoundedRectSprite(int texW, int texH, float radius, float borderThickness, float glowRange,
        Color fill, Color border)
    {
        texW = Mathf.Max(texW, 8);
        texH = Mathf.Max(texH, 8);
        Texture2D tex = new Texture2D(texW, texH, TextureFormat.RGBA32, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;

        Vector2 half = new Vector2(texW / 2f, texH / 2f);
        float r = Mathf.Min(radius, Mathf.Min(half.x, half.y));

        for (int y = 0; y < texH; y++)
        {
            for (int x = 0; x < texW; x++)
            {
                Vector2 p = new Vector2(x - half.x, y - half.y);
                Vector2 q = new Vector2(Mathf.Abs(p.x) - half.x + r, Mathf.Abs(p.y) - half.y + r);
                float outside = Vector2.Max(q, Vector2.zero).magnitude;
                float inside = Mathf.Min(Mathf.Max(q.x, q.y), 0f);
                float d = outside + inside - r;

                Color c;
                if (d < -borderThickness) c = fill;
                else if (d < 0f)
                {
                    float t = Mathf.InverseLerp(-borderThickness, 0f, d);
                    c = Color.Lerp(fill, border, t);
                }
                else if (glowRange > 0f && d < glowRange)
                {
                    float t = d / glowRange;
                    float a = border.a * (1f - t) * (1f - t);
                    c = border; c.a = a;
                }
                else c = new Color(0f, 0f, 0f, 0f);

                tex.SetPixel(x, y, c);
            }
        }
        tex.Apply();
        return Sprite.Create(tex, new Rect(0, 0, texW, texH), new Vector2(0.5f, 0.5f));
    }

    Sprite CreatePlaceholderThumb()
    {
        int w = 256, h = 144;
        Texture2D tex = new Texture2D(w, h);
        Color a = new Color(0.16f, 0.12f, 0.07f);
        Color b = new Color(0.08f, 0.06f, 0.04f);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                tex.SetPixel(x, y, Color.Lerp(a, b, (float)y / h));
        tex.Apply();
        return Sprite.Create(tex, new Rect(0, 0, w, h), new Vector2(0.5f, 0.5f));
    }

    private class HoverGlow : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler
    {
        public Image overlay;
        public float targetAlpha = 0.2f;
        private Coroutine fadeRoutine;

        public void OnPointerEnter(PointerEventData eventData) => Restart(targetAlpha);
        public void OnPointerExit(PointerEventData eventData) => Restart(0f);

        void Restart(float to)
        {
            if (fadeRoutine != null) StopCoroutine(fadeRoutine);
            fadeRoutine = StartCoroutine(Fade(to));
        }

        IEnumerator Fade(float to)
        {
            float start = overlay.color.a;
            float t = 0f;
            const float duration = 0.15f;
            while (t < duration)
            {
                t += Time.deltaTime;
                Color c = overlay.color;
                c.a = Mathf.Lerp(start, to, t / duration);
                overlay.color = c;
                yield return null;
            }
            Color cc = overlay.color; cc.a = to; overlay.color = cc;
        }
    }

    void SpawnGlowingCircle()
    {
        Camera cam = Camera.main;
        Vector3 position = cam != null
            ? cam.transform.position + cam.transform.forward * circleDistanceFromCamera
            : transform.position + Vector3.forward * circleDistanceFromCamera;

        GameObject circleGO = new GameObject("GlowingCircle");
        circleGO.transform.position = position;
        if (cam != null)
            circleGO.transform.rotation = Quaternion.LookRotation(circleGO.transform.position - cam.transform.position);
        circleGO.transform.localScale = Vector3.one * circleWorldRadius * 2f;

        MeshFilter mf = circleGO.AddComponent<MeshFilter>();
        mf.mesh = CreateQuadMesh();
        MeshRenderer mr = circleGO.AddComponent<MeshRenderer>();
        Material mat = CreateAdditiveGlowMaterial(circleColor, circleIntensity, CreateRadialGradientTexture(64));
        mr.material = mat;
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        mr.receiveShadows = false;

        circleGO.AddComponent<GlowPulse>().Init(mat, circleColor, circleIntensity, circleGO.transform, cam);
    }

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
                float alpha = Mathf.Pow(Mathf.Clamp01(1f - dist), 2f);
                tex.SetPixel(x, y, new Color(1f, 1f, 1f, alpha));
            }
        }
        tex.Apply();
        return tex;
    }

    Material CreateAdditiveGlowMaterial(Color color, float intensity, Texture2D texture)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Particles/Unlit")
                      ?? Shader.Find("Particles/Standard Unlit")
                      ?? Shader.Find("Legacy Shaders/Particles/Additive")
                      ?? Shader.Find("Sprites/Default");

        Material mat = new Material(shader);
        Color hdrColor = color * intensity;
        hdrColor.a = 1f;

        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", hdrColor);
        if (mat.HasProperty("_Color")) mat.SetColor("_Color", hdrColor);
        if (mat.HasProperty("_TintColor")) mat.SetColor("_TintColor", hdrColor);
        if (mat.HasProperty("_BaseMap")) mat.SetTexture("_BaseMap", texture);
        if (mat.HasProperty("_MainTex")) mat.SetTexture("_MainTex", texture);

        if (mat.HasProperty("_Surface")) mat.SetFloat("_Surface", 1f);
        if (mat.HasProperty("_Blend")) mat.SetFloat("_Blend", 1f);
        if (mat.HasProperty("_SrcBlend")) mat.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
        if (mat.HasProperty("_DstBlend")) mat.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.One);
        if (mat.HasProperty("_ZWrite")) mat.SetFloat("_ZWrite", 0f);

        mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        return mat;
    }

    private class GlowPulse : MonoBehaviour
    {
        private Material mat;
        private Color baseColor;
        private float baseIntensity;
        private Transform target;
        private Camera cam;

        public void Init(Material material, Color color, float intensity, Transform t, Camera camera)
        {
            mat = material; baseColor = color; baseIntensity = intensity; target = t; cam = camera;
        }

        void Update()
        {
            if (mat == null) return;
            float pulse = 1f + Mathf.Sin(Time.time * 1.4f) * 0.25f;
            Color c = baseColor * baseIntensity * pulse;
            c.a = 1f;
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", c);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", c);

            if (cam != null && target != null)
                target.rotation = Quaternion.LookRotation(target.position - cam.transform.position);
        }
    }
}