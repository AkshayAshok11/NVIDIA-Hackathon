using System.Collections;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Orchestrates a fixed show/hide timeline across existing scene GameObjects,
/// then spawns a glowing orange ring on the floor in front of the XR rig once
/// the person taps Yes or No.
///
/// FLOW:
///   1. Everything starts hidden.
///   2. dialogueObject becomes visible.
///   3. Wait 10s.
///   4. particlesObject hides.
///   5. dialogueObject hides.
///   6. Wait 1s.
///   7. thumbnailObject becomes visible.
///   8. Wait 3s.
///   9. yesButton / noButton become visible.
///  10. On either click: hide both buttons, spawn a glowing orange ring on the
///      floor in front of the rig.
///
/// Attach to an empty GameObject, drag your existing scene objects into the
/// fields below, and press Play.
/// </summary>
[DisallowMultipleComponent]
public class MemoryFlowSequencer : MonoBehaviour
{
    [Header("Scene References (existing GameObjects)")]
    public GameObject dialogueObject;
    public GameObject particlesObject;
    public GameObject thumbnailObject;
    public GameObject yesButtonObject;
    public GameObject noButtonObject;

    [Header("Timing")]
    public float dialogueVisibleDuration = 10f;
    public float pauseAfterDialogue = 4f;
    public float pauseBeforeButtons = 3f;

    [Header("XR Rig / Floor Ring")]
    [Tooltip("The player rig / XR Origin. Falls back to Camera.main if left empty.")]
    public Transform rig;
    public float ringDistanceFromRig = 1.5f;
    public float ringOuterRadius = 0.5f;
    public float ringInnerRadius = 0.38f;
    [Tooltip("How far to raycast downward from the rig to find the floor. Set to 0 to just use the rig's Y position.")]
    public float floorRaycastHeight = 2f;
    public LayerMask floorLayerMask = ~0;
    public Color ringColor = new Color(1f, 0.55f, 0.1f, 1f); // glowing orange
    [Range(1f, 10f)] public float ringIntensity = 6f;
    public GameObject ringPrefab; // optional prefab to use instead of procedural ring

    [Header("Behavior")]
    public bool autoStartOnPlay = true;

    private bool choiceMade;

    public GameObject aiDesign;

    void Start()
    {
        HideAll();
        if (autoStartOnPlay) StartCoroutine(RunFlow());
    }

    public void StartFlow()
    {
        HideAll();
        StartCoroutine(RunFlow());
    }

    void HideAll()
    {
        SetActive(dialogueObject, false);
        SetActive(particlesObject, false);
        SetActive(thumbnailObject, false);
        SetActive(yesButtonObject, false);
        SetActive(noButtonObject, false);
        SetActive(ringPrefab, false);
        SetActive(aiDesign, true);
    }

    IEnumerator RunFlow()
    {
        yield return new WaitForSeconds(4.0f);
        SetActive(dialogueObject, true);
        yield return new WaitForSeconds(dialogueVisibleDuration);

        SetActive(particlesObject, false);
        SetActive(dialogueObject, false);
        SetActive(aiDesign, false);
        yield return new WaitForSeconds(pauseAfterDialogue);

        SetActive(thumbnailObject, true);
        yield return new WaitForSeconds(pauseBeforeButtons);

        choiceMade = false;
        HookButton(yesButtonObject);
        HookButton(noButtonObject);
        SetActive(yesButtonObject, true);
        SetActive(noButtonObject, true);

        yield return new WaitForSeconds(5.0f);

        SetActive(yesButtonObject, false);
        SetActive(noButtonObject, false);

        SetActive(ringPrefab, true);
    }

    void HookButton(GameObject buttonObj)
    {
        if (buttonObj == null) return;
        Button btn = buttonObj.GetComponent<Button>();
        if (btn == null)
        {
            Debug.LogWarning($"MemoryFlowSequencer: '{buttonObj.name}' has no Button component to click.");
            return;
        }
        btn.onClick.RemoveListener(OnChoiceButtonClicked);
        btn.onClick.AddListener(OnChoiceButtonClicked);
    }

    void OnChoiceButtonClicked()
    {
        choiceMade = true;
    }

    void SetActive(GameObject go, bool active)
    {
        if (go != null) go.SetActive(active);
    }

    // ---------------- GLOWING FLOOR RING ----------------

    float FindFloorHeight(Vector3 atXZ, float fallbackY)
    {
        if (floorRaycastHeight <= 0f) return fallbackY;

        Vector3 rayStart = new Vector3(atXZ.x, fallbackY + floorRaycastHeight, atXZ.z);
        if (Physics.Raycast(rayStart, Vector3.down, out RaycastHit hit, floorRaycastHeight * 2f, floorLayerMask))
        {
            return hit.point.y;
        }
        return fallbackY;
    }

    // ---------------- PROCEDURAL GRAPHICS ----------------

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
}