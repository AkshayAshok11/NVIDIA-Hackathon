using System;
using UnityEngine;

// Glowing light-blue ring laid flat on the floor, with a soft additive glow
// pooled underneath it. Invites the player to walk into it; fires
// OnPlayerEntered once when the camera gets within triggerRadius.
[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class PortalRing : MonoBehaviour
{
    public float outerRadius = 0.9f;
    public float innerRadius = 0.75f;
    public int segments = 64;
    public Color ringColor = new Color(0.45f, 0.8f, 1f);
    public float triggerRadius = 1f;

    [Header("Glow")]
    public float glowRadiusMultiplier = 1.6f;
    public float glowIntensity = 0.8f;

    public event Action OnPlayerEntered;

    bool _triggered;
    Material _material;
    Transform _player;
    Transform _glow;

    void Awake()
    {
        BuildRingMesh();
        _material = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        _material.SetFloat("_Cull", 0f);
        _material.SetColor("_BaseColor", ringColor);
        GetComponent<MeshRenderer>().material = _material;

        BuildGlow();
    }

    void OnEnable()
    {
        _triggered = false;
    }

    void Update()
    {
        float pulse = (Mathf.Sin(Time.time * 2f) + 1f) * 0.5f;
        _material.SetColor("_BaseColor", ringColor + Color.white * pulse * 0.3f);
        if (_glow != null)
            _glow.localScale = Vector3.one * (outerRadius * 2f * glowRadiusMultiplier * (1f + pulse * 0.15f));

        if (_triggered) return;

        if (_player == null)
        {
            if (Camera.main == null) return;
            _player = Camera.main.transform;
        }

        Vector3 toPlayer = _player.position - transform.position;
        toPlayer.y = 0f;
        if (toPlayer.magnitude <= triggerRadius)
        {
            _triggered = true;
            OnPlayerEntered?.Invoke();
        }
    }

    void BuildGlow()
    {
        var glowObj = GameObject.CreatePrimitive(PrimitiveType.Quad);
        glowObj.name = "Glow";
        glowObj.transform.SetParent(transform, false);
        glowObj.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
        glowObj.transform.localPosition = new Vector3(0f, -0.001f, 0f);
        glowObj.transform.localScale = Vector3.one * (outerRadius * 2f * glowRadiusMultiplier);
        Destroy(glowObj.GetComponent<Collider>());
        _glow = glowObj.transform;

        var texture = GlowTextureUtility.BuildRadialGradient(ringColor * glowIntensity);
        var material = GlowTextureUtility.BuildAdditiveMaterial(texture);
        glowObj.GetComponent<MeshRenderer>().material = material;
    }

    void BuildRingMesh()
    {
        var mesh = new Mesh { name = "Portal Ring" };
        var vertices = new Vector3[segments * 2];
        var triangles = new int[segments * 6];

        for (int i = 0; i < segments; i++)
        {
            float angle = (float)i / segments * Mathf.PI * 2f;
            float cos = Mathf.Cos(angle);
            float sin = Mathf.Sin(angle);
            vertices[i * 2] = new Vector3(cos * innerRadius, 0f, sin * innerRadius);
            vertices[i * 2 + 1] = new Vector3(cos * outerRadius, 0f, sin * outerRadius);
        }

        for (int i = 0; i < segments; i++)
        {
            int next = (i + 1) % segments;
            int a = i * 2;
            int b = i * 2 + 1;
            int c = next * 2;
            int d = next * 2 + 1;

            triangles[i * 6] = a;
            triangles[i * 6 + 1] = b;
            triangles[i * 6 + 2] = c;

            triangles[i * 6 + 3] = b;
            triangles[i * 6 + 4] = d;
            triangles[i * 6 + 5] = c;
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        GetComponent<MeshFilter>().mesh = mesh;
    }
}
