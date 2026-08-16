using UnityEngine;
using UnityEngine.Rendering;

// Sets the dark void skybox/ambient and paints a procedural grid texture onto
// the floor - matches the "dark room, grid floor" look from the storyboard
// without needing any imported texture assets.
public class EnvironmentSetup : MonoBehaviour
{
    [Header("References")]
    public MeshRenderer floorRenderer;

    [Header("Sky")]
    public Color skyColor = new Color(0.02f, 0.02f, 0.03f);

    [Header("Floor Grid")]
    public Color floorBaseColor = new Color(0.05f, 0.05f, 0.06f);
    public Color gridLineColor = new Color(0.35f, 0.38f, 0.42f);
    public int textureSize = 256;
    public int cellsPerAxis = 8;
    public float tiling = 20f;

    void Awake()
    {
        var skyMat = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        skyMat.SetColor("_BaseColor", skyColor);
        RenderSettings.skybox = skyMat;

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.08f, 0.08f, 0.1f);

        if (floorRenderer != null)
            BuildFloorGrid();
    }

    void BuildFloorGrid()
    {
        var tex = new Texture2D(textureSize, textureSize) { wrapMode = TextureWrapMode.Repeat };
        int cell = textureSize / cellsPerAxis;

        for (int y = 0; y < textureSize; y++)
        {
            for (int x = 0; x < textureSize; x++)
            {
                bool line = x % cell == 0 || y % cell == 0;
                tex.SetPixel(x, y, line ? gridLineColor : floorBaseColor);
            }
        }
        tex.Apply();

        var mat = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        mat.SetTexture("_BaseMap", tex);
        mat.SetTextureScale("_BaseMap", new Vector2(tiling, tiling));
        mat.SetColor("_BaseColor", Color.white);

        floorRenderer.material = mat;
    }
}
