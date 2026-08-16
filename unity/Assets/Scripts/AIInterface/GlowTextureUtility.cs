using UnityEngine;
using UnityEngine.Rendering;

// Shared helper for the soft additive "glow" look used by the AI spark and
// the portal ring. Avoids relying on HDR/Bloom post-processing (expensive on
// Quest) - instead bakes a radial falloff into a small texture and renders it
// with additive blending, which is cheap and works identically in every build.
public static class GlowTextureUtility
{
    public static Texture2D BuildRadialGradient(Color color, int size = 128)
    {
        var tex = new Texture2D(size, size, TextureFormat.RGBA32, false)
        {
            wrapMode = TextureWrapMode.Clamp
        };

        var center = new Vector2(size / 2f, size / 2f);
        float maxDist = size / 2f;

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float dist = Vector2.Distance(new Vector2(x, y), center) / maxDist;
                float falloff = Mathf.Clamp01(1f - dist);
                falloff *= falloff;
                tex.SetPixel(x, y, color * falloff);
            }
        }
        tex.Apply();
        return tex;
    }

    // Remaps an existing texture's brightness through a new tint color,
    // e.g. to recolor an imported asset's baked-in color at runtime.
    public static Texture2D Recolor(Texture2D source, Color tint)
    {
        var pixels = source.GetPixels();
        var output = new Color[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            float luminance = Mathf.Max(pixels[i].r, Mathf.Max(pixels[i].g, pixels[i].b));
            var c = tint * luminance;
            c.a = pixels[i].a;
            output[i] = c;
        }

        var result = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false)
        {
            wrapMode = source.wrapMode,
            filterMode = source.filterMode
        };
        result.SetPixels(output);
        result.Apply();
        return result;
    }

    public static Material BuildAdditiveMaterial(Texture2D gradientTexture)
    {
        var mat = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        mat.SetTexture("_BaseMap", gradientTexture);
        mat.SetColor("_BaseColor", Color.white);
        mat.SetFloat("_Cull", 0f);
        mat.SetFloat("_ZWrite", 0f);
        mat.SetFloat("_SrcBlend", (float)BlendMode.One);
        mat.SetFloat("_DstBlend", (float)BlendMode.One);
        mat.renderQueue = (int)RenderQueue.Transparent;
        return mat;
    }
}
