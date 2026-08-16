using System.Collections;
using GaussianSplatting.Runtime;
using UnityEngine;

// Reveals the reconstructed memory scene (Gaussian splat + its floor) once
// the player steps through the portal ring. The splat is left active the
// whole time and faded via its opacity scale rather than toggled on/off, so
// its GPU buffers are already warmed up before the reveal and the fade
// itself doesn't cause a hitch. The floor's collider comes on at the start
// of the fade so there's never a moment where the player has nothing to
// stand on.
public class MemorySpaceReveal : MonoBehaviour
{
    public GaussianSplatRenderer gardenSplat;
    public GameObject floorPlane;
    public float fadeDuration = 2f;

    void Awake()
    {
        if (gardenSplat != null)
            gardenSplat.m_OpacityScale = 0f;
        if (floorPlane != null)
            floorPlane.SetActive(false);
    }

    public void Reveal()
    {
        StartCoroutine(RevealRoutine());
    }

    IEnumerator RevealRoutine()
    {
        if (floorPlane != null)
            floorPlane.SetActive(true);

        if (gardenSplat == null) yield break;

        float t = 0f;
        while (t < fadeDuration)
        {
            t += Time.deltaTime;
            gardenSplat.m_OpacityScale = Mathf.Clamp01(t / fadeDuration);
            yield return null;
        }
        gardenSplat.m_OpacityScale = 1f;
    }
}
