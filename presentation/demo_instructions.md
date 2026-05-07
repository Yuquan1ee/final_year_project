# DiffusionDesk — Live Demo Instructions

Pre-loaded test images and exact settings for the 3-feature live demo (~5 min).

---

## Pre-demo checklist

- [ ] Backend running on Colab T4 with **SD 1.5 Inpainting**, **SDXL img2img**, and **CodeFormer + Real-ESRGAN** pre-warmed
- [ ] ngrok tunnel live; frontend `.env` set to ngrok URL
- [ ] Sanity-run each of the three flows end-to-end at least once before walking in
- [ ] Backup screen recording ready as a fallback on Slide 16
- [ ] **Avoid demoing FLUX.1 Fill live** — too slow + VRAM-fragile

---

## 1. Inpainting Demo — Object Removal (~2.5 min)

**Image:** `demo_inpainting_boat.jpg`
A solitary white rowboat on a glassy lake at dusk, with a dark reflection in the water and a soft pastel pink/lavender sky. Single, well-defined object — ideal for clean object-removal showcase.

### Settings
| Field | Value |
|---|---|
| Model | **SD Inpainting** *(fastest, most reliable)* |
| Prompt | `calm empty lake water at sunset, smooth reflection, peaceful` |
| Negative prompt | `boat, vessel, object, ripples, distortion, blurry, low quality` |
| Guidance scale | `7.5` |
| Steps | `30` |
| Strength | `1.0` |

### Area to shade (mask)

Cover the **boat and its full reflection** in the water — both elements need to disappear together for a believable result.

```
+--------------------------------------+
|         pastel sky / clouds          |   (do NOT mask)
|                                      |
|                                      |
|  ~~~~~~~~~~~~~ horizon ~~~~~~~~~~~~  |   (do NOT mask)
|                                      |
|                  [BOAT]              |   ← MASK (the white hull)
|                  [refl]              |   ← MASK (the dark reflection
|                                      |       directly below)
|                                      |
|         shallow water / sand         |   (do NOT mask)
+--------------------------------------+
```

- Use a **medium brush** (~30-50 px).
- Mask **both the boat and the dark reflection** below it. If you only mask the boat, the reflection becomes a ghost giveaway.
- Add a small margin (5-10 px) around the boat — diffusion blends better with a slightly generous mask.
- Do **not** mask the horizon line, sky, or shallow foreground water — those give the model the colour and texture cues it needs to fill convincingly.

### What to say while it generates (~10 s)
> "The request goes from React to FastAPI as a base64 JSON payload. The backend decodes it, runs a 30-step DDIM denoising loop on the masked latent, and returns the result. This is a classic object-removal scenario — replacing the masked region with content that's consistent with the surrounding water and sky."

### Optional follow-up
If time permits, switch model dropdown to **SDXL Inpainting (4-bit)** and regenerate — show the quality difference (+15 s).

---

## 2. Style Transfer Demo (~1.5 min)

**Image:** `demo_style_mountain.jpg`
Dramatic snow-capped mountain peaks emerging from heavy fog. Mostly monochrome — Cyberpunk style will inject vivid neon colours and a futuristic feel while preserving the mountain silhouette.

### Settings
| Field | Value |
|---|---|
| Model | **SDXL img2img (4-bit)** |
| Preset | **Cyberpunk** |
| Strength | `0.4` *(lowered from 0.6 default — see note below)* |
| Guidance scale | `5.0` *(lowered from 7.5 default — see note below)* |
| Steps | `30` |

### Why these non-default values

The Cyberpunk preset's prompt is semantically very far from "foggy mountain landscape", so at the default strength of 0.6 + guidance 7.5 the model has enough freedom to reinterpret the whole scene as a cityscape — losing the mountain composition.

- **Strength 0.4** keeps the encoded latent only partially noised, so denoising stays anchored to the original mountain silhouette. The cyberpunk colour grade and neon highlights still come through.
- **Guidance scale 5.0** softens how aggressively the model follows the style prompt, so it leans more on the input image.

If the mountain still gets lost, drop strength further to `0.35`. If the style is too subtle, raise to `0.45`. **Avoid going above 0.55 with this preset on this image** — composition collapses.

### What to say
> "img2img reuses the diffusion process — instead of pure noise, we encode the image, partially noise it via the strength parameter, then denoise with the style prompt. I am using a lower strength of 0.4 here because the cyberpunk style is semantically far from a mountain landscape — at the default 0.6 the model would reinterpret the scene too aggressively. With strength 0.4, the mountain silhouette is preserved while the cyberpunk colour palette still comes through clearly."

### Optional follow-up
If time permits, regenerate at strength `0.6` to show how the same preset becomes more aggressive — the composition starts dissolving into a cyberpunk cityscape interpretation (+15 s). This is a useful live illustration of the strength parameter's role and trade-off.

---

## 3. Restoration Demo (~1.5 min)

**Image:** `demo_restoration_portrait.jpg`
Vintage sepia portrait of two women — visible scratches, faded edges, soft focus. Genuinely aged photograph with two clearly detectable faces, ideal for showing multi-face restoration.

### Settings
| Field | Value |
|---|---|
| Face model | **CodeFormer** |
| Fidelity (`w`) | `0.5` *(balanced quality vs. identity)* |
| Upscale | **2×** *(Real-ESRGAN)* |
| Background enhancement | On |

### What to say
> "Restoration uses three non-diffusion models because they're optimised for the inverse problem — recovering existing detail rather than generating new content. No prompt is needed. CodeFormer detects and enhances each face individually; Real-ESRGAN doubles the resolution of the background and re-pastes the enhanced faces."

### Talking point
This image has **two faces** — useful for highlighting that CodeFormer's face-detection step finds and enhances each one independently, which is one of the multi-face test cases I report in §7.

---

## Demo wrap

Close the browser, return to the slide deck, resume narrative on the inpainting results slide.

## Failure recovery

If any generation hangs >30 s past expected:
1. Cancel the request.
2. Switch to backup video on Slide 16.
3. Say: *"In the interest of time I'll show a pre-recorded run — the live system was working in rehearsal but Colab's GPU allocation can be variable."*
4. Never debug live. Move on.

---

## Image attribution

| File | Source | License |
|---|---|---|
| `demo_inpainting_boat.jpg` | [sunnykote on Pexels](https://www.pexels.com/photo/white-boat-on-shore-10770961/) | Pexels License (free use) |
| `demo_style_mountain.jpg` | [eberhardgross on Pexels](https://www.pexels.com/photo/landscape-photography-of-mountains-covered-in-snow-691668/) | Pexels License (free use) |
| `demo_restoration_portrait.jpg` | [Lisa (fotios-photos) on Pexels](https://www.pexels.com/photo/an-old-photograph-of-two-women-25792825/) | Pexels License (free use) |
