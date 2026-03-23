# DiffusionDesk Test Plan

This document defines the test images, prompts, and configurations for evaluating DiffusionDesk's three core features.

---

## 1. Inpainting (`test/diffusion/`)

| #   | File                  | Description                  | Mask Region                 | Prompt                               | Model             |
| --- | --------------------- | ---------------------------- | --------------------------- | ------------------------------------ | ----------------- |
| 1   | `group_photo.jpg`     | Group of students walking    | Mask one person on the edge | "empty sidewalk near brick building" | SD Inpainting     |
| 2   | `old_vintage_car.jpg` | Vintage car on a street      | Mask the car                | "empty cobblestone road"             | SDXL Inpainting   |
| 3   | `mountain_lake.jpg`   | Snowy mountain range         | Mask a mountain peak        | "clear blue sky with clouds"         | FLUX.1 Fill (NF4) |
| 4   | `beach.jpg`           | Crowded beach with people    | Mask a group of people      | "empty sandy beach"                  | SD Inpainting     |
| 5   | `coastal.jpg`         | Coastal cliffside with ocean | Mask the cliff/rocks        | "calm ocean water"                   | SDXL Inpainting   |

### Inpainting Parameters

| Parameter       | Default                          | Notes                                    |
| --------------- | -------------------------------- | ---------------------------------------- |
| Guidance Scale  | 7.5                              | Higher = more prompt adherence           |
| Inference Steps | 30                               | Balance between quality and speed        |
| Strength        | 1.0                              | Full inpainting for object removal       |
| Negative Prompt | "blurry, low quality, distorted" | Consistent across tests                  |
| Seed            | `null` (random)                  | Set to a fixed value for reproducibility |

---

## 2. Style Transfer (`test/style_transfer/`)

| #   | File                 | Description                      | Style Preset    | Custom Prompt (if any) |
| --- | -------------------- | -------------------------------- | --------------- | ---------------------- |
| 1   | `landscape.jpg`      | Waterfall with rocks and forest  | `oil_painting`  | —                      |
| 2   | `portrait.jpg`       | Woman with flowers in a field    | `anime`         | —                      |
| 3   | `architecture.jpg`   | European canal with boats        | `impressionist` | —                      |
| 4   | `nature.jpg`         | Forest with ocean panoramic view | `watercolor`    | —                      |
| 5   | `misty_mountain.jpg` | Misty mountain reflected in lake | `ghibli`        | —                      |

### Additional Style Variations (optional)

Each image can be tested with multiple styles to compare output diversity:

| File                 | Alt Style 1  | Alt Style 2     |
| -------------------- | ------------ | --------------- |
| `landscape.jpg`      | `watercolor` | `sketch`        |
| `portrait.jpg`       | `ghibli`     | `pop_art`       |
| `architecture.jpg`   | `sketch`     | `cyberpunk`     |
| `nature.jpg`         | `pixel_art`  | `oil_painting`  |
| `misty_mountain.jpg` | `watercolor` | `impressionist` |

### Style Transfer Parameters

| Parameter       | Default                          | Notes                                                                                        |
| --------------- | -------------------------------- | -------------------------------------------------------------------------------------------- |
| Strength        | 0.6                              | Controls structure vs style trade-off (0.2–0.4 subtle, 0.5–0.7 moderate, 0.8–1.0 aggressive) |
| Guidance Scale  | 7.5                              | Higher = more prompt adherence                                                               |
| Inference Steps | 30                               | Balance between quality and speed                                                            |
| Model           | `sdxl-img2img`                   | SDXL for best quality; `sd-img2img` as fallback                                              |
| Negative Prompt | "blurry, low quality, distorted" | Consistent across tests                                                                      |

---

## 3. Restoration (`test/restoration/`)

| #   | File                   | Description                            | Face Model | Upscale | Scratch Removal |
| --- | ---------------------- | -------------------------------------- | ---------- | ------- | --------------- |
| 1   | `degraded_face_01.png` | Low-quality cropped child face         | CodeFormer | 2x      | No              |
| 2   | `degraded_face_02.jpg` | Old photo of two children (full image) | CodeFormer | 2x      | No              |
| 3   | `degraded_face_03.png` | Old B&W cropped child face             | GFPGAN     | 2x      | No              |
| 4   | `degraded_face_04.png` | Low-quality cropped toddler face       | CodeFormer | 4x      | No              |
| 5   | `degraded_face_05.jpg` | Side-by-side child/adult comparison    | GFPGAN     | 2x      | No              |

### Restoration Parameters

| Parameter       | Default      | Notes                                                               |
| --------------- | ------------ | ------------------------------------------------------------------- |
| Face Model      | `codeformer` | CodeFormer generally produces better results; GFPGAN as alternative |
| Fidelity        | 0.5          | CodeFormer only — 0 = max enhancement, 1 = max fidelity to original |
| Upscale         | `2x`         | Real-ESRGAN; `4x` for higher resolution output                      |
| Scratch Removal | `false`      | Enable for visibly scratched/damaged photos                         |
| Colorize        | `false`      | Enable for B&W photos (e.g., `degraded_face_03.png`)                |

### Additional Restoration Variations (optional)

| File                   | Variation                  | Notes                              |
| ---------------------- | -------------------------- | ---------------------------------- |
| `degraded_face_01.png` | Fidelity 0.0 vs 0.5 vs 1.0 | Compare quality–fidelity trade-off |
| `degraded_face_03.png` | Enable colorize            | Test B&W → color on grayscale face |
| `degraded_face_02.jpg` | CodeFormer vs GFPGAN       | Compare face model outputs         |
| `degraded_face_04.png` | 2x vs 4x upscale           | Compare upscale quality            |

---

## Image Sources

- Inpainting & Style Transfer images: [Picsum Photos / Unsplash](https://picsum.photos) and [Pexels](https://www.pexels.com) (free license)
- Restoration images: [TencentARC/GFPGAN test inputs](https://github.com/TencentARC/GFPGAN/tree/master/inputs) (open source)
