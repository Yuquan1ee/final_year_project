# Test Rerun Tasks

Re-run experiments with the **correct model** for each test case. Save results to `test/` directories, then copy to `report/images/`.

---

## Inpainting (`test/diffusion/`)

| # | Image | Mask | Prompt | Model | Save As |
|---|-------|------|--------|-------|---------|
| 1 | `beach.jpg` | `beach_mask.png` | "empty sandy beach" | **SD Inpainting** | `beach_result.png` |
| 2 | `old_vintage_car.jpg` | `old_vintage_car_mask.png` | "empty cobblestone road" | **SDXL Inpainting** | `old_vintage_car_result.png` |
| 3 | `mountain_lake.jpg` | `mountain_mask.png` | "clear blue sky with clouds" | **FLUX.1 Fill (NF4)** | `mountain_result.png` |
| 4 | `coastal.jpg` | `coastal_mask.png` | "calm ocean water" | **Kandinsky 2.2** | `coastal_result.png` |

**Settings for all:** guidance scale 7.5, inference steps 30, strength 1.0, negative prompt "blurry, low quality, distorted"

### Steps per test:
1. Open **Inpainting** tab
2. Upload the source image
3. Draw the mask (or load existing mask)
4. **Select the correct model** from the dropdown
5. Enter the prompt, keep default parameters
6. Click **Generate**
7. **Download** and rename to the filename above
8. Place result in `test/diffusion/`

### After all 4 inpainting tests — copy to report:
```bash
cp test/diffusion/beach_result.png report/images/inpaint_beach_result.png
cp test/diffusion/old_vintage_car_result.png report/images/inpaint_car_result.png
cp test/diffusion/mountain_result.png report/images/inpaint_mountain_result.png
cp test/diffusion/coastal_result.png report/images/inpaint_coastal_result.png
```

---

## Restoration (`test/restoration/`)

| # | Image | Face Model | Fidelity | Upscale | Save As |
|---|-------|-----------|----------|---------|---------|
| 1 | `degraded_face_01.png` | **CodeFormer** | 0.5 | 2x | `restored_face_01.png` |
| 2 | `degraded_face_02.jpg` | **CodeFormer** | 0.5 | 2x | `restored_face_02.png` |
| 3 | `degraded_face_03.png` | **GFPGAN** | — | 2x | `restored_face_03.png` |
| 4 | `degraded_face_04.png` | **CodeFormer** | 0.5 | 4x | `restored_face_04.png` |
| 5 | `degraded_face_05.jpg` | **GFPGAN** | — | 2x | `restored_face_05.png` |

**Settings:** Face Enhancement ON, Scratch Removal OFF

### Steps per test:
1. Open **Restoration** tab
2. Upload the image
3. **Select the correct face model** (CodeFormer or GFPGAN)
4. Set fidelity to **0.5** (CodeFormer only)
5. Set upscale to **2x** (or 4x for test #4)
6. Click **Restore**
7. **Download** and rename to the filename above
8. Place result in `test/restoration/`

### After all 5 restoration tests — copy to report:
```bash
cp test/restoration/restored_face_01.png report/images/restore_face01_result.png
```

---

## Style Transfer — NO ACTION NEEDED

Style transfer results are already correct. No rerun required.

---

## Final Steps

After placing all new images:
1. Recompile the report: `cd report && pdflatex report.tex && pdflatex report.tex`
2. Verify all figures look correct in the PDF
3. Commit and push
