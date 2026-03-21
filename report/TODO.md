# FYP Report — Submission TODO

**Deadline:** 23 March 2026 (tomorrow)

---

## HIGH PRIORITY (Must fix before submission)

- [x] **Fix examiner placeholder** — Replace `Prof (placeholder)` with `A/P Lin Guosheng` in Acknowledgements (line 156)
- [x] **Create Iterative Methodology Diagram** — Section 5.1 has an `\fbox` placeholder (line 1562). Create `images/iterative_methodology.png` using Draw.io showing the iterative development cycle (Requirements → Design → Implement → Test → Evaluate → repeat). Remove the `\fbox` and uncomment the `\includegraphics` line.
- [x] **Create System Architecture Diagram** — Section 5.2 has an `\fbox` placeholder (line 1631). Create `images/system_architecture.png` using Draw.io. Remove the `\fbox` and uncomment the `\includegraphics` line. The diagram should show:

  **Frontend (React SPA)** box — hosted on Vercel/Netlify or localhost:5173:
  - Tab Components: `InpaintingTab`, `StyleTransferTab`, `RestorationTab`
  - Shared Components: `MaskCanvas`, `ImageUpload`
  - API Client: `imageApi.ts` + `config.ts`

  ↕ HTTP/JSON (Base64 images + JSON params) ↕

  **Backend (FastAPI)** box — hosted on Colab + ngrok or localhost:8000:
  - **Router Layer** (`routers/`):
    - `/api/inpainting/`
    - `/api/style/`
    - `/api/restoration/`
    - `/api/health`
  - **Schema Layer** (`schemas/image.py`):
    - `InpaintingRequest`
    - `StyleTransferRequest`
    - `RestorationRequest`
    - `ImageResponse`
  - **Service Layer** (`services/`):
    - `DiffusionService` (inpainting + style transfer)
    - `RestorationService` (CodeFormer, GFPGAN, Real-ESRGAN)
  - **GPU Layer**:
    - PyTorch + Diffusers
    - Model Cache (lazy loading + caching)
    - CUDA

  Flow: Routers → Schemas (validation) → Services → GPU

---

## MEDIUM PRIORITY (Strongly recommended)

- [ ] **Add Results/Evaluation section** — Currently the report jumps from Section 6 (Implementation) to Section 7 (Difficulties) with no evidence the system works. Add a section between them covering:
  - Demo screenshots: inpainting result, style transfer result, restoration result
  - Performance benchmarks (inference times per model, VRAM usage)
  - Output quality comparison across models
  - This addresses Project Objective #6: "evaluate the system's performance through processing speed benchmarks, output quality assessments, and usability considerations"

- [ ] **Add Conclusion section** — No formal conclusion exists. Add a brief section (after Future Implementation) summarising whether project objectives were met and key contributions.

---

## LOW PRIORITY (Nice to have)

- [ ] **Swagger API docs screenshot** — Screenshot of FastAPI `/docs` page, optional addition to Implementation or Results
- [ ] **Colab deployment screenshot** — Shows the deployment pipeline works
- [ ] **Verify bibliography** — Rebuild PDF and check for undefined `\citep{}`/`\citet{}` references or missing bib entries
- [ ] **Check for overfull hbox warnings** — Run `pdflatex` and review any margin overflow warnings
- [ ] **Rebuild PDF** — Final `pdflatex` + `bibtex` + `pdflatex` × 2 to ensure TOC, LOF, LOT, and references are all up to date
