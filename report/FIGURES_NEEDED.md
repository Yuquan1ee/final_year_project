# Figures and Screenshots Needed for FYP Report

Status: Only `ntu_logo.png` exists in `report/images/`. All figures below need to be created.

---

## Section 4: Software Requirements

### 1. Use Case Diagram

- **File:** `images/use_case_diagram.png`
- **Location:** Section 4.1 (line ~965), `\label{fig:use_case_diagram}`
- **Tool:** Draw.io or Lucidchart
- **Description:** UML use case diagram showing User and System actors interacting with DiffusionDesk. 10 use cases: Upload Image, Perform Inpainting, Draw Inpainting Mask, Apply Style Transfer, Select Style Preset, Restore Image, Select Model, Configure Generation Parameters, Download Result Image, Load Diffusion Model. Include `<<include>>` and `<<extend>>` relationships as specified in the LaTeX comments (lines 930-962).
- **Size:** `width=0.85\textwidth`

---

## Section 5: Planning and Design

### 2. Iterative Development Methodology Diagram

- **File:** `images/iterative_methodology.png`
- **Location:** Section 5.1 (line ~1594), `\label{fig:iterative_methodology}`
- **Tool:** Draw.io
- **Description:** Circular/iterative diagram showing the development cycle: Requirements -> Design -> Implement -> Test -> Evaluate -> (repeat). Label the 4 iterations described in the report.
- **Size:** `width=0.7\textwidth`

### 3. System Architecture Diagram

- **File:** `images/system_architecture.png`
- **Location:** Section 5.2 (line ~1663), `\label{fig:system_architecture}`
- **Tool:** Draw.io
- **Description:** Two-box architecture diagram showing Frontend (React SPA) and Backend (FastAPI) connected via HTTP/JSON. Frontend box contains: Tab components (InpaintingTab, StyleTransferTab, RestorationTab), shared components (MaskCanvas, ImageUpload), API Client (imageApi.ts). Backend box contains: Routers (/api/inpainting/, /api/style/, /api/restoration/, /api/health), Schemas (Pydantic), Services (DiffusionService, RestorationService), GPU layer (PyTorch + Diffusers). Show deployment options (Vercel/localhost:5173 for frontend, Colab+ngrok/localhost:8000 for backend). Detailed ASCII layout in LaTeX comments (lines 1619-1658).
- **Size:** `width=0.9\textwidth`

### 4. Home Page Wireframe

- **File:** `images/wireframe_home.png`
- **Location:** Section 5.3 (line ~1764), `\label{fig:wireframe_home}`
- **Tool:** Draw.io or Figma
- **Description:** Wireframe showing: top nav bar with app name + 4 tab buttons (Home, Inpainting, Style, Restoration), welcome text, 3 feature cards (Inpainting, Style Transfer, Restoration) with brief descriptions. ASCII layout in LaTeX comments (lines 1744-1759).
- **Size:** `width=0.85\textwidth`

### 5. Inpainting Page Wireframe

- **File:** `images/wireframe_inpainting.png`
- **Location:** Section 5.3 (line ~1808), `\label{fig:wireframe_inpainting}`
- **Tool:** Draw.io or Figma
- **Description:** Wireframe showing: left side = image canvas with mask overlay + brush size slider + Clear Mask button + Upload Image button; right side = model dropdown, prompt input, negative prompt input, collapsible Advanced Settings (guidance, steps, strength, seed, quantisation sliders); Generate button; bottom = side-by-side Original vs Result comparison + Download button. ASCII layout in LaTeX comments (lines 1776-1803).
- **Size:** `width=0.85\textwidth`

### 6. Style Transfer Page Wireframe

- **File:** `images/wireframe_style.png`
- **Location:** Section 5.3 (line ~1853), `\label{fig:wireframe_style}`
- **Tool:** Draw.io or Figma
- **Description:** Wireframe showing: left side = upload + image preview; right side = model dropdown, grid of style preset buttons (Anime, Oil Painting, Watercolour, Pixel Art, Cinematic, Sketch, Cyberpunk, Pop Art), custom prompt checkbox + text field, strength slider; Generate button; bottom = side-by-side comparison + Download. ASCII layout in LaTeX comments (lines 1822-1848).
- **Size:** `width=0.85\textwidth`

### 7. Restoration Page Wireframe

- **File:** `images/wireframe_restoration.png`
- **Location:** Section 5.3 (line ~1895), `\label{fig:wireframe_restoration}`
- **Tool:** Draw.io or Figma
- **Description:** Wireframe showing: left side = upload + image preview; right side = checkbox options for Face Enhancement (model selector: CodeFormer/GFPGAN, fidelity slider), Upscaling (None/2x/4x radio buttons), Scratch Removal toggle, Colourisation toggle; Restore button; bottom = side-by-side comparison + Download. ASCII layout in LaTeX comments (lines 1865-1890).
- **Size:** `width=0.85\textwidth`

---

## Section 6: Implementation (future - not yet written)

These figures will likely be needed when Section 6 is written:

### 8. Backend Project Structure (Directory Tree)

- **File:** `images/backend_structure.png` (or use a `\lstlisting` code block instead)
- **Location:** Section 6.1.1
- **Description:** Directory tree of `backend/app/` showing routers/, services/, schemas/ structure.

### 9. Code Snippets / Listings

- Not image files -- use `\lstlisting` blocks for key code excerpts (API endpoint definitions, diffusion pipeline loading, quantisation logic, mask canvas implementation).

### 10. Inpainting Demo Screenshots

- **File:** `images/demo_inpainting.png`
- **Location:** Section 6.1.3 or a Results subsection
- **Description:** Screenshot of the app performing inpainting: show original image with mask drawn, and the AI-generated result side-by-side.

### 11. Style Transfer Demo Screenshots

- **File:** `images/demo_style_transfer.png`
- **Location:** Section 6.1.4 or a Results subsection
- **Description:** Screenshot showing original image and stylised output (e.g., anime or oil painting style).

### 12. Restoration Demo Screenshots

- **File:** `images/demo_restoration.png`
- **Location:** Section 6.1.5 or a Results subsection
- **Description:** Screenshot showing before/after face enhancement or upscaling.

### 13. Frontend UI Screenshots (actual, not wireframes)

- **File:** `images/ui_home.png`, `images/ui_inpainting.png`, `images/ui_style.png`, `images/ui_restoration.png`
- **Location:** Section 6.2.2
- **Description:** Actual screenshots of the running application for each tab. These complement the wireframes from Section 5.

### 14. Google Colab Deployment Screenshot

- **File:** `images/colab_deployment.png`
- **Location:** Section 6 (deployment subsection)
- **Description:** Screenshot of the Colab notebook running the backend with ngrok URL visible.

### 15. API Documentation Screenshot

- **File:** `images/api_docs.png`
- **Location:** Section 6.1.2
- **Description:** Screenshot of the FastAPI auto-generated Swagger UI at `/docs`.

---

## Summary Checklist

| #   | Figure                    | File                        | Section | Priority                 |
| --- | ------------------------- | --------------------------- | ------- | ------------------------ |
| 1   | Use Case Diagram          | `use_case_diagram.png`      | 4.1     | HIGH - needed now        |
| 2   | Iterative Methodology     | `iterative_methodology.png` | 5.1     | HIGH - needed now        |
| 3   | System Architecture       | `system_architecture.png`   | 5.2     | HIGH - needed now        |
| 4   | Wireframe: Home           | `wireframe_home.png`        | 5.3     | HIGH - needed now        |
| 5   | Wireframe: Inpainting     | `wireframe_inpainting.png`  | 5.3     | HIGH - needed now        |
| 6   | Wireframe: Style Transfer | `wireframe_style.png`       | 5.3     | HIGH - needed now        |
| 7   | Wireframe: Restoration    | `wireframe_restoration.png` | 5.3     | HIGH - needed now        |
| 8   | Backend Directory Tree    | code listing or image       | 6.1.1   | MEDIUM - when writing S6 |
| 9   | Code Snippets             | `\lstlisting` blocks        | 6.x     | MEDIUM - when writing S6 |
| 10  | Demo: Inpainting          | `demo_inpainting.png`       | 6.x     | MEDIUM - when writing S6 |
| 11  | Demo: Style Transfer      | `demo_style_transfer.png`   | 6.x     | MEDIUM - when writing S6 |
| 12  | Demo: Restoration         | `demo_restoration.png`      | 6.x     | MEDIUM - when writing S6 |
| 13  | UI Screenshots (x4)       | `ui_*.png`                  | 6.2.2   | MEDIUM - when writing S6 |
| 14  | Colab Deployment          | `colab_deployment.png`      | 6.x     | LOW - when writing S6    |
| 15  | API Docs (Swagger)        | `api_docs.png`              | 6.1.2   | LOW - when writing S6    |

**Immediate action:** Create figures 1-7 (use case diagram, methodology diagram, system architecture, 4 wireframes) and place them in `report/images/`. Then uncomment the `\includegraphics` lines and remove the `\fbox` placeholder lines in the LaTeX source.
