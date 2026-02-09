# Figures and Screenshots Needed for FYP Report

Status: 6 of 7 high-priority figures completed. See checklist at the bottom.

---

## Section 4: Software Requirements

### 1. Use Case Diagram - DONE

- **File:** `images/UML_Diagram.png`
- **Location:** Section 4.1, `\label{fig:use_case_diagram}`
- **Status:** Inserted in report.

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

### 4. Home Page Wireframe - DONE

- **File:** `images/wireframe_home.png`
- **Location:** Section 5.3, `\label{fig:wireframe_home}`
- **Status:** Inserted in report.

### 5. Inpainting Page Wireframe - DONE

- **File:** `images/wireframe_inpainting.png`
- **Location:** Section 5.3, `\label{fig:wireframe_inpainting}`
- **Status:** Inserted in report.

### 6. Style Transfer Page Wireframe - DONE

- **File:** `images/wireframe_style.png`
- **Location:** Section 5.3, `\label{fig:wireframe_style}`
- **Status:** Inserted in report.

### 7. Restoration Page Wireframe - DONE

- **File:** `images/wireframe_restoration.png`
- **Location:** Section 5.3, `\label{fig:wireframe_restoration}`
- **Status:** Inserted in report.

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

| #   | Figure                    | File                        | Section | Status                   |
| --- | ------------------------- | --------------------------- | ------- | ------------------------ |
| 1   | Use Case Diagram          | `UML_Diagram.png`           | 4.1     | DONE                     |
| 2   | Iterative Methodology     | `iterative_methodology.png` | 5.1     | TODO                     |
| 3   | System Architecture       | `system_architecture.png`   | 5.2     | TODO                     |
| 4   | Wireframe: Home           | `wireframe_home.png`        | 5.3     | DONE                     |
| 5   | Wireframe: Inpainting     | `wireframe_inpainting.png`  | 5.3     | DONE                     |
| 6   | Wireframe: Style Transfer | `wireframe_style.png`       | 5.3     | DONE                     |
| 7   | Wireframe: Restoration    | `wireframe_restoration.png` | 5.3     | DONE                     |
| 8   | Backend Directory Tree    | code listing or image       | 6.1.1   | MEDIUM - when writing S6 |
| 9   | Code Snippets             | `\lstlisting` blocks        | 6.x     | MEDIUM - when writing S6 |
| 10  | Demo: Inpainting          | `demo_inpainting.png`       | 6.x     | MEDIUM - when writing S6 |
| 11  | Demo: Style Transfer      | `demo_style_transfer.png`   | 6.x     | MEDIUM - when writing S6 |
| 12  | Demo: Restoration         | `demo_restoration.png`      | 6.x     | MEDIUM - when writing S6 |
| 13  | UI Screenshots (x4)       | `ui_*.png`                  | 6.2.2   | MEDIUM - when writing S6 |
| 14  | Colab Deployment          | `colab_deployment.png`      | 6.x     | LOW - when writing S6    |
| 15  | API Docs (Swagger)        | `api_docs.png`              | 6.1.2   | LOW - when writing S6    |

**Remaining action:** Create figures 2-3 (iterative methodology diagram, system architecture diagram) and place them in `report/images/`. Then update the `\includegraphics` lines and remove the `\fbox` placeholder lines in the LaTeX source.
