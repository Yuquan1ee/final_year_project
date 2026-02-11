# Figures and Screenshots Needed for FYP Report

Status: 6 of 7 high-priority figures completed. Section 6 uses code listings (`\lstlisting`) and tables only -- no image figures needed. See checklist at the bottom.

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

## Section 6: Implementation (WRITTEN)

Section 6 uses **20 code listings** (`\lstlisting`) and **3 tables** -- no image figures. Demo screenshots (items 10-13) are deferred to a Results section if needed.

### 8. Backend Directory Tree -- DONE (Listing 1, `\lstlisting`)
### 9. Code Snippets / Listings -- DONE (Listings 1-20 in Section 6)
### 10-12. Demo Screenshots -- NOT NEEDED for Section 6 (defer to Results)
### 13. UI Screenshots -- NOT NEEDED (wireframes in Section 5.3 are sufficient)
### 14. Colab Deployment Screenshot -- LOW PRIORITY (optional for future sections)
### 15. API Docs (Swagger) -- LOW PRIORITY (optional)

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
| 8   | Backend Directory Tree    | `\lstlisting` block         | 6.1.1   | DONE (Listing 1)         |
| 9   | Code Snippets (x20)      | `\lstlisting` blocks        | 6.x     | DONE (Listings 1-20)     |
| 10  | Demo: Inpainting          | `demo_inpainting.png`       | Results | DEFERRED                 |
| 11  | Demo: Style Transfer      | `demo_style_transfer.png`   | Results | DEFERRED                 |
| 12  | Demo: Restoration         | `demo_restoration.png`      | Results | DEFERRED                 |
| 13  | UI Screenshots (x4)       | `ui_*.png`                  | ---     | NOT NEEDED               |
| 14  | Colab Deployment          | `colab_deployment.png`      | ---     | LOW                      |
| 15  | API Docs (Swagger)        | `api_docs.png`              | ---     | LOW                      |

**Remaining action:** Create figures 2-3 (iterative methodology diagram, system architecture diagram) and place them in `report/images/`. Then update the `\includegraphics` lines and remove the `\fbox` placeholder lines in the LaTeX source.
