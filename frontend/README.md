# Diffusion Image Editor - Frontend

React frontend for intelligent image editing using diffusion models. Built with TypeScript, Vite, and Tailwind CSS.

## Features

- **Inpainting Tab**: Upload image, draw mask, generate with AI
- **Style Transfer Tab**: Apply artistic styles to images
- **Restoration Tab**: Enhance faces, upscale, remove scratches
- **Responsive Design**: Works on desktop and mobile
- **Real-time Preview**: See your edits before generating

## Tech Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first styling

## Requirements

- Node.js 18+
- npm or yarn

## Installation

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at http://localhost:5173

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |

## Configuration

### Backend URL

Create a `.env` file to configure the backend URL:

```bash
# For local development
VITE_API_URL=http://localhost:8000

# For Google Colab (replace with your ngrok URL)
VITE_API_URL=https://xxxx-xx-xxx-xxx-xxx.ngrok-free.app

# For production deployment
VITE_API_URL=https://your-api-domain.com
```

If not set, defaults to `http://localhost:8000`.

## Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   ├── config.ts        # API URL, base64 utilities
│   │   ├── imageApi.ts      # API client functions
│   │   └── index.ts         # Module exports
│   ├── components/
│   │   ├── HomeTab.tsx      # Project info, welcome page
│   │   ├── InpaintingTab.tsx    # Inpainting feature
│   │   ├── StyleTransferTab.tsx # Style transfer feature
│   │   ├── RestorationTab.tsx   # Restoration feature
│   │   ├── ImageUpload.tsx  # Drag-drop image upload
│   │   └── MaskCanvas.tsx   # Draw mask on image
│   ├── App.tsx              # Main app with tab navigation
│   ├── main.tsx             # React entry point
│   └── index.css            # Tailwind CSS
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── .env.example
└── README.md
```

## Components

### App.tsx

Main application with tab navigation:
- Home (project info)
- Inpainting
- Style Transfer
- Restoration

### InpaintingTab.tsx

1. Upload source image
2. Draw mask over areas to edit (red overlay)
3. Enter prompt describing desired content
4. Select model (SD, SDXL, Kandinsky, FLUX)
5. Click Generate
6. Download result

### StyleTransferTab.tsx

1. Upload source image
2. Select style preset OR enter custom description
3. Adjust strength slider (30-90%)
4. Click Generate
5. Download result

**Style Presets**: Anime, Oil Painting, Watercolor, Sketch, Cyberpunk, Studio Ghibli, Impressionist, 3D Render

### RestorationTab.tsx

1. Upload old/damaged photo
2. Enable desired options:
   - Face Enhancement (CodeFormer/GFPGAN)
   - Upscaling (2x/4x with Real-ESRGAN)
   - Scratch Removal
   - Colorization (B&W photos)
3. Click Restore
4. Download result

### ImageUpload.tsx

Reusable upload component:
- Drag and drop support
- Click to select
- File type/size validation
- Preview with replace/clear buttons
- Mobile camera support

### MaskCanvas.tsx

Interactive mask drawing:
- Brush tool with adjustable size
- Eraser tool
- Clear mask button
- Red overlay for visibility
- Touch support for mobile

## API Integration

The frontend communicates with the backend via REST API:

```typescript
// Example: Inpainting
import { inpaintImage } from '../api';

const response = await inpaintImage({
  image: file,           // File object
  mask: maskDataUrl,     // base64 from canvas
  prompt: "a garden",
  model: "sd-inpainting",
});

if (response.success) {
  setResultImage(response.image);  // base64 data URL
}
```

### Available API Functions

| Function | Description |
|----------|-------------|
| `checkHealth()` | Check backend status and GPU info |
| `inpaintImage(params)` | Inpaint masked regions |
| `editImage(params)` | Edit with text instructions |
| `styleTransfer(params)` | Apply style to image |
| `restoreImage(params)` | Restore/enhance image |
| `getInpaintingModels()` | List available models |
| `getStylePresets()` | List style presets |

## Building for Production

```bash
# Build
npm run build

# Output is in dist/
# Deploy dist/ to any static hosting (Vercel, Netlify, S3, etc.)
```

## Deployment Options

### Vercel (Recommended)

```bash
npm install -g vercel
vercel
```

### Netlify

```bash
npm run build
# Upload dist/ folder to Netlify
```

### Docker

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
```

## Tailwind CSS

Tailwind is configured via the Vite plugin. All utility classes are available:

```tsx
// Example usage
<div className="bg-slate-800 p-4 rounded-lg">
  <h1 className="text-xl font-bold text-white">Title</h1>
</div>
```

Reference: [Tailwind CSS Documentation](https://tailwindcss.com/docs)

## Troubleshooting

### API Connection Failed

1. Check backend is running
2. Verify `VITE_API_URL` is correct in `.env`
3. Check browser console for CORS errors
4. Ensure backend CORS allows your frontend origin

### Images Not Loading

- Check image is valid format (PNG, JPEG, WebP)
- Check file size < 10MB
- Check browser console for errors

### Mask Not Working

- Ensure image is loaded first
- Try clearing and redrawing
- Check browser console for canvas errors

## Browser Support

- Chrome 90+
- Firefox 90+
- Safari 14+
- Edge 90+

## License

MIT License - See LICENSE file in root directory.
