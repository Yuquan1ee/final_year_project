# Frontend - Diffusion Image Editor

React web application for the Diffusion Models FYP project.

## Tech Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first styling

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev
# Opens at http://localhost:5173

# Build for production
npm run build

# Preview production build
npm run preview

# Run linter
npm run lint
```

## Project Structure

```
src/
├── components/              # React components
│   ├── HomeTab.tsx          # Welcome page with project info
│   ├── InpaintingTab.tsx    # Inpainting feature
│   ├── StyleTransferTab.tsx # Style transfer feature
│   └── RestorationTab.tsx   # Photo restoration feature
├── App.tsx                  # Main app with tab navigation
├── main.tsx                 # React entry point
└── index.css                # Tailwind CSS import
```

## Features

| Tab | Description |
|-----|-------------|
| Home | Project overview, feature descriptions, tech stack info |
| Inpainting | Remove objects or fill masked regions using AI |
| Style Transfer | Apply artistic styles (anime, oil painting, etc.) |
| Restoration | Restore old/damaged photos, enhance quality |

## Environment Variables

Create a `.env` file (optional):

```env
VITE_API_URL=http://localhost:8000
```

If not set, defaults to `http://localhost:8000`.

## Adding New Components

1. Create component in `src/components/`
2. Use TypeScript for type safety
3. Use Tailwind classes for styling
4. Import and use in `App.tsx`

## Tailwind CSS

Tailwind is configured via the Vite plugin. All utility classes are available:

```tsx
// Example usage
<div className="bg-slate-800 p-4 rounded-lg">
  <h1 className="text-xl font-bold text-white">Title</h1>
</div>
```

Reference: [Tailwind CSS Documentation](https://tailwindcss.com/docs)
