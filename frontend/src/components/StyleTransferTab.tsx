/**
 * StyleTransferTab Component
 *
 * Purpose:
 * - Transform images into different artistic styles
 * - Supports preset styles (anime, oil painting, watercolor, etc.)
 * - Supports custom style descriptions via text prompt
 *
 * TODO:
 * - [x] Add image upload component
 * - [ ] Add style preset grid/buttons
 * - [ ] Add custom style text input
 * - [ ] Add strength slider (how much style to apply)
 * - [ ] Add generate button
 * - [ ] Add result display area
 * - [ ] Add side-by-side comparison view
 * - [ ] Add download button for result
 * - [ ] Add loading state during generation
 * - [ ] Add error handling and display
 */

import { useState } from 'react'
import ImageUpload from './ImageUpload'

function StyleTransferTab() {
  const [sourceImage, setSourceImage] = useState<File | null>(null)

  const handleSourceImageSelect = (file: File, _previewUrl: string) => {
    setSourceImage(file)
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Style Transfer</h2>
      <p className="text-slate-400 mb-6">
        Apply artistic styles like anime, oil painting, or watercolor to your images.
      </p>

      <ImageUpload
        label="Source Image"
        onImageSelect={handleSourceImageSelect}
        onImageClear={() => setSourceImage(null)}
      />

      {/* Placeholder for style selection and controls */}
      {sourceImage && (
        <div className="mt-6 p-4 bg-slate-800 rounded-lg">
          <p className="text-slate-300">
            Image uploaded. Style selection and generation controls coming next.
          </p>
        </div>
      )}
    </div>
  )
}

export default StyleTransferTab
