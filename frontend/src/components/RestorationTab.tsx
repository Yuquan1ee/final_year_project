/**
 * RestorationTab Component
 *
 * Purpose:
 * - Restore old, damaged, or low-quality photos
 * - Enhance image resolution and quality
 * - Remove scratches, noise, and artifacts
 * - No prompt needed - restoration is automatic based on selected options
 *
 * User Flow:
 * 1. Upload source image (old/damaged photo)
 * 2. Select restoration options (checkboxes/toggles)
 * 3. Adjust settings if needed
 * 4. Click Restore
 * 5. View/download result
 *
 * Restoration Options:
 * - Face Enhancement (CodeFormer/GFPGAN)
 * - Upscaling (Real-ESRGAN 2x/4x)
 * - Scratch/Artifact Removal
 * - Denoising
 * - Colorization (for B&W photos)
 *
 * TODO:
 * - [x] Image upload component
 * - [x] Restoration options panel (checkboxes/toggles)
 *   - [x] Face enhancement toggle
 *   - [x] Upscale factor selector (None, 2x, 4x)
 *   - [x] Scratch removal toggle
 *   - [x] Colorize toggle (for B&W)
 * - [x] Fidelity slider for CodeFormer (0.0-1.0)
 * - [x] Restore button
 * - [x] Result display with before/after comparison
 * - [x] Download button for result
 * - [x] Loading state during API call
 * - [x] Error handling and user feedback
 * - [ ] Connect to backend API
 */

import { useState } from 'react'
import ImageUpload from './ImageUpload'

// Upscale options
const UPSCALE_OPTIONS = [
  { id: 'none', name: 'None', factor: 1 },
  { id: '2x', name: '2x Upscale', factor: 2 },
  { id: '4x', name: '4x Upscale', factor: 4 },
] as const

type UpscaleId = typeof UPSCALE_OPTIONS[number]['id']

// Face restoration models
const FACE_MODELS = [
  { id: 'codeformer', name: 'CodeFormer', description: 'Best quality, adjustable fidelity' },
  { id: 'gfpgan', name: 'GFPGAN', description: 'Fast, good for most faces' },
] as const

type FaceModelId = typeof FACE_MODELS[number]['id']

function RestorationTab() {
  // Image state
  const [sourceImageUrl, setSourceImageUrl] = useState<string | null>(null)
  const [sourceImageFile, setSourceImageFile] = useState<File | null>(null)

  // Restoration options
  const [enableFaceEnhance, setEnableFaceEnhance] = useState(true)
  const [faceModel, setFaceModel] = useState<FaceModelId>('codeformer')
  const [fidelity, setFidelity] = useState(0.5)
  const [upscale, setUpscale] = useState<UpscaleId>('2x')
  const [enableScratchRemoval, setEnableScratchRemoval] = useState(false)
  const [enableColorize, setEnableColorize] = useState(false)

  // Generation state
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null)

  const handleSourceImageSelect = (file: File, previewUrl: string) => {
    setSourceImageUrl(previewUrl)
    setSourceImageFile(file)
    setResultImageUrl(null)
    setError(null)
  }

  const handleImageClear = () => {
    setSourceImageUrl(null)
    setSourceImageFile(null)
    setResultImageUrl(null)
    setError(null)
  }

  const hasAnyOption = enableFaceEnhance || upscale !== 'none' || enableScratchRemoval || enableColorize
  const canGenerate = sourceImageUrl && hasAnyOption

  const handleGenerate = async () => {
    if (!canGenerate || !sourceImageFile) return

    setIsGenerating(true)
    setError(null)
    setResultImageUrl(null)

    try {
      // TODO: Implement actual API call to backend
      // For now, simulate a delay
      await new Promise(resolve => setTimeout(resolve, 2000))

      // Placeholder: In real implementation, this would be the result from the API
      setResultImageUrl(sourceImageUrl)

      console.log('Restoration params:', {
        hasImage: !!sourceImageFile,
        enableFaceEnhance,
        faceModel,
        fidelity,
        upscale,
        enableScratchRemoval,
        enableColorize,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during restoration')
    } finally {
      setIsGenerating(false)
    }
  }

  const handleDownload = () => {
    if (!resultImageUrl) return

    const link = document.createElement('a')
    link.href = resultImageUrl
    link.download = `restored-${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Restoration</h2>
      <p className="text-slate-400 mb-6">
        Restore old or damaged photos automatically. Enable the options you need and let AI do the work.
      </p>

      {/* Step 1: Upload image */}
      {!sourceImageUrl && (
        <ImageUpload
          label="Source Image"
          onImageSelect={handleSourceImageSelect}
          onImageClear={handleImageClear}
        />
      )}

      {/* Step 2: Configure restoration options */}
      {sourceImageUrl && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left column: Source image preview */}
          <div>
            <div className="flex justify-between items-center mb-4">
              <span className="text-slate-300 font-medium">Source Image</span>
              <button
                onClick={handleImageClear}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300
                           rounded-lg text-sm transition-colors"
              >
                Change Image
              </button>
            </div>

            <div className="rounded-lg overflow-hidden bg-slate-800">
              <img
                src={sourceImageUrl}
                alt="Source"
                className="w-full h-auto"
              />
            </div>
          </div>

          {/* Right column: Restoration options */}
          <div className="space-y-6">
            <div className="p-4 bg-slate-800 rounded-lg space-y-5">
              <h3 className="text-slate-200 font-medium">Restoration Options</h3>

              {/* Face Enhancement */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="faceEnhance"
                      checked={enableFaceEnhance}
                      onChange={(e) => setEnableFaceEnhance(e.target.checked)}
                      className="w-4 h-4 rounded border-slate-600 bg-slate-700
                                 text-indigo-600 focus:ring-indigo-500"
                    />
                    <label htmlFor="faceEnhance" className="text-sm font-medium text-slate-300">
                      Face Enhancement
                    </label>
                  </div>
                  <span className="text-xs text-slate-500">Recommended</span>
                </div>

                {enableFaceEnhance && (
                  <div className="ml-7 space-y-3">
                    {/* Face model selection */}
                    <div className="flex gap-2">
                      {FACE_MODELS.map((model) => (
                        <button
                          key={model.id}
                          onClick={() => setFaceModel(model.id)}
                          className={`px-3 py-1.5 rounded-lg text-sm transition-colors
                                      ${faceModel === model.id
                                        ? 'bg-indigo-600 text-white'
                                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                                      }`}
                        >
                          {model.name}
                        </button>
                      ))}
                    </div>

                    {/* Fidelity slider (CodeFormer only) */}
                    {faceModel === 'codeformer' && (
                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <label className="text-xs text-slate-400">Fidelity</label>
                          <span className="text-xs text-slate-500">{fidelity.toFixed(1)}</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={fidelity}
                          onChange={(e) => setFidelity(Number(e.target.value))}
                          className="w-full accent-indigo-500"
                        />
                        <div className="flex justify-between text-xs text-slate-500 mt-1">
                          <span>Quality</span>
                          <span>Fidelity</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Upscaling */}
              <div className="space-y-3">
                <label className="text-sm font-medium text-slate-300">Upscaling</label>
                <div className="flex gap-2">
                  {UPSCALE_OPTIONS.map((option) => (
                    <button
                      key={option.id}
                      onClick={() => setUpscale(option.id)}
                      className={`px-4 py-2 rounded-lg text-sm transition-colors
                                  ${upscale === option.id
                                    ? 'bg-indigo-600 text-white'
                                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                                  }`}
                    >
                      {option.name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Scratch/Artifact Removal */}
              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  id="scratchRemoval"
                  checked={enableScratchRemoval}
                  onChange={(e) => setEnableScratchRemoval(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-600 bg-slate-700
                             text-indigo-600 focus:ring-indigo-500"
                />
                <label htmlFor="scratchRemoval" className="text-sm font-medium text-slate-300">
                  Scratch & Artifact Removal
                </label>
              </div>

              {/* Colorization */}
              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  id="colorize"
                  checked={enableColorize}
                  onChange={(e) => setEnableColorize(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-600 bg-slate-700
                             text-indigo-600 focus:ring-indigo-500"
                />
                <label htmlFor="colorize" className="text-sm font-medium text-slate-300">
                  Colorize (for B&W photos)
                </label>
              </div>
            </div>

            {/* Generate button */}
            <button
              onClick={handleGenerate}
              disabled={!canGenerate || isGenerating}
              className={`w-full py-3 px-6 rounded-lg font-medium text-white
                          transition-all duration-200 flex items-center justify-center gap-2
                          ${canGenerate && !isGenerating
                            ? 'bg-indigo-600 hover:bg-indigo-700 cursor-pointer'
                            : 'bg-slate-700 cursor-not-allowed opacity-60'
                          }`}
            >
              {isGenerating ? (
                <>
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Restoring...
                </>
              ) : (
                'Restore'
              )}
            </button>

            {/* Validation hint */}
            {!hasAnyOption && (
              <p className="text-amber-400 text-sm">
                Enable at least one restoration option.
              </p>
            )}

            {/* Error display */}
            {error && (
              <div className="p-4 bg-red-900/30 border border-red-700 rounded-lg">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Result display */}
      {resultImageUrl && (
        <div className="mt-8">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-medium text-slate-200">Result</h3>
            <button
              onClick={handleDownload}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white
                         rounded-lg text-sm font-medium transition-colors
                         flex items-center gap-2"
            >
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download
            </button>
          </div>

          {/* Side by side comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-slate-400 text-sm mb-2">Original</p>
              <div className="rounded-lg overflow-hidden bg-slate-800">
                <img
                  src={sourceImageUrl!}
                  alt="Original"
                  className="w-full h-auto"
                />
              </div>
            </div>
            <div>
              <p className="text-slate-400 text-sm mb-2">Restored</p>
              <div className="rounded-lg overflow-hidden bg-slate-800">
                <img
                  src={resultImageUrl}
                  alt="Result"
                  className="w-full h-auto"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default RestorationTab
