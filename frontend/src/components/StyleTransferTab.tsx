/**
 * StyleTransferTab Component
 *
 * Purpose:
 * - Transform images into different artistic styles
 * - Supports preset styles (anime, oil painting, watercolor, etc.)
 * - Supports custom style descriptions via text prompt
 * - No mask needed - style applies to entire image
 *
 * User Flow:
 * 1. Upload source image
 * 2. Select style preset OR enter custom style description
 * 3. Adjust strength slider (how much style to apply)
 * 4. Click Generate
 * 5. View/download result
 *
 * Available Style Presets:
 * - Anime / Japanese Animation
 * - Oil Painting
 * - Watercolor
 * - Sketch / Pencil Drawing
 * - Cyberpunk
 * - Studio Ghibli
 * - Impressionist
 * - 3D Render
 *
 * TODO:
 * - [x] Image upload component
 * - [x] Style preset grid (clickable style cards)
 * - [x] Custom style text input
 * - [x] Strength slider (0.3 - 0.9)
 * - [x] Model selection (IP-Adapter, SDXL img2img, etc.)
 * - [x] Generate button with validation
 * - [x] Result display with before/after comparison
 * - [x] Download button for result
 * - [x] Loading state during API call
 * - [x] Error handling and user feedback
 * - [ ] Connect to backend API
 */

import { useState } from 'react'
import ImageUpload from './ImageUpload'

// Available style presets
const STYLE_PRESETS = [
  { id: 'anime', name: 'Anime', icon: '🎌', prompt: 'anime style, cel shading, vibrant colors, japanese animation' },
  { id: 'oil_painting', name: 'Oil Painting', icon: '🎨', prompt: 'oil painting style, thick brushstrokes, vibrant colors, artistic' },
  { id: 'watercolor', name: 'Watercolor', icon: '💧', prompt: 'watercolor painting style, soft edges, flowing colors, artistic' },
  { id: 'sketch', name: 'Sketch', icon: '✏️', prompt: 'pencil sketch, black and white, detailed linework, hand drawn' },
  { id: 'cyberpunk', name: 'Cyberpunk', icon: '🌃', prompt: 'cyberpunk style, neon lights, futuristic, dark atmosphere' },
  { id: 'ghibli', name: 'Studio Ghibli', icon: '🏯', prompt: 'studio ghibli style, soft colors, whimsical, animated' },
  { id: 'impressionist', name: 'Impressionist', icon: '🌻', prompt: 'impressionist painting, monet style, soft brushstrokes, light and color' },
  { id: '3d_render', name: '3D Render', icon: '💎', prompt: '3d rendered, octane render, highly detailed, realistic lighting' },
] as const

type StyleId = typeof STYLE_PRESETS[number]['id']

// Available style transfer models
const STYLE_MODELS = [
  { id: 'sdxl-img2img', name: 'SDXL img2img', vram: '8-12 GB' },
  { id: 'ip-adapter', name: 'IP-Adapter (SD 1.5)', vram: '8-12 GB' },
  { id: 'sdxl-lightning', name: 'SDXL Lightning (Fast)', vram: '8-10 GB' },
] as const

type ModelId = typeof STYLE_MODELS[number]['id']

function StyleTransferTab() {
  // Image state
  const [sourceImageUrl, setSourceImageUrl] = useState<string | null>(null)
  const [sourceImageFile, setSourceImageFile] = useState<File | null>(null)

  // Style parameters
  const [selectedStyle, setSelectedStyle] = useState<StyleId | null>(null)
  const [customPrompt, setCustomPrompt] = useState('')
  const [useCustomPrompt, setUseCustomPrompt] = useState(false)
  const [strength, setStrength] = useState(0.7)
  const [selectedModel, setSelectedModel] = useState<ModelId>('sdxl-img2img')

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

  const handleStyleSelect = (styleId: StyleId) => {
    setSelectedStyle(styleId)
    setUseCustomPrompt(false)
  }

  const getActivePrompt = (): string => {
    if (useCustomPrompt) return customPrompt
    const style = STYLE_PRESETS.find(s => s.id === selectedStyle)
    return style?.prompt || ''
  }

  const canGenerate = sourceImageUrl && (selectedStyle || (useCustomPrompt && customPrompt.trim()))

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

      console.log('Style Transfer params:', {
        model: selectedModel,
        style: selectedStyle,
        prompt: getActivePrompt(),
        strength,
        hasImage: !!sourceImageFile,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during generation')
    } finally {
      setIsGenerating(false)
    }
  }

  const handleDownload = () => {
    if (!resultImageUrl) return

    const link = document.createElement('a')
    link.href = resultImageUrl
    link.download = `styled-${selectedStyle || 'custom'}-${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Style Transfer</h2>
      <p className="text-slate-400 mb-6">
        Transform your images into different artistic styles. Upload an image, choose a style, and generate.
      </p>

      {/* Step 1: Upload image */}
      {!sourceImageUrl && (
        <ImageUpload
          label="Source Image"
          onImageSelect={handleSourceImageSelect}
          onImageClear={handleImageClear}
        />
      )}

      {/* Step 2: Select style and configure */}
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

          {/* Right column: Style selection and controls */}
          <div className="space-y-6">
            {/* Style presets grid */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Select Style
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {STYLE_PRESETS.map((style) => (
                  <button
                    key={style.id}
                    onClick={() => handleStyleSelect(style.id)}
                    className={`p-3 rounded-lg border-2 transition-all duration-200
                                flex flex-col items-center gap-1 text-center
                                ${selectedStyle === style.id && !useCustomPrompt
                                  ? 'border-indigo-500 bg-indigo-500/20'
                                  : 'border-slate-700 bg-slate-800 hover:border-slate-600'
                                }`}
                  >
                    <span className="text-2xl">{style.icon}</span>
                    <span className="text-xs text-slate-300">{style.name}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Custom prompt toggle and input */}
            <div>
              <div className="flex items-center gap-3 mb-2">
                <input
                  type="checkbox"
                  id="useCustomPrompt"
                  checked={useCustomPrompt}
                  onChange={(e) => setUseCustomPrompt(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-600 bg-slate-800
                             text-indigo-600 focus:ring-indigo-500"
                />
                <label htmlFor="useCustomPrompt" className="text-sm font-medium text-slate-300">
                  Use custom style description
                </label>
              </div>

              {useCustomPrompt && (
                <textarea
                  value={customPrompt}
                  onChange={(e) => setCustomPrompt(e.target.value)}
                  placeholder="Describe the style you want... (e.g., 'van gogh starry night style, swirling brushstrokes')"
                  rows={3}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg
                             text-slate-200 placeholder-slate-500 focus:outline-none
                             focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500
                             resize-none"
                />
              )}
            </div>

            {/* Strength slider */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-medium text-slate-300">
                  Style Strength
                </label>
                <span className="text-sm text-slate-400">{Math.round(strength * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.3"
                max="0.9"
                step="0.05"
                value={strength}
                onChange={(e) => setStrength(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
              <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>Subtle</span>
                <span>Strong</span>
              </div>
            </div>

            {/* Model selection */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value as ModelId)}
                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg
                           text-slate-200 focus:outline-none focus:border-indigo-500
                           focus:ring-1 focus:ring-indigo-500"
              >
                {STYLE_MODELS.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({model.vram})
                  </option>
                ))}
              </select>
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
                  Generating...
                </>
              ) : (
                'Generate'
              )}
            </button>

            {/* Validation hint */}
            {!selectedStyle && !useCustomPrompt && (
              <p className="text-amber-400 text-sm">
                Select a style preset or enable custom style description.
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
              <p className="text-slate-400 text-sm mb-2">
                Styled ({selectedStyle ? STYLE_PRESETS.find(s => s.id === selectedStyle)?.name : 'Custom'})
              </p>
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

export default StyleTransferTab
