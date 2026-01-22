import { useState, useRef } from 'react'
import type { DragEvent, ChangeEvent } from 'react'

interface ImageUploadProps {
  onImageSelect: (file: File, previewUrl: string) => void
  onImageClear?: () => void
  acceptedTypes?: string[]
  maxSizeMB?: number
  label?: string
  className?: string
}

/**
 * Reusable Image Upload Component
 *
 * Features:
 * - Drag and drop support (desktop)
 * - Click to select (desktop + mobile)
 * - Image preview
 * - File type and size validation
 * - Mobile camera/gallery support via accept attribute
 */
function ImageUpload({
  onImageSelect,
  onImageClear,
  acceptedTypes = ['image/jpeg', 'image/png', 'image/webp'],
  maxSizeMB = 10,
  label = 'Upload Image',
  className = '',
}: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const maxSizeBytes = maxSizeMB * 1024 * 1024

  const validateFile = (file: File): string | null => {
    if (!acceptedTypes.includes(file.type)) {
      return `Invalid file type. Accepted: ${acceptedTypes.map(t => t.split('/')[1]).join(', ')}`
    }
    if (file.size > maxSizeBytes) {
      return `File too large. Maximum size: ${maxSizeMB}MB`
    }
    return null
  }

  const handleFile = (file: File) => {
    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      return
    }

    setError(null)
    const previewUrl = URL.createObjectURL(file)
    setPreview(previewUrl)
    onImageSelect(file, previewUrl)
  }

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFile(files[0])
    }
  }

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFile(files[0])
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const handleClear = () => {
    setPreview(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    onImageClear?.()
  }

  return (
    <div className={className}>
      <label className="block text-sm font-medium text-slate-300 mb-2">
        {label}
      </label>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={acceptedTypes.join(',')}
        onChange={handleInputChange}
        className="hidden"
      />

      {!preview ? (
        // Upload area
        <div
          onClick={handleClick}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
            transition-colors duration-200
            ${isDragging
              ? 'border-indigo-500 bg-indigo-500/10'
              : 'border-slate-600 hover:border-slate-500 hover:bg-slate-800/50'
            }
          `}
        >
          {/* Upload icon */}
          <svg
            className="mx-auto h-12 w-12 text-slate-500 mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>

          <p className="text-slate-300 mb-1">
            <span className="text-indigo-400 font-medium">Click to upload</span>
            {' '}or drag and drop
          </p>
          <p className="text-slate-500 text-sm">
            {acceptedTypes.map(t => t.split('/')[1].toUpperCase()).join(', ')} up to {maxSizeMB}MB
          </p>
        </div>
      ) : (
        // Preview area
        <div className="relative rounded-lg overflow-hidden bg-slate-800">
          <img
            src={preview}
            alt="Preview"
            className="w-full h-auto max-h-80 object-contain"
          />

          {/* Clear button */}
          <button
            onClick={handleClear}
            className="absolute top-2 right-2 p-2 bg-slate-900/80 hover:bg-red-600
                       rounded-full transition-colors"
            title="Remove image"
          >
            <svg
              className="h-5 w-5 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>

          {/* Replace button */}
          <button
            onClick={handleClick}
            className="absolute bottom-2 right-2 px-3 py-1.5 bg-slate-900/80
                       hover:bg-indigo-600 rounded-lg text-sm transition-colors"
          >
            Replace
          </button>
        </div>
      )}

      {/* Error message */}
      {error && (
        <p className="mt-2 text-red-400 text-sm">{error}</p>
      )}
    </div>
  )
}

export default ImageUpload
