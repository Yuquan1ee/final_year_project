import { useRef, useEffect, useState, useCallback } from 'react'
import type { MouseEvent, TouchEvent } from 'react'

interface MaskCanvasProps {
  imageUrl: string
  onMaskChange?: (maskDataUrl: string | null) => void
  className?: string
}

type Tool = 'brush' | 'eraser'

/**
 * MaskCanvas Component
 *
 * A canvas overlay for drawing masks on images.
 * Features:
 * - Brush tool to paint mask areas
 * - Eraser tool to remove mask areas
 * - Adjustable brush size
 * - Touch support for mobile devices
 * - Export mask as data URL
 */
function MaskCanvas({ imageUrl, onMaskChange, className = '' }: MaskCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [tool, setTool] = useState<Tool>('brush')
  const [brushSize, setBrushSize] = useState(30)
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 })
  const [canvasScale, setCanvasScale] = useState(1)
  const lastPosRef = useRef<{ x: number; y: number } | null>(null)

  // Mask color - semi-transparent red for visibility
  const MASK_COLOR = 'rgba(255, 0, 0, 0.5)'

  // Load image and set canvas dimensions
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      setImageDimensions({ width: img.width, height: img.height })

      const canvas = canvasRef.current
      if (canvas) {
        canvas.width = img.width
        canvas.height = img.height

        // Calculate scale for coordinate mapping
        if (containerRef.current) {
          const containerWidth = containerRef.current.clientWidth
          const scale = containerWidth / img.width
          setCanvasScale(scale)
        }
      }
    }
    img.src = imageUrl
  }, [imageUrl])

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && imageDimensions.width > 0) {
        const containerWidth = containerRef.current.clientWidth
        const scale = containerWidth / imageDimensions.width
        setCanvasScale(scale)
      }
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [imageDimensions.width])

  // Get canvas coordinates from event
  const getCanvasCoordinates = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current
      if (!canvas) return { x: 0, y: 0 }

      const rect = canvas.getBoundingClientRect()
      const x = (clientX - rect.left) / canvasScale
      const y = (clientY - rect.top) / canvasScale

      return { x, y }
    },
    [canvasScale]
  )

  // Draw on canvas
  const draw = useCallback(
    (x: number, y: number) => {
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')
      if (!ctx) return

      ctx.beginPath()
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.lineWidth = brushSize

      if (tool === 'brush') {
        ctx.globalCompositeOperation = 'source-over'
        ctx.strokeStyle = MASK_COLOR
      } else {
        ctx.globalCompositeOperation = 'destination-out'
        ctx.strokeStyle = 'rgba(0,0,0,1)'
      }

      // Draw line from last position to current
      if (lastPosRef.current) {
        ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y)
        ctx.lineTo(x, y)
        ctx.stroke()
      } else {
        // Draw a dot if no previous position
        ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2)
        ctx.fill()
      }

      lastPosRef.current = { x, y }
    },
    [tool, brushSize]
  )

  // Export mask as black and white image
  const exportMask = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return null

    // Create a temporary canvas for the mask
    const maskCanvas = document.createElement('canvas')
    maskCanvas.width = canvas.width
    maskCanvas.height = canvas.height
    const maskCtx = maskCanvas.getContext('2d')
    if (!maskCtx) return null

    // Fill with black (areas to keep)
    maskCtx.fillStyle = 'black'
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height)

    // Get the drawing data
    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)

    // Convert: any non-transparent pixel becomes white (area to inpaint)
    for (let i = 0; i < imageData.data.length; i += 4) {
      if (imageData.data[i + 3] > 0) {
        // If alpha > 0, it's part of the mask
        maskImageData.data[i] = 255 // R
        maskImageData.data[i + 1] = 255 // G
        maskImageData.data[i + 2] = 255 // B
        maskImageData.data[i + 3] = 255 // A
      }
    }

    maskCtx.putImageData(maskImageData, 0, 0)
    return maskCanvas.toDataURL('image/png')
  }, [])

  // Notify parent of mask changes
  const notifyMaskChange = useCallback(() => {
    if (onMaskChange) {
      const maskDataUrl = exportMask()
      onMaskChange(maskDataUrl)
    }
  }, [onMaskChange, exportMask])

  // Mouse event handlers
  const handleMouseDown = (e: MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    setIsDrawing(true)
    const { x, y } = getCanvasCoordinates(e.clientX, e.clientY)
    lastPosRef.current = null
    draw(x, y)
  }

  const handleMouseMove = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return
    const { x, y } = getCanvasCoordinates(e.clientX, e.clientY)
    draw(x, y)
  }

  const handleMouseUp = () => {
    if (isDrawing) {
      setIsDrawing(false)
      lastPosRef.current = null
      notifyMaskChange()
    }
  }

  // Touch event handlers
  const handleTouchStart = (e: TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    if (e.touches.length === 1) {
      setIsDrawing(true)
      const touch = e.touches[0]
      const { x, y } = getCanvasCoordinates(touch.clientX, touch.clientY)
      lastPosRef.current = null
      draw(x, y)
    }
  }

  const handleTouchMove = (e: TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    if (!isDrawing || e.touches.length !== 1) return
    const touch = e.touches[0]
    const { x, y } = getCanvasCoordinates(touch.clientX, touch.clientY)
    draw(x, y)
  }

  const handleTouchEnd = () => {
    if (isDrawing) {
      setIsDrawing(false)
      lastPosRef.current = null
      notifyMaskChange()
    }
  }

  // Clear the canvas
  const handleClear = () => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      notifyMaskChange()
    }
  }

  return (
    <div className={className}>
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-4 mb-4 p-3 bg-slate-800 rounded-lg">
        {/* Tool selection */}
        <div className="flex gap-2">
          <button
            onClick={() => setTool('brush')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              tool === 'brush'
                ? 'bg-indigo-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Brush
          </button>
          <button
            onClick={() => setTool('eraser')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              tool === 'eraser'
                ? 'bg-indigo-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Eraser
          </button>
        </div>

        {/* Brush size slider */}
        <div className="flex items-center gap-3">
          <label className="text-slate-300 text-sm">Size:</label>
          <input
            type="range"
            min="5"
            max="100"
            value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
            className="w-24 accent-indigo-500"
          />
          <span className="text-slate-400 text-sm w-8">{brushSize}</span>
        </div>

        {/* Clear button */}
        <button
          onClick={handleClear}
          className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg
                     font-medium transition-colors ml-auto"
        >
          Clear
        </button>
      </div>

      {/* Canvas container */}
      <div
        ref={containerRef}
        className="relative rounded-lg overflow-hidden bg-slate-800"
        style={{ maxWidth: '100%' }}
      >
        {/* Background image */}
        <img
          src={imageUrl}
          alt="Source"
          className="w-full h-auto block"
          draggable={false}
        />

        {/* Drawing canvas overlay */}
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          className="absolute top-0 left-0 w-full h-full cursor-crosshair"
          style={{ touchAction: 'none' }}
        />
      </div>

      {/* Instructions */}
      <p className="text-slate-500 text-sm mt-2">
        Paint over the areas you want to edit. Use the eraser to fix mistakes.
      </p>
    </div>
  )
}

export default MaskCanvas
