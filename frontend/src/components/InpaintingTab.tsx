import { useState } from 'react';
import { ImageUpload } from './ImageUpload';
import { ResultDisplay } from './ResultDisplay';
import { inpaintImage, fileToBase64, base64ToDataUrl } from '../services/api';
import type { ImageResponse } from '../types';

export function InpaintingTab() {
  const [image, setImage] = useState<string>('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [mask, setMask] = useState<string>('');
  const [maskFile, setMaskFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('blurry, low quality, distorted');
  const [model, setModel] = useState<'sd-inpainting' | 'sdxl-inpainting'>('sd-inpainting');
  const [result, setResult] = useState<ImageResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleImageSelect = async (file: File) => {
    setImageFile(file);
    const base64 = await fileToBase64(file);
    setImage(base64);
  };

  const handleMaskSelect = async (file: File) => {
    setMaskFile(file);
    const base64 = await fileToBase64(file);
    setMask(base64);
  };

  const handleSubmit = async () => {
    if (!image || !mask || !prompt) {
      alert('Please provide an image, mask, and prompt');
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const response = await inpaintImage({
        image,
        mask,
        prompt,
        negative_prompt: negativePrompt,
        model,
      });
      setResult(response);
    } catch (error) {
      setResult({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="tab-content">
      <div className="input-section">
        <div className="image-inputs">
          <ImageUpload
            label="Input Image"
            onImageSelect={handleImageSelect}
            currentImage={image ? base64ToDataUrl(image) : undefined}
          />
          <ImageUpload
            label="Mask (white = inpaint)"
            onImageSelect={handleMaskSelect}
            currentImage={mask ? base64ToDataUrl(mask) : undefined}
          />
        </div>

        <div className="form-group">
          <label>Prompt</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe what to generate in the masked area..."
            rows={3}
          />
        </div>

        <div className="form-group">
          <label>Negative Prompt</label>
          <input
            type="text"
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            placeholder="What to avoid..."
          />
        </div>

        <div className="form-group">
          <label>Model</label>
          <select value={model} onChange={(e) => setModel(e.target.value as typeof model)}>
            <option value="sd-inpainting">Stable Diffusion Inpainting</option>
            <option value="sdxl-inpainting">SDXL Inpainting (higher quality)</option>
          </select>
        </div>

        <button onClick={handleSubmit} disabled={isLoading || !image || !mask || !prompt} className="submit-btn">
          {isLoading ? 'Processing...' : 'Inpaint'}
        </button>
      </div>

      <div className="result-section">
        <h3>Result</h3>
        <ResultDisplay result={result} isLoading={isLoading} />
      </div>
    </div>
  );
}
