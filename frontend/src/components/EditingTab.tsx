import { useState } from 'react';
import { ImageUpload } from './ImageUpload';
import { ResultDisplay } from './ResultDisplay';
import { editImage, fileToBase64, base64ToDataUrl } from '../services/api';
import type { ImageResponse } from '../types';

const EXAMPLE_INSTRUCTIONS = [
  'make it winter with snow',
  'make it sunset',
  'add sunglasses',
  'turn into a painting',
  'make it look vintage',
  'add rain and clouds',
];

export function EditingTab() {
  const [image, setImage] = useState<string>('');
  const [instruction, setInstruction] = useState('');
  const [mode, setMode] = useState<'instruct' | 'img2img'>('instruct');
  const [strength, setStrength] = useState(0.75);
  const [result, setResult] = useState<ImageResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleImageSelect = async (file: File) => {
    const base64 = await fileToBase64(file);
    setImage(base64);
  };

  const handleSubmit = async () => {
    if (!image || !instruction) {
      alert('Please provide an image and instruction');
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const response = await editImage({
        image,
        instruction,
        mode,
        strength,
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
        <ImageUpload
          label="Input Image"
          onImageSelect={handleImageSelect}
          currentImage={image ? base64ToDataUrl(image) : undefined}
        />

        <div className="form-group">
          <label>Edit Instruction</label>
          <textarea
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            placeholder="Describe the edit (e.g., 'make it winter')"
            rows={3}
          />
          <div className="examples">
            <span>Try: </span>
            {EXAMPLE_INSTRUCTIONS.map((ex) => (
              <button key={ex} className="example-btn" onClick={() => setInstruction(ex)}>
                {ex}
              </button>
            ))}
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Mode</label>
            <select value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}>
              <option value="instruct">InstructPix2Pix (recommended)</option>
              <option value="img2img">Img2Img</option>
            </select>
          </div>

          <div className="form-group">
            <label>Strength: {strength.toFixed(2)}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={strength}
              onChange={(e) => setStrength(parseFloat(e.target.value))}
            />
          </div>
        </div>

        <button onClick={handleSubmit} disabled={isLoading || !image || !instruction} className="submit-btn">
          {isLoading ? 'Processing...' : 'Edit Image'}
        </button>
      </div>

      <div className="result-section">
        <h3>Result</h3>
        <ResultDisplay result={result} isLoading={isLoading} />
      </div>
    </div>
  );
}
