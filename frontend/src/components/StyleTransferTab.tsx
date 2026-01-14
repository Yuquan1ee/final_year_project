import { useState } from 'react';
import { ImageUpload } from './ImageUpload';
import { ResultDisplay } from './ResultDisplay';
import { applyStyle, fileToBase64, base64ToDataUrl } from '../services/api';
import type { ImageResponse } from '../types';

const STYLE_PRESETS = [
  { key: 'oil_painting', name: 'Oil Painting' },
  { key: 'watercolor', name: 'Watercolor' },
  { key: 'anime', name: 'Anime' },
  { key: 'sketch', name: 'Pencil Sketch' },
  { key: 'cyberpunk', name: 'Cyberpunk' },
  { key: 'ghibli', name: 'Studio Ghibli' },
  { key: 'pixel_art', name: 'Pixel Art' },
  { key: 'pop_art', name: 'Pop Art' },
  { key: 'impressionist', name: 'Impressionist' },
];

export function StyleTransferTab() {
  const [image, setImage] = useState<string>('');
  const [style, setStyle] = useState('anime');
  const [customStyle, setCustomStyle] = useState('');
  const [strength, setStrength] = useState(0.6);
  const [result, setResult] = useState<ImageResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleImageSelect = async (file: File) => {
    const base64 = await fileToBase64(file);
    setImage(base64);
  };

  const handleSubmit = async () => {
    if (!image) {
      alert('Please provide an image');
      return;
    }

    const selectedStyle = style === 'custom' ? customStyle : style;
    if (!selectedStyle) {
      alert('Please select or enter a style');
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const response = await applyStyle({
        image,
        style: selectedStyle,
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
          <label>Style Preset</label>
          <div className="style-grid">
            {STYLE_PRESETS.map((preset) => (
              <button
                key={preset.key}
                className={`style-btn ${style === preset.key ? 'active' : ''}`}
                onClick={() => setStyle(preset.key)}
              >
                {preset.name}
              </button>
            ))}
            <button
              className={`style-btn ${style === 'custom' ? 'active' : ''}`}
              onClick={() => setStyle('custom')}
            >
              Custom
            </button>
          </div>
        </div>

        {style === 'custom' && (
          <div className="form-group">
            <label>Custom Style Prompt</label>
            <textarea
              value={customStyle}
              onChange={(e) => setCustomStyle(e.target.value)}
              placeholder="Describe the style (e.g., 'van gogh starry night style')"
              rows={2}
            />
          </div>
        )}

        <div className="form-group">
          <label>Strength: {strength.toFixed(2)}</label>
          <input
            type="range"
            min="0.3"
            max="0.9"
            step="0.05"
            value={strength}
            onChange={(e) => setStrength(parseFloat(e.target.value))}
          />
          <span className="hint">Lower = more original, Higher = more stylized</span>
        </div>

        <button onClick={handleSubmit} disabled={isLoading || !image} className="submit-btn">
          {isLoading ? 'Processing...' : 'Apply Style'}
        </button>
      </div>

      <div className="result-section">
        <h3>Result</h3>
        <ResultDisplay result={result} isLoading={isLoading} />
      </div>
    </div>
  );
}
