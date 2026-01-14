import { Download, Clock, Cpu } from 'lucide-react';
import type { ImageResponse } from '../types';
import { base64ToDataUrl } from '../services/api';

interface ResultDisplayProps {
  result: ImageResponse | null;
  isLoading: boolean;
}

export function ResultDisplay({ result, isLoading }: ResultDisplayProps) {
  const handleDownload = () => {
    if (!result?.image) return;

    const link = document.createElement('a');
    link.href = base64ToDataUrl(result.image);
    link.download = `result_${Date.now()}.png`;
    link.click();
  };

  if (isLoading) {
    return (
      <div className="result-display loading">
        <div className="spinner"></div>
        <p>Generating image...</p>
        <p className="hint">This may take 30-60 seconds</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-display empty">
        <p>Result will appear here</p>
      </div>
    );
  }

  if (!result.success) {
    return (
      <div className="result-display error">
        <p>Error: {result.error}</p>
      </div>
    );
  }

  return (
    <div className="result-display success">
      {result.image && (
        <>
          <img src={base64ToDataUrl(result.image)} alt="Generated result" className="result-image" />
          <div className="result-info">
            {result.processing_time && (
              <span className="info-item">
                <Clock size={14} />
                {result.processing_time.toFixed(1)}s
              </span>
            )}
            {result.model_used && (
              <span className="info-item">
                <Cpu size={14} />
                {result.model_used}
              </span>
            )}
            <button onClick={handleDownload} className="download-btn">
              <Download size={14} />
              Download
            </button>
          </div>
        </>
      )}
    </div>
  );
}
