import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon } from 'lucide-react';

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  currentImage?: string;
  label?: string;
}

export function ImageUpload({ onImageSelect, currentImage, label = 'Upload Image' }: ImageUploadProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onImageSelect(acceptedFiles[0]);
      }
    },
    [onImageSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp'],
    },
    maxFiles: 1,
  });

  return (
    <div className="image-upload">
      <label className="label">{label}</label>
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${currentImage ? 'has-image' : ''}`}
      >
        <input {...getInputProps()} />
        {currentImage ? (
          <img src={currentImage} alt="Uploaded" className="preview-image" />
        ) : (
          <div className="placeholder">
            {isDragActive ? (
              <>
                <Upload size={48} />
                <p>Drop image here</p>
              </>
            ) : (
              <>
                <ImageIcon size={48} />
                <p>Drag & drop an image, or click to select</p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
