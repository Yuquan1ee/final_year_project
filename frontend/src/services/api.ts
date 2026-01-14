import axios from 'axios';
import type {
  ImageResponse,
  InpaintingRequest,
  EditingRequest,
  StyleTransferRequest,
  StylePreset,
  Model,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 minute timeout for image generation
});

// Health check
export const healthCheck = async (): Promise<boolean> => {
  try {
    const response = await api.get('/api/health');
    return response.data.status === 'healthy';
  } catch {
    return false;
  }
};

// Inpainting
export const inpaintImage = async (request: InpaintingRequest): Promise<ImageResponse> => {
  const response = await api.post<ImageResponse>('/api/inpainting/', request);
  return response.data;
};

export const getInpaintingModels = async (): Promise<Model[]> => {
  const response = await api.get<{ models: Model[] }>('/api/inpainting/models');
  return response.data.models;
};

// Image Editing
export const editImage = async (request: EditingRequest): Promise<ImageResponse> => {
  const response = await api.post<ImageResponse>('/api/editing/', request);
  return response.data;
};

export const getEditingExamples = async (): Promise<{ instruction: string; category: string }[]> => {
  const response = await api.get<{ examples: { instruction: string; category: string }[] }>('/api/editing/examples');
  return response.data.examples;
};

// Style Transfer
export const applyStyle = async (request: StyleTransferRequest): Promise<ImageResponse> => {
  const response = await api.post<ImageResponse>('/api/style/', request);
  return response.data;
};

export const getStylePresets = async (): Promise<StylePreset[]> => {
  const response = await api.get<{ presets: StylePreset[] }>('/api/style/presets');
  return response.data.presets;
};

// Utility: Convert file to base64
export const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Remove data URL prefix (e.g., "data:image/png;base64,")
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

// Utility: Convert base64 to data URL for display
export const base64ToDataUrl = (base64: string, mimeType = 'image/png'): string => {
  return `data:${mimeType};base64,${base64}`;
};

export default api;
