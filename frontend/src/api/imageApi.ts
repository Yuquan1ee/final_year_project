/**
 * Image Processing API Client
 *
 * Handles all communication with the backend API.
 */

import { API_BASE_URL, fileToBase64, base64ToDataUrl } from './config';

// =============================================================================
// Types
// =============================================================================

export interface ImageResponse {
  success: boolean;
  image?: string;
  error?: string;
  model_used?: string;
  processing_time?: number;
}

export interface InpaintingParams {
  image: File;
  mask: string; // base64 data URL from canvas
  prompt: string;
  negativePrompt?: string;
  model?: string;
  guidanceScale?: number;
  numInferenceSteps?: number;
}

export interface StyleTransferParams {
  image: File;
  style: string;
  model?: string;
  strength?: number;
}

export interface RestorationParams {
  image: File;
  enableFaceEnhance?: boolean;
  faceModel?: 'codeformer' | 'gfpgan';
  fidelity?: number;
  upscale?: 'none' | '2x' | '4x';
  enableScratchRemoval?: boolean;
  enableColorize?: boolean;
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * Check API health and GPU status
 */
export async function checkHealth(): Promise<{
  status: string;
  gpu: {
    available: boolean;
    device_name?: string;
    total_memory_gb?: number;
  };
  loaded_pipelines: string[];
}> {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Perform inpainting on an image
 */
export async function inpaintImage(params: InpaintingParams): Promise<ImageResponse> {
  const imageBase64 = await fileToBase64(params.image);

  const response = await fetch(`${API_BASE_URL}/api/inpainting/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: imageBase64,
      mask: params.mask,
      prompt: params.prompt,
      negative_prompt: params.negativePrompt || 'blurry, low quality, distorted, deformed',
      model: params.model || 'sd-inpainting',
      guidance_scale: params.guidanceScale || 7.5,
      num_inference_steps: params.numInferenceSteps || 30,
    }),
  });

  if (!response.ok) {
    throw new Error(`Inpainting request failed: ${response.statusText}`);
  }

  const data: ImageResponse = await response.json();

  // Convert base64 to data URL for display
  if (data.success && data.image) {
    data.image = base64ToDataUrl(data.image);
  }

  return data;
}

/**
 * Apply style transfer to an image
 */
export async function styleTransfer(params: StyleTransferParams): Promise<ImageResponse> {
  const imageBase64 = await fileToBase64(params.image);

  const response = await fetch(`${API_BASE_URL}/api/style/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: imageBase64,
      style: params.style,
      model: params.model || 'sdxl-img2img',
      strength: params.strength || 0.6,
    }),
  });

  if (!response.ok) {
    throw new Error(`Style transfer request failed: ${response.statusText}`);
  }

  const data: ImageResponse = await response.json();

  if (data.success && data.image) {
    data.image = base64ToDataUrl(data.image);
  }

  return data;
}

/**
 * Restore an image (face enhancement, upscaling, etc.)
 */
export async function restoreImage(params: RestorationParams): Promise<ImageResponse> {
  const imageBase64 = await fileToBase64(params.image);

  const response = await fetch(`${API_BASE_URL}/api/restoration/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: imageBase64,
      enable_face_enhance: params.enableFaceEnhance ?? true,
      face_model: params.faceModel || 'codeformer',
      fidelity: params.fidelity ?? 0.5,
      upscale: params.upscale || '2x',
      enable_scratch_removal: params.enableScratchRemoval ?? false,
      enable_colorize: params.enableColorize ?? false,
    }),
  });

  if (!response.ok) {
    throw new Error(`Restoration request failed: ${response.statusText}`);
  }

  const data: ImageResponse = await response.json();

  if (data.success && data.image) {
    data.image = base64ToDataUrl(data.image);
  }

  return data;
}

/**
 * Get available inpainting models
 */
export async function getInpaintingModels(): Promise<{
  models: Array<{ key: string; name: string; description: string; vram: string }>;
}> {
  const response = await fetch(`${API_BASE_URL}/api/inpainting/models`);
  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get available style presets
 */
export async function getStylePresets(): Promise<{
  presets: Array<{ key: string; name: string; prompt: string }>;
}> {
  const response = await fetch(`${API_BASE_URL}/api/style/presets`);
  if (!response.ok) {
    throw new Error(`Failed to fetch presets: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get restoration options
 */
export async function getRestorationOptions(): Promise<{
  face_models: string[];
  upscale_options: string[];
  defaults: Record<string, unknown>;
}> {
  const response = await fetch(`${API_BASE_URL}/api/restoration/options`);
  if (!response.ok) {
    throw new Error(`Failed to fetch options: ${response.statusText}`);
  }
  return response.json();
}
