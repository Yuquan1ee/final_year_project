// API Types

export interface ImageResponse {
  success: boolean;
  image?: string;
  error?: string;
  model_used?: string;
  processing_time?: number;
}

export interface InpaintingRequest {
  image: string;
  mask: string;
  prompt: string;
  negative_prompt?: string;
  model?: 'sd-inpainting' | 'sdxl-inpainting' | 'kandinsky-inpainting';
  guidance_scale?: number;
  num_inference_steps?: number;
}

export interface EditingRequest {
  image: string;
  instruction: string;
  mode?: 'instruct' | 'img2img';
  strength?: number;
  guidance_scale?: number;
  image_guidance_scale?: number;
}

export interface StyleTransferRequest {
  image: string;
  style: string;
  strength?: number;
  mode?: 'img2img' | 'controlnet' | 'ip-adapter';
}

export interface StylePreset {
  key: string;
  name: string;
  prompt: string;
}

export interface Model {
  key: string;
  name: string;
  description: string;
}
