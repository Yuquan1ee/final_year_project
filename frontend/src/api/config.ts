/**
 * API Configuration
 *
 * Configure the backend URL here. For local development, use localhost.
 * For cloud deployment (Colab, AWS), update this to the public URL.
 */

// Backend API base URL
// Change this when deploying to cloud (e.g., ngrok URL from Colab)
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Convert a File to base64 string
 */
export async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const result = reader.result as string;
      // Remove the data URL prefix (e.g., "data:image/png;base64,")
      // Backend expects raw base64, but we'll keep the prefix for compatibility
      resolve(result);
    };
    reader.onerror = (error) => reject(error);
  });
}

/**
 * Convert base64 string to data URL for display
 */
export function base64ToDataUrl(base64: string, mimeType: string = 'image/png'): string {
  // If it already has the data URL prefix, return as-is
  if (base64.startsWith('data:')) {
    return base64;
  }
  return `data:${mimeType};base64,${base64}`;
}
