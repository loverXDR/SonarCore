/**
 * API Client for the SonarCore Python Backend.
 *
 * Server-side fetch wrapper used by Next.js API routes
 * to proxy requests to the Python FastAPI service.
 */

const API_BASE_URL =
  process.env.SONARCORE_API_URL || "http://localhost:8000";

/**
 * Makes a request to the SonarCore Python API.
 *
 * @param endpoint - API endpoint path (e.g., "/sessions")
 * @param options  - Standard fetch options
 * @returns Parsed JSON response
 * @throws Error with status code and message on failure
 */
export async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      detail: response.statusText,
    }));
    throw new Error(error.detail || `API Error: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

/**
 * Makes a multipart form request to the SonarCore Python API.
 * Used for file uploads (audio transcription).
 *
 * @param endpoint - API endpoint path
 * @param formData - FormData with file/fields
 * @returns Parsed JSON response
 */
export async function apiFormRequest<T>(
  endpoint: string,
  formData: FormData
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    method: "POST",
    body: formData,
    // Don't set Content-Type — browser sets it with boundary
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      detail: response.statusText,
    }));
    throw new Error(error.detail || `API Error: ${response.status}`);
  }

  return response.json() as Promise<T>;
}
