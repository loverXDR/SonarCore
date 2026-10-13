/**
 * Shared TypeScript types for the frontend application.
 * Mirrors the Python API schemas for type safety.
 */

/** Transcription segment with timestamps */
export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
  speaker?: string | null;
}

/** Response from the transcription endpoint */
export interface TranscriptionResponse {
  text: string;
  segments: TranscriptionSegment[];
}

/** Response when creating a session via the API */
export interface SessionResponse {
  session_id: string;
  message: string;
}

/** Chat message in a conversation */
export interface AgentMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

/** Response from the chat endpoint */
export interface ChatMessageResponse {
  answer: string;
  sources: Record<string, unknown>[];
  history: AgentMessage[];
}

/** Health check response */
export interface HealthCheckResponse {
  status: string;
  version: string;
  uptime_seconds: number;
  dependencies: Record<string, string>;
}

/** Session info from the API */
export interface SessionInfo {
  session_id: string;
  created_at: string;
  last_active: string;
  message_count: number;
}

/** Frontend session with DB data */
export interface UserSessionData {
  id: string;
  apiSessionId: string | null;
  title: string;
  sourceType: string;
  status: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}

/** Frontend transcription data */
export interface TranscriptionData {
  id: string;
  fileName: string | null;
  text: string;
  segments: TranscriptionSegment[];
  createdAt: string;
}
