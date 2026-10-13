import { auth } from "@/lib/auth";
import { apiFormRequest } from "@/lib/api-client";
import { prisma } from "@/lib/prisma";
import { SessionResponse } from "@/lib/types";
import { NextRequest, NextResponse } from "next/server";
import { revalidatePath } from "next/cache";

/**
 * POST /api/sessions/audio
 * Create a session from an audio file.
 */
export async function POST(request: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const title = (formData.get("title") as string) || "Аудио сессия";

    if (!file) {
      return NextResponse.json({ error: "Audio file is required" }, { status: 400 });
    }

    // 1. Transcribe audio via Python API
    let transcriptionResponse: any;
    try {
      transcriptionResponse = await apiFormRequest<{text: string; segments: any[]}>(
        "/transcribe",
        formData
      );
    } catch (error: any) {
      console.error("Transcription failed:", error);
      return NextResponse.json(
        { error: `Transcription failed: ${error.message}` },
        { status: 500 }
      );
    }

    const text = transcriptionResponse.text;
    if (!text || typeof text !== "string") {
      return NextResponse.json(
        { error: "Transcription returned no text" },
        { status: 500 }
      );
    }

    // 2. Create session from transcribed text
    let apiResponse: SessionResponse;
    try {
      const apiRes = await fetch(`${process.env.SONARCORE_API_URL}/sessions/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!apiRes.ok) {
        throw new Error(`API returned ${apiRes.status}`);
      }
      apiResponse = await apiRes.json();
    } catch (error: any) {
      console.error("Session creation failed:", error);
      return NextResponse.json(
        { error: `Session creation failed: ${error.message}` },
        { status: 500 }
      );
    }

    // 3. Save to DB
    const userSession = await prisma.userSession.create({
      data: {
        userId: session.user.id,
        apiSessionId: apiResponse.session_id,
        title: title || file.name,
        sourceType: "audio",
        sourceText: JSON.stringify(transcriptionResponse.segments || []),
      },
    });

    revalidatePath("/dashboard");

    return NextResponse.json({
      id: userSession.id,
      apiSessionId: apiResponse.session_id,
      message: "Audio session created successfully",
    });

  } catch (error: any) {
    console.error("Audio session error:", error);
    return NextResponse.json(
      { error: error.message || "Internal Server Error" },
      { status: 500 }
    );
  }
}
