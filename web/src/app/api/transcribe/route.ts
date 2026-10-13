import { auth } from "@/lib/auth";
import { apiFormRequest } from "@/lib/api-client";
import { prisma } from "@/lib/prisma";
import { TranscriptionResponse } from "@/lib/types";
import { NextRequest, NextResponse } from "next/server";
import { revalidatePath } from "next/cache";

/**
 * POST /api/transcribe
 * Direct transcription (no session).
 */
export async function POST(request: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "File is required" }, { status: 400 });
    }

    // Proxy to Python API
    let apiResponse: TranscriptionResponse;
    try {
      apiResponse = await apiFormRequest<TranscriptionResponse>(
        "/transcribe",
        formData
      );
    } catch (error: any) {
      console.error("Python API transcription error:", error);
      return NextResponse.json(
        { error: `Transcription failed: ${error.message}` },
        { status: 500 }
      );
    }

    // Validate response
    if (!apiResponse.text) {
      return NextResponse.json(
        { error: "API returned empty transcription" },
        { status: 500 }
      );
    }

    // Save to DB
    const transcription = await prisma.transcription.create({
      data: {
        userId: session.user.id,
        fileName: file.name,
        text: apiResponse.text,
        segments: apiResponse.segments as any,
      },
    });

    revalidatePath("/dashboard");

    return NextResponse.json(transcription);
  } catch (error: any) {
    console.error("Transcription error:", error);
    return NextResponse.json(
      { error: error.message || "Internal Server Error" },
      { status: 500 }
    );
  }
}

/**
 * GET /api/transcriptions
 */
export async function GET() {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    try {
      const transcriptions = await prisma.transcription.findMany({
        where: { userId: session.user.id },
        orderBy: { createdAt: "desc" },
      });

      return NextResponse.json(transcriptions);
    } catch (error: any) {
      console.error("Error fetching transcriptions:", error);
      return NextResponse.json(
        { error: "Error fetching transcriptions" },
        { status: 500 }
      );
    }
}
