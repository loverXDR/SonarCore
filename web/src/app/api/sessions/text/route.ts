import { auth } from "@/lib/auth";
import { apiRequest } from "@/lib/api-client";
import { prisma } from "@/lib/prisma";
import { SessionResponse } from "@/lib/types";
import { NextRequest, NextResponse } from "next/server";
import { revalidatePath } from "next/cache";

/**
 * POST /api/sessions/text
 * Create a new session in Python API and save to DB.
 */
export async function POST(request: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const { text, title = "Текстовая сессия" } = await request.json();

    if (!text || typeof text !== "string" || !text.trim()) {
      return NextResponse.json({ error: "Text is required" }, { status: 400 });
    }

    // 1. Call Python API
    let apiResponse: SessionResponse;
    try {
      apiResponse = await apiRequest<SessionResponse>("/sessions/text", {
        method: "POST",
        body: JSON.stringify({ text }),
      });
    } catch (error: any) {
      console.error("Python API error:", error);
      return NextResponse.json(
        { error: `Session creation failed: ${error.message}` },
        { status: 500 }
      );
    }

    // 2. Save to DB
    const userSession = await prisma.userSession.create({
      data: {
        userId: session.user.id,
        apiSessionId: apiResponse.session_id,
        title,
        sourceType: "text",
        sourceText: text,
      },
    });

    revalidatePath("/dashboard");

    return NextResponse.json({
      id: userSession.id,
      apiSessionId: apiResponse.session_id,
      message: "Session created successfully",
    });
  } catch (error: any) {
    console.error("Create text session error:", error);
    return NextResponse.json(
      { error: error.message || "Internal Server Error" },
      { status: 500 }
    );
  }
}
