import { auth } from "@/lib/auth";
import { apiRequest } from "@/lib/api-client";
import { prisma } from "@/lib/prisma";
import { ChatMessageResponse } from "@/lib/types";
import { NextRequest, NextResponse } from "next/server";
import { revalidatePath } from "next/cache";

/**
 * POST /api/sessions/[id]/chat
 * Send message to Python API and save history to DB.
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await auth();
  const { id } = await params;

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const { message } = await request.json();
    if (!message || typeof message !== "string" || !message.trim()) {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }

    // 1. Get session from DB to find apiSessionId
    const userSession = await prisma.userSession.findUnique({
      where: { id, userId: session.user.id },
    });

    if (!userSession) {
      return NextResponse.json({ error: "Session not found" }, { status: 404 });
    }

    // If apiSessionId is missing or session died in memory, fail gracefully
    if (!userSession.apiSessionId) {
       return NextResponse.json({ error: "API Session is not active" }, { status: 400 });
    }

    // 2. Chat with Python API
    let apiResponse: ChatMessageResponse;
    try {
      apiResponse = await apiRequest<ChatMessageResponse>(
        `/sessions/${userSession.apiSessionId}/chat`,
        {
          method: "POST",
          body: JSON.stringify({ message }),
        }
      );
    } catch (apiError: any) {
      console.error("Python API error:", apiError);
      return NextResponse.json(
        { error: `Chat failed: ${apiError.message}` },
        { status: 500 }
      );
    }

    // 3. Save messages to DB
    await prisma.$transaction([
      prisma.chatMessage.create({
        data: {
          userSessionId: id,
          role: "user",
          content: message,
        },
      }),
      prisma.chatMessage.create({
        data: {
          userSessionId: id,
          role: "assistant",
          content: apiResponse.answer || "",
          sources: apiResponse.sources as any,
        },
      }),
      prisma.userSession.update({
        where: { id },
        data: { updatedAt: new Date() },
      }),
    ]);

    // Invalidate cache for this session
    revalidatePath(`/session/${id}`);

    return NextResponse.json(apiResponse);
  } catch (error: any) {
    console.error("Chat error:", error);
    return NextResponse.json(
      { error: error.message || "Internal Server Error" },
      { status: 500 }
    );
  }
}
