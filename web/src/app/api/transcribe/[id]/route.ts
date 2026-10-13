import { auth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { NextRequest, NextResponse } from "next/server";
import { revalidatePath } from "next/cache";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await auth();
  const { id } = await params;

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const transcription = await prisma.transcription.findUnique({
      where: { id, userId: session.user.id }
    });

    if (!transcription) {
      return NextResponse.json({ error: "Transcription not found" }, { status: 404 });
    }

    return NextResponse.json(transcription);
  } catch (error: any) {
    console.error("Fetch transcription error:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await auth();
  const { id } = await params;

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const body = await request.json();
    const { text, segments } = body;

    if (!text) {
      return NextResponse.json({ error: "Text is required" }, { status: 400 });
    }

    const updated = await prisma.transcription.update({
      where: { id, userId: session.user.id },
      data: {
        text,
        segments: segments || [],
      }
    });

    revalidatePath("/transcribe");

    return NextResponse.json(updated);
  } catch (error: any) {
    console.error("Update transcription error:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
