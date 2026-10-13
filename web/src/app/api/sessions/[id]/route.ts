import { auth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { NextRequest, NextResponse } from "next/server";
import { revalidatePath } from "next/cache";

/**
 * GET /api/sessions/[id]
 * Fetch single session details + history.
 */
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
    const userSession = await prisma.userSession.findUnique({
      where: { id, userId: session.user.id },
      include: {
        messages: {
          orderBy: { createdAt: "asc" },
        },
      },
    });

    if (!userSession) {
      return NextResponse.json({ error: "Session not found" }, { status: 404 });
    }

    return NextResponse.json(userSession);
  } catch (error) {
    console.error("Fetch session detail error:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/sessions/[id]
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
    const session = await auth();
    const { id } = await params;

    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    try {
      await prisma.userSession.delete({
        where: { id, userId: session.user.id },
      });

      revalidatePath("/dashboard");

      return NextResponse.json({ message: "Deleted" });
  } catch (error: any) {
    console.error("Delete session error:", error);
    return NextResponse.json({ error: "Error deleting session" }, { status: 500 });
  }
}

/**
 * PATCH /api/sessions/[id]
 */
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

    const data: Record<string, unknown> = { updatedAt: new Date() };
    if (body.sourceText !== undefined) data.sourceText = body.sourceText;
    if (body.title !== undefined) data.title = body.title;

    const updated = await prisma.userSession.update({
      where: { id, userId: session.user.id },
      data,
    });

    revalidatePath(`/session/${id}`);

    return NextResponse.json(updated);
  } catch (error: any) {
    console.error("Update session error:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
