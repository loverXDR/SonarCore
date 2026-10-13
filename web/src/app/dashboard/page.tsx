import { auth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { redirect } from "next/navigation";
import DashboardClient from "./DashboardClient";

export default async function DashboardPage() {
  const session = await auth();

  if (!session || !session.user?.id) {
    redirect("/login");
  }

  const userId = session.user.id;

  const userSessions = await prisma.userSession.findMany({
    where: { userId },
    orderBy: { updatedAt: "desc" },
    include: {
      _count: {
        select: { messages: true },
      },
      messages: {
        orderBy: { createdAt: "desc" },
        take: 1,
        select: { content: true },
      },
    },
  });

  const transcriptions = await prisma.transcription.findMany({
    where: { userId },
    orderBy: { createdAt: "desc" },
    take: 5,
  });

  // Serialize dates for client component
  const serializedSessions = userSessions.map((s) => ({
    id: s.id,
    title: s.title,
    sourceType: s.sourceType,
    updatedAt: s.updatedAt.toISOString(),
    _count: s._count,
    lastMessage: s.messages[0]?.content || null,
  }));

  const serializedTranscriptions = transcriptions.map((t) => ({
    id: t.id,
    fileName: t.fileName,
    text: typeof t.text === "string" ? t.text : "",
    createdAt: t.createdAt.toISOString(),
  }));

  const userName = session.user?.name || "Пользователь";

  return (
    <DashboardClient
      userName={userName}
      sessions={serializedSessions}
      transcriptions={serializedTranscriptions}
    />
  );
}
