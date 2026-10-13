"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { Search, Mic, FileText, Rocket } from "lucide-react";
import SessionCard from "@/components/SessionCard";
import { SessionCardSkeleton } from "@/components/Skeleton";
import styles from "./dashboard.module.css";

interface SerializedSession {
  id: string;
  title: string;
  sourceType: string;
  updatedAt: string;
  _count: { messages: number };
  lastMessage: string | null;
}

interface SerializedTranscription {
  id: string;
  fileName: string | null;
  text: string;
  createdAt: string;
}

interface DashboardClientProps {
  userName: string;
  sessions: SerializedSession[];
  transcriptions: SerializedTranscription[];
}

type FilterTab = "all" | "audio" | "text";

export default function DashboardClient({
  userName,
  sessions: initialSessions,
  transcriptions,
}: DashboardClientProps) {
  const [sessions, setSessions] = useState(initialSessions);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<FilterTab>("all");

  const totalMessages = useMemo(
    () => sessions.reduce((acc, s) => acc + (s._count?.messages || 0), 0),
    [sessions]
  );

  const filteredSessions = useMemo(() => {
    let result = sessions;
    if (filter !== "all") {
      result = result.filter((s) => s.sourceType === filter);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter((s) => s.title.toLowerCase().includes(q));
    }
    return result;
  }, [sessions, filter, search]);

  const handleDeleteSession = (id: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Привет, {userName}</h1>
        <div className={styles.actions}>
          <Link href="/transcribe" className={styles.btnSecondary}>
            Новая транскрибация
          </Link>
          <Link href="/transcribe?createSession=true" className={styles.btn}>
            + Новая сессия
          </Link>
        </div>
      </header>

      {/* Stats row */}
      <div className={styles.stats}>
        <div className={styles.statCard}>
          <span className={styles.statValue}>{sessions.length}</span>
          <span className={styles.statLabel}>Сессий</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statValue}>{totalMessages}</span>
          <span className={styles.statLabel}>Сообщений</span>
        </div>
      </div>

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>Ваши сессии чата</h2>

        {/* Search and filter */}
        <div className={styles.toolbar}>
          <div className={styles.searchWrap}>
            <Search size={16} className={styles.searchIcon} />
            <input
              type="text"
              placeholder="Поиск сессий..."
              className={styles.searchInput}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <div className={styles.filterTabs}>
            <button
              className={`${styles.tab} ${filter === "all" ? styles.tabActive : ""}`}
              onClick={() => setFilter("all")}
            >
              Все
            </button>
            <button
              className={`${styles.tab} ${filter === "audio" ? styles.tabActive : ""}`}
              onClick={() => setFilter("audio")}
            >
              <Mic size={14} />
              Аудио
            </button>
            <button
              className={`${styles.tab} ${filter === "text" ? styles.tabActive : ""}`}
              onClick={() => setFilter("text")}
            >
              <FileText size={14} />
              Текст
            </button>
          </div>
        </div>

        <div className={styles.grid}>
          {filteredSessions.length > 0 ? (
            filteredSessions.map((s) => (
              <SessionCard
                key={s.id}
                session={{
                  ...s,
                  updatedAt: new Date(s.updatedAt),
                  lastMessage: s.lastMessage || undefined,
                }}
                onDelete={handleDeleteSession}
              />
            ))
          ) : sessions.length === 0 ? (
            <div className={styles.empty}>
              <div className={styles.emptyIcon}>
                <Rocket size={48} />
              </div>
              <h3>У вас пока нет сессий</h3>
              <p>Создайте новую сессию из аудио или текста, чтобы начать общение с ИИ.</p>
              <Link
                href="/transcribe?createSession=true"
                className={styles.btn}
                style={{ margin: "20px auto 0", width: "fit-content" }}
              >
                Создать первую сессию
              </Link>
            </div>
          ) : (
            <div className={styles.empty}>
              <h3>Ничего не найдено</h3>
              <p>Попробуйте другой запрос или сбросьте фильтры.</p>
            </div>
          )}
        </div>
      </section>

      {transcriptions.length > 0 && (
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Последние транскрибации</h2>
          <div className={styles.grid}>
            {transcriptions.map((t) => (
              <Link
                href={`/transcribe?id=${t.id}`}
                key={t.id}
                className={styles.transcriptionCard}
              >
                <div className={styles.cardInner}>
                  <div className={styles.tIconWrap}>
                    <Mic size={18} />
                  </div>
                  <div className={styles.tContent}>
                    <div className={styles.tTitle}>{t.fileName || "Без названия"}</div>
                    <div className={styles.tText}>
                      {t.text.substring(0, 100)}...
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
