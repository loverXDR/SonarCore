"use client";

import Link from "next/link";
import { useState } from "react";
import { Music, FileText, MessageSquare, MoreVertical, Trash2 } from "lucide-react";
import styles from "./SessionCard.module.css";
import { formatDistanceToNow } from "date-fns";
import { ru } from "date-fns/locale";

interface SessionCardProps {
  session: {
    id: string;
    title: string;
    sourceType: string;
    updatedAt: Date;
    _count?: { messages: number };
    lastMessage?: string;
  };
  onDelete?: (id: string) => void;
}

export default function SessionCard({ session, onDelete }: SessionCardProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const isAudio = session.sourceType === "audio";

  const handleDelete = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDeleting(true);
    try {
      const res = await fetch(`/api/sessions/${session.id}`, {
        method: "DELETE",
      });
      if (res.ok && onDelete) {
        onDelete(session.id);
      }
    } catch {
      // silent
    } finally {
      setDeleting(false);
      setMenuOpen(false);
    }
  };

  return (
    <Link href={`/session/${session.id}`} className={styles.card}>
      <div className={styles.header}>
        <div className={`${styles.icon} ${isAudio ? styles.iconAudio : styles.iconText}`}>
          {isAudio ? <Music size={20} /> : <FileText size={20} />}
        </div>
        <div className={styles.headerRight}>
          <div className={`${styles.badge} ${isAudio ? styles.badgeAudio : ""}`}>
            {isAudio ? "Аудио" : "Текст"}
          </div>
          <div className={styles.menuWrap}>
            <button
              className={styles.menuBtn}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setMenuOpen((prev) => !prev);
              }}
              aria-label="Опции"
            >
              <MoreVertical size={16} />
            </button>
            {menuOpen && (
              <div className={styles.dropdown}>
                <button
                  className={styles.dropdownItem}
                  onClick={handleDelete}
                  disabled={deleting}
                >
                  <Trash2 size={14} />
                  {deleting ? "Удаление..." : "Удалить"}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
      <h3 className={styles.title}>{session.title}</h3>
      {session.lastMessage && (
        <p className={styles.preview}>{session.lastMessage}</p>
      )}
      <div className={styles.footer}>
        <span className={styles.messages}>
          <MessageSquare size={14} />
          {session._count?.messages || 0} сообщений
        </span>
        <span className={styles.date}>
          {formatDistanceToNow(new Date(session.updatedAt), {
            addSuffix: true,
            locale: ru,
          })}
        </span>
      </div>
    </Link>
  );
}
