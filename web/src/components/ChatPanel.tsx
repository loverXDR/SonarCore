"use client";

import { AgentMessage } from "@/lib/types";
import { User, Bot, Copy, MessageSquare } from "lucide-react";
import { ChatMessageSkeleton } from "@/components/Skeleton";
import styles from "./ChatPanel.module.css";
import { useEffect, useRef, useState } from "react";

interface ChatPanelProps {
  messages: AgentMessage[];
  loading: boolean;
  onCopyMessage?: (content: string) => void;
  isStreaming?: boolean;
}

function formatTimestamp(date?: string) {
  if (!date) return null;
  try {
    return new Date(date).toLocaleTimeString("ru-RU", {
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return null;
  }
}

export default function ChatPanel({
  messages,
  loading,
  onCopyMessage,
  isStreaming,
}: ChatPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const handleCopy = (content: string, idx: number) => {
    navigator.clipboard.writeText(content);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
    if (onCopyMessage) onCopyMessage(content);
  };

  return (
    <div className={styles.container} ref={scrollRef}>
      {messages.length === 0 && !loading && (
        <div className={styles.welcome}>
          <div className={styles.welcomeIcon}>
            <MessageSquare size={48} />
          </div>
          <h2>Это начало вашего чата</h2>
          <p>Задайте любой вопрос по содержимому транскрибации или текста.</p>
        </div>
      )}

      <div className={styles.messageList}>
        {messages.map((msg: AgentMessage & { createdAt?: string }, idx: number) => {
          const isUser = msg.role === "user";
          const isAssistant = msg.role === "assistant";
          const timestamp = formatTimestamp(msg.createdAt);
          const isLast = idx === messages.length - 1;

          return (
            <div
              key={idx}
              className={`${styles.message} ${
                isUser ? styles.user : styles.assistant
              }`}
            >
              <div className={`${styles.avatar} ${isUser ? styles.avatarUser : styles.avatarBot}`}>
                {isUser ? <User size={16} /> : <Bot size={16} />}
              </div>
              <div className={styles.bubble}>
                <div className={styles.roleLine}>
                  <span className={styles.role}>
                    {isUser ? "Вы" : "Sonar AI"}
                  </span>
                  {timestamp && (
                    <span className={styles.timestamp}>{timestamp}</span>
                  )}
                </div>
                <div className={styles.content}>
                  {msg.content}
                  {isStreaming && isAssistant && isLast && (
                    <span className={styles.cursor}>|</span>
                  )}
                </div>
                {isAssistant && (
                  <button
                    className={styles.copyBtn}
                    onClick={() => handleCopy(msg.content, idx)}
                    aria-label="Копировать сообщение"
                  >
                    <Copy size={13} />
                    {copiedIdx === idx ? "Скопировано" : "Копировать"}
                  </button>
                )}
              </div>
            </div>
          );
        })}
        {loading && (
          <ChatMessageSkeleton role="assistant" />
        )}
      </div>
    </div>
  );
}
