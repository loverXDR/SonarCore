"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import ChatPanel from "@/components/ChatPanel";
import { useToast } from "@/components/Toast";
import {
  ChevronDown,
  FileText as FileTextIcon,
  Mic,
  Download,
  Pencil,
  Clipboard,
  Check,
  X,
  Send,
  RotateCcw,
} from "lucide-react";
import styles from "./session.module.css";
import { AgentMessage } from "@/lib/types";
import { useParams } from "next/navigation";

export default function SessionPage() {
  const params = useParams();
  const id = params.id as string;
  const { toast } = useToast();

  const [session, setSession] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [chatLoading, setChatLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [showTranscript, setShowTranscript] = useState(false);
  const [copied, setCopied] = useState(false);

  // Editable session title
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");

  // Transcript editing
  const [isEditing, setIsEditing] = useState(false);
  const [editedSegments, setEditedSegments] = useState<any[]>([]);
  const [editedText, setEditedText] = useState("");
  const [saving, setSaving] = useState(false);

  // Failed message for retry
  const [failedMessage, setFailedMessage] = useState<string | null>(null);

  // Scroll-to-bottom
  const chatAreaRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  // Auto-resize textarea ref
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const parseSourceText = (text: string) => {
    if (!text) return null;
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) return parsed;
    } catch {
      // Not JSON, return as string
    }
    return text;
  };

  const startEditing = () => {
    const parsed = parseSourceText(session?.sourceText);
    if (!parsed) return;
    if (Array.isArray(parsed)) {
      setEditedSegments(JSON.parse(JSON.stringify(parsed)));
      setEditedText("");
    } else {
      setEditedSegments([]);
      setEditedText(parsed);
    }
    setIsEditing(true);
  };

  const saveEdits = async () => {
    setSaving(true);
    try {
      const newSourceText =
        editedSegments.length > 0
          ? JSON.stringify(editedSegments)
          : editedText;

      const res = await fetch(`/api/sessions/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sourceText: newSourceText }),
      });

      if (!res.ok) throw new Error("Failed to save changes");
      setSession({ ...session, sourceText: newSourceText });
      setIsEditing(false);
      toast("Изменения сохранены", "success");
    } catch (e: any) {
      setError(e.message);
      toast("Ошибка при сохранении", "error");
    } finally {
      setSaving(false);
    }
  };

  const formatTime = (seconds: number) => {
    const date = new Date(seconds * 1000);
    const hh = date.getUTCHours().toString().padStart(2, "0");
    const mm = date.getUTCMinutes().toString().padStart(2, "0");
    const ss = date.getUTCSeconds().toString().padStart(2, "0");
    return `${hh === "00" ? "" : hh + ":"}${mm}:${ss}`;
  };

  const formatSRTTime = (seconds: number) => {
    const ds = seconds.toFixed(3).toString().split(".");
    const intSeconds = parseInt(ds[0], 10);
    const ms = ds[1];
    const date = new Date(intSeconds * 1000);
    const hh = date.getUTCHours().toString().padStart(2, "0");
    const mm = date.getUTCMinutes().toString().padStart(2, "0");
    const ss = date.getUTCSeconds().toString().padStart(2, "0");
    return `${hh}:${mm}:${ss},${ms}`;
  };

  const buildSRTContent = useCallback(() => {
    const parsed = parseSourceText(session?.sourceText);
    if (!Array.isArray(parsed)) return null;

    let srt = "";
    parsed.forEach((segment: any, index: number) => {
      srt += `${index + 1}\n`;
      srt += `${formatSRTTime(segment.start)} --> ${formatSRTTime(segment.end)}\n`;
      srt += `${segment.speaker ? `${segment.speaker}: ` : ""}${segment.text}\n\n`;
    });
    return srt;
  }, [session?.sourceText]);

  const srtDataUrl = useMemo(() => {
    const srt = buildSRTContent();
    if (!srt) return null;
    const blob = new Blob([srt], { type: "text/plain;charset=utf-8" });
    return URL.createObjectURL(blob);
  }, [buildSRTContent]);

  const copyText = () => {
    const parsed = parseSourceText(session?.sourceText);
    let textToCopy = "";
    if (Array.isArray(parsed)) {
      textToCopy = parsed
        .map((s: any) => `${s.speaker ? s.speaker + ": " : ""}${s.text}`)
        .join(" ");
    } else {
      textToCopy = parsed || "";
    }
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    toast("Текст скопирован", "success");
    setTimeout(() => setCopied(false), 2000);
  };

  // Save session title
  const saveTitle = async () => {
    if (!titleDraft.trim() || titleDraft === session?.title) {
      setEditingTitle(false);
      return;
    }
    try {
      const res = await fetch(`/api/sessions/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: titleDraft }),
      });
      if (res.ok) {
        setSession((prev: any) => ({ ...prev, title: titleDraft }));
        toast("Название обновлено", "success");
      }
    } catch {
      toast("Ошибка при сохранении названия", "error");
    }
    setEditingTitle(false);
  };

  // Fetch session
  useEffect(() => {
    async function fetchSession() {
      try {
        const res = await fetch(`/api/sessions/${id}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        setSession({
          ...data,
          messages: data.messages || [],
        });
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchSession();
  }, [id]);

  // Scroll monitoring for ChatPanel container
  useEffect(() => {
    const chatArea = chatAreaRef.current;
    if (!chatArea) return;
    const scrollContainer = chatArea.firstElementChild as HTMLElement | null;
    if (!scrollContainer) return;

    const handler = () => {
      const distFromBottom =
        scrollContainer.scrollHeight -
        scrollContainer.scrollTop -
        scrollContainer.clientHeight;
      setShowScrollBtn(distFromBottom > 200);
    };
    scrollContainer.addEventListener("scroll", handler);
    return () => scrollContainer.removeEventListener("scroll", handler);
  }, [session]);

  const scrollToBottom = () => {
    const chatArea = chatAreaRef.current;
    if (!chatArea) return;
    const scrollContainer = chatArea.firstElementChild as HTMLElement | null;
    if (scrollContainer) {
      scrollContainer.scrollTop = scrollContainer.scrollHeight;
    }
  };

  // Auto-resize textarea
  const autoResize = () => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    const maxRows = 5;
    const lineHeight = 22;
    const maxHeight = lineHeight * maxRows;
    ta.style.height = Math.min(ta.scrollHeight, maxHeight) + "px";
  };

  const sendMessage = async (msgToSend: string) => {
    if (!msgToSend || chatLoading) return;

    const userMsg: AgentMessage = { role: "user", content: msgToSend };
    setMessage("");
    setFailedMessage(null);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    setSession((prev: any) => ({
      ...prev,
      messages: [...(prev?.messages || []), userMsg],
    }));
    setChatLoading(true);

    try {
      const res = await fetch(`/api/sessions/${id}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msgToSend }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error);

      setSession((prev: any) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content: data.answer,
            sources: data.sources,
            createdAt: new Date().toISOString(),
          },
        ],
      }));
    } catch (err: any) {
      setFailedMessage(msgToSend);
      toast("Ошибка при отправке сообщения", "error");
    } finally {
      setChatLoading(false);
    }
  };

  const handleSend = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = message.trim();
    if (!trimmed) return;
    sendMessage(trimmed);
  };

  const handleRetry = () => {
    if (!failedMessage) return;
    // Remove the last user message (the failed one)
    setSession((prev: any) => ({
      ...prev,
      messages: prev.messages.slice(0, -1),
    }));
    sendMessage(failedMessage);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleSend();
    }
  };

  if (loading) return <div className={styles.loading}>Загрузка сессии...</div>;
  if (error && !session)
    return <div className={styles.errorBox}>Ошибка: {error}</div>;

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerInfo}>
          {editingTitle ? (
            <div className={styles.titleEditWrap}>
              <input
                className={styles.titleInput}
                value={titleDraft}
                onChange={(e) => setTitleDraft(e.target.value)}
                onBlur={saveTitle}
                onKeyDown={(e) => {
                  if (e.key === "Enter") saveTitle();
                  if (e.key === "Escape") setEditingTitle(false);
                }}
                autoFocus
              />
            </div>
          ) : (
            <h1
              className={styles.title}
              onClick={() => {
                setTitleDraft(session.title);
                setEditingTitle(true);
              }}
              title="Нажмите, чтобы переименовать"
            >
              {session.title}
              <Pencil size={14} className={styles.titlePencil} />
            </h1>
          )}
          <div className={styles.meta}>
            {session.sourceType === "audio" ? (
              <Mic size={13} />
            ) : (
              <FileTextIcon size={13} />
            )}
            <span>
              {session.sourceType === "audio" ? "Аудио" : "Текст"} {" "}
              {session.messages.length} сообщений
            </span>
          </div>
        </div>
        <button
          className={styles.transcriptBtn}
          onClick={() => setShowTranscript(true)}
        >
          <FileTextIcon size={15} />
          Показать текст
        </button>
      </header>

      <div className={styles.chatArea} ref={chatAreaRef}>
        <ChatPanel
          messages={session.messages}
          loading={chatLoading}
          onCopyMessage={() => toast("Сообщение скопировано", "success")}
        />

        {showScrollBtn && (
          <button
            className={styles.scrollBtn}
            onClick={scrollToBottom}
            aria-label="Прокрутить вниз"
          >
            <ChevronDown size={20} />
          </button>
        )}
      </div>

      {/* Error retry bubble */}
      {failedMessage && !chatLoading && (
        <div className={styles.retryBar}>
          <span>Ошибка отправки сообщения.</span>
          <button className={styles.retryBtn} onClick={handleRetry}>
            <RotateCcw size={14} />
            Повторить
          </button>
        </div>
      )}

      <footer className={styles.footer}>
        <form className={styles.inputForm} onSubmit={handleSend}>
          <textarea
            ref={textareaRef}
            className={styles.input}
            placeholder="Спросите что-нибудь... (Ctrl+Enter для отправки)"
            value={message}
            onChange={(e) => {
              setMessage(e.target.value);
              autoResize();
            }}
            onKeyDown={handleKeyDown}
            onInput={autoResize}
            disabled={chatLoading}
            rows={1}
          />
          <button
            type="submit"
            className={styles.sendBtn}
            disabled={!message.trim() || chatLoading}
            aria-label="Отправить"
          >
            <Send size={18} />
          </button>
        </form>
      </footer>

      {showTranscript && (
        <div
          className={styles.modalOverlay}
          onClick={() => setShowTranscript(false)}
        >
          <div
            className={styles.modalContent}
            onClick={(e) => e.stopPropagation()}
          >
            <div className={styles.modalHeader}>
              <div className={styles.modalHeaderLeft}>
                <h2 className={styles.modalTitle}>Исходный текст</h2>
                {session?.sourceText && !isEditing && (
                  <div className={styles.actionButtons}>
                    {Array.isArray(parseSourceText(session.sourceText)) && (
                      <a
                        href={srtDataUrl || "#"}
                        download={`transcript-${id}.srt`}
                        className={styles.actionBtn}
                        onClick={(e) => {
                          if (!srtDataUrl) e.preventDefault();
                        }}
                      >
                        <Download size={14} />
                        SRT
                      </a>
                    )}
                    <button
                      className={styles.actionBtn}
                      onClick={startEditing}
                    >
                      <Pencil size={14} />
                      Редактировать
                    </button>
                    <button className={styles.actionBtn} onClick={copyText}>
                      {copied ? (
                        <>
                          <Check size={14} />
                          Скопировано
                        </>
                      ) : (
                        <>
                          <Clipboard size={14} />
                          Копировать
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>
              <button
                className={styles.closeBtn}
                onClick={() => setShowTranscript(false)}
                aria-label="Закрыть"
              >
                <X size={20} />
              </button>
            </div>
            <div className={styles.modalBody}>
              {isEditing ? (
                <div className={styles.editModeBox}>
                  <div className={styles.editHeader}>
                    <h3 className={styles.editModeTitle}>
                      Редактирование текста
                    </h3>
                    <div className={styles.actionButtons}>
                      <button
                        className={styles.saveBtn}
                        onClick={saveEdits}
                        disabled={saving}
                      >
                        {saving ? "Сохранение..." : "Сохранить"}
                      </button>
                      <button
                        className={styles.cancelBtn}
                        onClick={() => setIsEditing(false)}
                        disabled={saving}
                      >
                        Отмена
                      </button>
                    </div>
                  </div>
                  {editedSegments.length > 0 ? (
                    <div className={styles.segmentsEditList}>
                      {editedSegments.map((seg, idx) => (
                        <div key={idx} className={styles.segmentEdit}>
                          <div className={styles.segmentEditHeader}>
                            <div className={styles.inputGroup}>
                              <label>От (с):</label>
                              <input
                                type="number"
                                step="0.1"
                                value={seg.start}
                                onChange={(e) => {
                                  const ns = [...editedSegments];
                                  ns[idx].start =
                                    parseFloat(e.target.value) || 0;
                                  setEditedSegments(ns);
                                }}
                              />
                            </div>
                            <div className={styles.inputGroup}>
                              <label>До (с):</label>
                              <input
                                type="number"
                                step="0.1"
                                value={seg.end}
                                onChange={(e) => {
                                  const ns = [...editedSegments];
                                  ns[idx].end =
                                    parseFloat(e.target.value) || 0;
                                  setEditedSegments(ns);
                                }}
                              />
                            </div>
                            <div className={styles.inputGroup}>
                              <label>Спикер:</label>
                              <input
                                type="text"
                                value={seg.speaker || ""}
                                onChange={(e) => {
                                  const ns = [...editedSegments];
                                  ns[idx].speaker = e.target.value;
                                  setEditedSegments(ns);
                                }}
                              />
                            </div>
                          </div>
                          <textarea
                            className={styles.segmentEditTextarea}
                            value={seg.text}
                            onChange={(e) => {
                              const ns = [...editedSegments];
                              ns[idx].text = e.target.value;
                              setEditedSegments(ns);
                            }}
                          />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <textarea
                      className={styles.editTextarea}
                      value={editedText}
                      onChange={(e) => setEditedText(e.target.value)}
                    />
                  )}
                </div>
              ) : (
                (() => {
                  const parsed = parseSourceText(session?.sourceText);
                  if (!parsed) {
                    return (
                      <span className={styles.noText}>
                        Текст не найден.
                      </span>
                    );
                  }

                  if (Array.isArray(parsed)) {
                    return (
                      <div className={styles.segmentsList}>
                        {parsed.map((seg: any, i: number) => (
                          <div key={i} className={styles.segment}>
                            <div className={styles.segmentMeta}>
                              <span className={styles.timecode}>
                                [{formatTime(seg.start)} -{" "}
                                {formatTime(seg.end)}]
                              </span>
                              {seg.speaker && (
                                <span className={styles.speaker}>
                                  {seg.speaker}
                                </span>
                              )}
                            </div>
                            <div className={styles.segmentText}>
                              {seg.text}
                            </div>
                          </div>
                        ))}
                      </div>
                    );
                  }

                  return parsed;
                })()
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
