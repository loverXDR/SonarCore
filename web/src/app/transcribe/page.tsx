"use client";

import { useState, Suspense, useEffect, useMemo } from "react";
import AudioUploader from "@/components/AudioUploader";
import { useToast } from "@/components/Toast";
import {
  Download,
  Pencil,
  Clipboard,
  Check,
  Copy,
} from "lucide-react";
import styles from "./transcribe.module.css";
import { TranscriptionData, TranscriptionSegment } from "@/lib/types";
import { useRouter, useSearchParams } from "next/navigation";

// Speaker color palette
const SPEAKER_COLORS = [
  "#6366f1",
  "#f59e0b",
  "#10b981",
  "#ef4444",
  "#8b5cf6",
  "#ec4899",
  "#14b8a6",
  "#f97316",
];

function getSpeakerColor(speaker: string, speakerMap: Map<string, string>): string {
  if (speakerMap.has(speaker)) return speakerMap.get(speaker)!;
  const color = SPEAKER_COLORS[speakerMap.size % SPEAKER_COLORS.length];
  speakerMap.set(speaker, color);
  return color;
}

function TranscribeContent() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TranscriptionData | null>(null);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  // Progress bar
  const [progress, setProgress] = useState(0);

  const [isEditing, setIsEditing] = useState(false);
  const [editedSegments, setEditedSegments] = useState<any[]>([]);
  const [editedText, setEditedText] = useState("");
  const [saving, setSaving] = useState(false);

  // Segment copy tracking
  const [copiedSegIdx, setCopiedSegIdx] = useState<number | null>(null);

  const searchParams = useSearchParams();
  const createSessionParam = searchParams.get("createSession");
  const idParam = searchParams.get("id");
  const router = useRouter();
  const { toast } = useToast();

  // Speaker color map
  const speakerColorMap = useMemo(() => {
    const map = new Map<string, string>();
    if (result?.segments) {
      result.segments.forEach((seg) => {
        if (seg.speaker) getSpeakerColor(seg.speaker, map);
      });
    }
    return map;
  }, [result?.segments]);

  const startEditing = () => {
    if (!result) return;
    setEditedSegments(
      result.segments ? JSON.parse(JSON.stringify(result.segments)) : []
    );
    setEditedText(result.text || "");
    setIsEditing(true);
  };

  const saveEdits = async () => {
    if (!result?.id) return;
    setSaving(true);
    try {
      const res = await fetch(`/api/transcribe/${result.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: editedText,
          segments: editedSegments,
        }),
      });
      if (!res.ok) throw new Error("Failed to save changes");
      setResult({
        ...result,
        text: editedText,
        segments: editedSegments,
      } as TranscriptionData);
      setIsEditing(false);
      toast("Изменения сохранены", "success");
    } catch (e: any) {
      setError(e.message);
      toast("Ошибка при сохранении", "error");
    } finally {
      setSaving(false);
    }
  };

  useEffect(() => {
    if (idParam) {
      async function fetchTranscription() {
        setLoading(true);
        try {
          const res = await fetch(`/api/transcribe/${idParam}`);
          const data = await res.json();
          if (!res.ok) throw new Error(data.error);
          setResult(data);
        } catch (err: any) {
          setError(err.message);
        } finally {
          setLoading(false);
        }
      }
      fetchTranscription();
    }
  }, [idParam]);

  // Fake progress animation
  useEffect(() => {
    if (!loading) {
      if (progress > 0 && progress < 100) {
        setProgress(100);
        const timer = setTimeout(() => setProgress(0), 600);
        return () => clearTimeout(timer);
      }
      return;
    }
    setProgress(0);
    const interval = setInterval(() => {
      // Quick to 60%, slow to 90%
      setProgress((prev) => {
        if (prev >= 90) return 90;
        if (prev < 60) return prev + 3;
        return prev + 0.3;
      });
    }, 100);
    return () => clearInterval(interval);
  }, [loading]);

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

  const buildSRTContent = () => {
    if (!result || !result.segments) return null;
    let srt = "";
    result.segments.forEach((segment: any, index: number) => {
      srt += `${index + 1}\n`;
      srt += `${formatSRTTime(segment.start)} --> ${formatSRTTime(segment.end)}\n`;
      srt += `${segment.speaker ? `${segment.speaker}: ` : ""}${segment.text}\n\n`;
    });
    return srt;
  };

  const srtDataUrl = useMemo(() => {
    const srt = buildSRTContent();
    if (!srt) return null;
    const blob = new Blob([srt], { type: "text/plain;charset=utf-8" });
    return URL.createObjectURL(blob);
  }, [result?.segments]);

  const copyText = () => {
    if (!result) return;
    const textToCopy = result.segments
      ? result.segments
          .map((s: any) => `${s.speaker ? s.speaker + ": " : ""}${s.text}`)
          .join(" ")
      : result.text;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    toast("Текст скопирован", "success");
    setTimeout(() => setCopied(false), 2000);
  };

  const copySegment = (text: string, idx: number) => {
    navigator.clipboard.writeText(text);
    setCopiedSegIdx(idx);
    toast("Сегмент скопирован", "success");
    setTimeout(() => setCopiedSegIdx(null), 2000);
  };

  const handleUpload = async (formData: FormData) => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      if (createSessionParam === "true") {
        const res = await fetch("/api/sessions/audio", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        router.push(`/session/${data.id}`);
        return;
      }

      const res = await fetch("/api/transcribe", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setResult(data);
      toast("Транскрибация завершена", "success");
    } catch (err: any) {
      setError(err.message);
      toast("Ошибка транскрибации", "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>
          {createSessionParam === "true"
            ? "Новая сессия чата"
            : "Транскрибация аудио"}
        </h1>
        <p className={styles.subtitle}>
          Загрузите аудиофайл или укажите ссылку, чтобы получить текст
        </p>
      </header>

      {/* Progress bar */}
      {progress > 0 && (
        <div className={styles.progressWrap}>
          <div
            className={styles.progressBar}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      <div className={styles.main}>
        <div className={styles.uploaderBox}>
          <AudioUploader onUpload={handleUpload} loading={loading} />
          {error && <div className={styles.error}>{error}</div>}
        </div>

        {result && (
          <div className={styles.resultBox}>
            <div className={styles.resultHeader}>
              <h2>Результат транскрибации</h2>
              <div className={styles.headerActions}>
                {result.segments && result.segments.length > 0 && (
                  <a
                    href={srtDataUrl || "#"}
                    download={`transcript-${result.fileName || "audio"}.srt`}
                    className={styles.actionBtn}
                    onClick={(e) => {
                      if (!srtDataUrl) e.preventDefault();
                    }}
                  >
                    <Download size={14} />
                    SRT
                  </a>
                )}
                {result.id && !isEditing && (
                  <button className={styles.actionBtn} onClick={startEditing}>
                    <Pencil size={14} />
                    Редактировать
                  </button>
                )}
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
                <button
                  className={styles.createBtn}
                  onClick={async () => {
                    setLoading(true);
                    try {
                      const res = await fetch("/api/sessions/text", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          text: result.text,
                          title: result.fileName || "Новая сессия",
                        }),
                      });
                      const data = await res.json();
                      router.push(`/session/${data.id}`);
                    } catch {
                      setError("Ошибка при создании сессии");
                      setLoading(false);
                    }
                  }}
                >
                  Начать чат по этому тексту
                </button>
              </div>
            </div>
            {isEditing ? (
              <div className={styles.editModeBox}>
                <h3 className={styles.editModeTitle}>Общий текст:</h3>
                <textarea
                  className={styles.editTextarea}
                  value={editedText}
                  onChange={(e) => setEditedText(e.target.value)}
                />

                {editedSegments.length > 0 && (
                  <h3 className={styles.editModeTitle}>
                    Таймкоды и спикеры:
                  </h3>
                )}
                <div className={styles.segmentsEditList}>
                  {editedSegments.map((seg, idx) => (
                    <div key={idx} className={styles.segmentEdit}>
                      <div className={styles.segmentEditHeader}>
                        <div className={styles.inputGroup}>
                          <label>От (сек):</label>
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
                          <label>До (сек):</label>
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
                            placeholder="Спикер"
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

                <div className={styles.editActions}>
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
            ) : (
              <>
                <div className={styles.textContent}>{result.text}</div>

                <div className={styles.segments}>
                  {result.segments.map(
                    (seg: TranscriptionSegment, idx: number) => {
                      const speakerLabel =
                        seg.speaker || `Спикер ${(idx % 2) + 1}`;
                      const color = getSpeakerColor(
                        speakerLabel,
                        speakerColorMap
                      );
                      return (
                        <div key={idx} className={styles.segment}>
                          <div className={styles.segmentMeta}>
                            <span
                              className={styles.speaker}
                              style={{ color }}
                            >
                              {speakerLabel}
                            </span>
                            <span className={styles.time}>
                              {Math.floor(seg.start)}с -{" "}
                              {Math.floor(seg.end)}с
                            </span>
                            <button
                              className={styles.segCopyBtn}
                              onClick={() => copySegment(seg.text, idx)}
                              aria-label="Копировать сегмент"
                            >
                              {copiedSegIdx === idx ? (
                                <Check size={12} />
                              ) : (
                                <Copy size={12} />
                              )}
                            </button>
                          </div>
                          <div
                            className={styles.segmentText}
                            style={{
                              borderLeftColor: color,
                            }}
                          >
                            {seg.text}
                          </div>
                        </div>
                      );
                    }
                  )}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function TranscribePage() {
  return (
    <Suspense fallback={<div>Загрузка...</div>}>
      <TranscribeContent />
    </Suspense>
  );
}
