"use client";

import { useState, useRef } from "react";
import { Upload, FileAudio } from "lucide-react";
import styles from "./AudioUploader.module.css";

interface AudioUploaderProps {
  onUpload: (data: FormData) => void;
  loading: boolean;
}

const ACCEPTED_FORMATS = ["WAV", "MP3", "M4A", "OGG", "FLAC"];

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

export default function AudioUploader({ onUpload, loading }: AudioUploaderProps) {
  const [dragActive, setDragActive] = useState(false);
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [useDiarization, setUseDiarization] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setUrl("");
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUrl("");
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!file && !url) return;

    const formData = new FormData();
    if (file) {
      formData.append("file", file);
    } else if (url) {
      formData.append("url", url);
    }
    if (useDiarization) {
      formData.append("diarize", "true");
    }

    onUpload(formData);
  };

  return (
    <div className={styles.container}>
      <form
        className={`${styles.dropzone} ${dragActive ? styles.active : ""}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          className={styles.hiddenInput}
          onChange={handleChange}
        />

        <div className={styles.iconWrap}>
          {file ? <FileAudio size={36} /> : <Upload size={36} />}
        </div>

        <h3>{file ? file.name : "Перетащите аудиофайл сюда"}</h3>

        {/* File metadata */}
        {file && (
          <div className={styles.fileMeta}>
            {formatFileSize(file.size)}
          </div>
        )}

        <p>или</p>
        <button
          type="button"
          className={styles.browseBtn}
          onClick={() => fileInputRef.current?.click()}
        >
          Выберите файл
        </button>

        {/* Format chips */}
        <div className={styles.formatChips}>
          {ACCEPTED_FORMATS.map((fmt) => (
            <span key={fmt} className={styles.chip}>
              {fmt}
            </span>
          ))}
        </div>
      </form>

      {/* Diarization toggle */}
      <div className={styles.toggleRow}>
        <span className={styles.toggleLabel}>Диаризация спикеров</span>
        <button
          type="button"
          role="switch"
          aria-checked={useDiarization}
          className={`${styles.toggle} ${useDiarization ? styles.toggleOn : ""}`}
          onClick={() => setUseDiarization((prev) => !prev)}
        >
          <span className={styles.toggleThumb} />
        </button>
      </div>

      <div className={styles.divider}>или укажите прямую ссылку</div>

      <div className={styles.urlGroup}>
        <input
          type="url"
          className={styles.urlInput}
          placeholder="https://example.com/audio.mp3"
          value={url}
          onChange={(e) => {
            setUrl(e.target.value);
            setFile(null);
          }}
        />
        <button
          className={styles.submitBtn}
          disabled={loading || (!file && !url)}
          onClick={handleSubmit}
        >
          {loading ? "Загрузка..." : "Транскрибировать"}
        </button>
      </div>
    </div>
  );
}
