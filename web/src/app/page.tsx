import Link from "next/link";
import {
  BrainCircuit,
  Database,
  MessageSquare,
  FileAudio,
} from "lucide-react";
import styles from "./landing.module.css";
import { auth } from "@/lib/auth";

function AudioWaveform() {
  return (
    <div className={styles.waveform} aria-hidden="true">
      <span className={styles.bar} />
      <span className={styles.bar} />
      <span className={styles.bar} />
      <span className={styles.bar} />
      <span className={styles.bar} />
    </div>
  );
}

const features = [
  {
    Icon: BrainCircuit,
    title: "Мульти-агентный RAG",
    text: "Специализированные агенты анализируют ваши записи, объединяя лучшие практики LLM и поиска.",
  },
  {
    Icon: Database,
    title: "Персистентность",
    text: "Все ваши сессии, чаты и результаты транскрибации надежно хранятся в вашей базе данных.",
  },
  {
    Icon: MessageSquare,
    title: "Умный Чат",
    text: "Задавайте вопросы прямо по содержимому аудиофайлов и получайте точные ответы с ссылками на источники.",
  },
  {
    Icon: FileAudio,
    title: "Транскрибация",
    text: "Поддержка Whisper и Sherpa. Диаризация докладчиков.",
  },
];

const steps = [
  { number: "01", title: "Загрузите аудио", description: "Перетащите файл или укажите ссылку на аудиозапись." },
  { number: "02", title: "ИИ анализирует", description: "Whisper транскрибирует, агенты индексируют содержимое." },
  { number: "03", title: "Общайтесь с данными", description: "Задавайте вопросы и получайте точные ответы с цитатами." },
];

export default async function Home() {
  const session = await auth();

  return (
    <div className={styles.container}>
      <div className={styles.hero}>
        <div className={styles.badge}>
          <AudioWaveform />
          Представляем SonarCore v1.0
        </div>
        <h1 className={styles.title}>
          Трансформируйте <span>аудио</span> в интеллектуальные инсайты
        </h1>
        <p className={styles.subtitle}>
          Автоматизируйте транскрибацию, разделяйте спикеров и общайтесь с вашими
          данными с помощью мульти-агентного ИИ нового поколения.
        </p>

        <div className={styles.actions}>
          {session ? (
            <>
              <Link href="/dashboard" className={`${styles.btn} ${styles.primary}`}>
                Дашборд
              </Link>
              <Link href="/transcribe" className={`${styles.btn} ${styles.secondary}`}>
                Быстрая транскрибация
              </Link>
            </>
          ) : (
            <>
              <Link href="/register" className={`${styles.btn} ${styles.primary}`}>
                Начать бесплатно
              </Link>
              <Link href="/login" className={`${styles.btn} ${styles.secondary}`}>
                Войти в проект
              </Link>
            </>
          )}
        </div>
      </div>

      <div className={styles.features}>
        {features.map((f) => (
          <div key={f.title} className={styles.feature}>
            <div className={styles.fIcon}>
              <f.Icon size={28} />
            </div>
            <h3 className={styles.fTitle}>{f.title}</h3>
            <p className={styles.fText}>{f.text}</p>
          </div>
        ))}
      </div>

      <section className={styles.howSection}>
        <h2 className={styles.howTitle}>Как это работает</h2>
        <div className={styles.steps}>
          {steps.map((step, i) => (
            <div key={step.number} className={styles.step}>
              <div className={styles.stepNumber}>{step.number}</div>
              <h3 className={styles.stepTitle}>{step.title}</h3>
              <p className={styles.stepDesc}>{step.description}</p>
              {i < steps.length - 1 && (
                <div className={styles.stepLine} aria-hidden="true" />
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
