import styles from "./Skeleton.module.css";

interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  borderRadius?: string | number;
}

export function Skeleton({
  width = "100%",
  height = 16,
  borderRadius = 8,
}: SkeletonProps) {
  return (
    <div
      className={styles.skeleton}
      style={{
        width: typeof width === "number" ? `${width}px` : width,
        height: typeof height === "number" ? `${height}px` : height,
        borderRadius:
          typeof borderRadius === "number"
            ? `${borderRadius}px`
            : borderRadius,
      }}
    />
  );
}

export function SessionCardSkeleton() {
  return (
    <div className={styles.cardSkeleton}>
      <div className={styles.cardHeader}>
        <Skeleton width={40} height={40} borderRadius={10} />
        <Skeleton width={60} height={22} borderRadius={12} />
      </div>
      <Skeleton width="75%" height={20} borderRadius={6} />
      <Skeleton width="50%" height={14} borderRadius={6} />
      <div className={styles.cardFooter}>
        <Skeleton width={100} height={14} borderRadius={6} />
        <Skeleton width={80} height={12} borderRadius={6} />
      </div>
    </div>
  );
}

export function ChatMessageSkeleton({
  role = "assistant",
}: {
  role?: "user" | "assistant";
}) {
  const isUser = role === "user";
  return (
    <div
      className={`${styles.messageSkeleton} ${
        isUser ? styles.messageUser : styles.messageAssistant
      }`}
    >
      <Skeleton width={32} height={32} borderRadius={8} />
      <div className={styles.messageBody}>
        <Skeleton width={60} height={10} borderRadius={4} />
        <Skeleton
          width={isUser ? 160 : 260}
          height={42}
          borderRadius={12}
        />
      </div>
    </div>
  );
}
