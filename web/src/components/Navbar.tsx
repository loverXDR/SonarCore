import { auth, signOut } from "@/lib/auth";
import Link from "next/link";
import NavbarClient from "./NavbarClient";
import styles from "./Navbar.module.css";

function WaveformIcon() {
  return (
    <svg
      width="28"
      height="20"
      viewBox="0 0 28 20"
      fill="none"
      className={styles.waveformSvg}
    >
      <path
        d="M1 10 C3 4, 5 16, 7 10 C9 4, 11 16, 13 10 C15 4, 17 16, 19 10 C21 4, 23 16, 25 10 C26 7, 27 13, 28 10"
        stroke="url(#waveGrad)"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
      />
      <defs>
        <linearGradient id="waveGrad" x1="0" y1="0" x2="28" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export default async function Navbar() {
  const session = await auth();

  return (
    <nav className={styles.nav}>
      <div className={styles.container}>
        <Link href="/" className={styles.logo}>
          <WaveformIcon />
          Sonar<span>Core</span>
        </Link>

        <NavbarClient
          isAuthenticated={!!session}
          userName={session?.user?.name || null}
        />
      </div>
    </nav>
  );
}
