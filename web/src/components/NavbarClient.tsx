"use client";

import { useState } from "react";
import Link from "next/link";
import { signOut } from "next-auth/react";
import { Menu, X, LayoutDashboard, Mic } from "lucide-react";
import styles from "./Navbar.module.css";

interface NavbarClientProps {
  isAuthenticated: boolean;
  userName: string | null;
}

function UserAvatar({ name }: { name: string }) {
  const initial = name.charAt(0).toUpperCase();
  return (
    <div className={styles.avatar} title={name}>
      {initial}
    </div>
  );
}

export default function NavbarClient({
  isAuthenticated,
  userName,
}: NavbarClientProps) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <>
      <button
        className={styles.hamburger}
        onClick={() => setMenuOpen((prev) => !prev)}
        aria-label={menuOpen ? "Закрыть меню" : "Открыть меню"}
      >
        {menuOpen ? <X size={22} /> : <Menu size={22} />}
      </button>
      <div className={`${styles.links} ${menuOpen ? styles.linksOpen : ""}`}>
        {isAuthenticated ? (
          <>
            <Link
              href="/dashboard"
              className={styles.link}
              onClick={() => setMenuOpen(false)}
            >
              <LayoutDashboard size={16} />
              Дашборд
            </Link>
            <Link
              href="/transcribe"
              className={styles.link}
              onClick={() => setMenuOpen(false)}
            >
              <Mic size={16} />
              Транскрибация
            </Link>
            <div className={styles.user}>
              <UserAvatar name={userName || "U"} />
              <span className={styles.username}>{userName}</span>
              <button
                type="button"
                className={styles.logoutBtn}
                onClick={() => signOut({ callbackUrl: "/login" })}
              >
                Выйти
              </button>
            </div>
          </>
        ) : (
          <>
            <Link
              href="/login"
              className={styles.link}
              onClick={() => setMenuOpen(false)}
            >
              Вход
            </Link>
            <Link
              href="/register"
              className={styles.btn}
              onClick={() => setMenuOpen(false)}
            >
              Регистрация
            </Link>
          </>
        )}
      </div>
      {menuOpen && (
        <div
          className={styles.backdrop}
          onClick={() => setMenuOpen(false)}
        />
      )}
    </>
  );
}
