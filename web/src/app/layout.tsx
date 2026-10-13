import type { Metadata } from "next";
import "./globals.css";
import { Inter } from "next/font/google";
import Navbar from "@/components/Navbar";
import { ToastProvider } from "@/components/Toast";

const inter = Inter({ subsets: ["latin", "cyrillic"] });

export const metadata: Metadata = {
  title: "SonarCore - Мультиагентная RAG Система",
  description: "Транскрибация аудио, суммаризация и умный чат на базе ИИ",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru">
      <body className={inter.className}>
        <ToastProvider>
          <Navbar />
          <main className="container">{children}</main>
        </ToastProvider>
      </body>
    </html>
  );
}
