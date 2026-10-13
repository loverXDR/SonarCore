/**
 * Edge-compatible NextAuth Configuration
 *
 * This file contains the configuration that can run in the Edge Runtime.
 * We avoid importing Prisma here.
 */

import Credentials from "next-auth/providers/credentials";
import type { NextAuthConfig } from "next-auth";

export const authConfig = {
  session: {
    strategy: "jwt",
  },
  pages: {
    signIn: "/login",
  },
  providers: [
    // We'll define the full provider in auth.ts, 
    // but we need an empty array here for the middleware to work with types
    Credentials({}), 
  ],
  callbacks: {
    authorized({ auth, request: { nextUrl } }) {
      const isLoggedIn = !!auth?.user;
      const isOnDashboard = nextUrl.pathname.startsWith("/dashboard");
      const isOnSession = nextUrl.pathname.startsWith("/session");
      const isOnTranscribe = nextUrl.pathname.startsWith("/transcribe");

      if (isOnDashboard || isOnSession || isOnTranscribe) {
        if (isLoggedIn) return true;
        return false; // Redirect unauthenticated users to login page
      }
      return true;
    },
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (token?.id) {
        session.user.id = token.id as string;
      }
      return session;
    },
  },
} satisfies NextAuthConfig;
