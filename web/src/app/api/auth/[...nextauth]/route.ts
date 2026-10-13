/**
 * NextAuth Route Handler
 *
 * Exposes the NextAuth API endpoints under /api/auth/*
 */

import { handlers } from "@/lib/auth";

export const { GET, POST } = handlers;
