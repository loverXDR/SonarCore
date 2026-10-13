# Frontend Dockerfile
FROM node:22-alpine AS base
RUN apk add --no-cache libc6-compat openssl
WORKDIR /app

# 1. Install dependencies
FROM base AS deps
COPY package.json package-lock.json* ./
RUN npm install

# 2. Builder
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

ENV NEXT_TELEMETRY_DISABLED 1
ENV NODE_ENV production
ENV DATABASE_URL="postgresql://user:pass@localhost:5432/db"

# Generate Prisma client (Prisma 7 compatible)
RUN npx prisma generate

RUN npm run build

# 3. Runner
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

# Alpine-specific user creation
RUN addgroup -S -g 1001 nodejs
RUN adduser -S -u 1001 nextjs -G nodejs

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# IMPORTANT: For standalone mode with Prisma, we might need the prisma folder
COPY --from=builder /app/prisma ./prisma
COPY --from=builder /app/entrypoint.sh ./entrypoint.sh
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/prisma.config.ts ./prisma.config.ts

USER root
RUN chmod +x ./entrypoint.sh
USER nextjs

ENTRYPOINT ["./entrypoint.sh"]
CMD ["node", "server.js"]