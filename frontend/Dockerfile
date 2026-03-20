# ── Stage 1: Build React app ──────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies (cached layer if package.json unchanged)
COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline

# Copy source and build
COPY . .
ARG REACT_APP_API_URL=https://plantdisease.yourdomain.com
ENV REACT_APP_API_URL=$REACT_APP_API_URL
RUN npm run build

# ── Stage 2: Serve with nginx ─────────────────────────────────────────────────
FROM nginx:1.25-alpine AS production

COPY --from=builder /app/build /usr/share/nginx/html

# SPA routing — redirect all 404s to index.html
COPY <<EOF /etc/nginx/conf.d/default.conf
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    gzip on;
    gzip_types text/plain application/javascript text/css application/json image/svg+xml;

    location / {
        try_files \$uri \$uri/ /index.html;
        expires 1h;
        add_header Cache-Control "public, no-transform";
    }

    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location /api/ {
        return 404;  # Nginx should not proxy API in production — handled by ingress
    }
}
EOF

EXPOSE 80
HEALTHCHECK --interval=30s CMD wget -qO- http://localhost/ || exit 1
