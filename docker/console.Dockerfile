# Merchant console image.
#
# Development-oriented: runs the Vite dev server so the console is editable
# inside the compose stack. A production deployment would build to static assets
# and serve them from a CDN or an nginx layer; that is a Phase 6 concern and is
# deliberately not pretended-at here.

FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
