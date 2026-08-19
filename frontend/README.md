# UniAssist Frontend

ChatGPT-style chat UI for the UniAssist RAG backend, built with the Next.js App Router.

## Features
- Multi-turn chat with per-conversation history persisted in `localStorage`
- Markdown rendering: headings, bullet lists, and tables
- Source citations, suggestion chips, retry-on-error, and request cancellation

## Getting Started

```bash
npm install

# Optional: point the UI at a non-default backend
cp .env.example .env.local   # then set NEXT_PUBLIC_API_BASE

npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Base URL of the FastAPI backend |

## Scripts
- `npm run dev` — start the dev server
- `npm run build` — production build
- `npm run start` — serve the production build
- `npm run lint` — run ESLint
