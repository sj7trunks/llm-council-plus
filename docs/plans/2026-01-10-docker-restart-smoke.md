# Docker Restart + Smoke Checks (Local)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Restart the app locally via Docker Compose and verify the core “create conversation → send message” flow (including web search tool outputs).

**Architecture:** `docker compose` starts two containers: `llm-council-backend` (FastAPI) on `:8001` and `llm-council-frontend` (nginx + static build) on `:80`.

**Tech Stack:** Docker Compose, FastAPI (backend), nginx + Vite build (frontend), OpenRouter (router), DuckDuckGo (`ddgs`) for web search.

---

### Task 1: Start the stack

**Files:**
- Modify: none (operational)

**Step 1: Start containers**

Run: `cd .worktrees/feature-router-per-conversation && docker compose up -d`

Expected:
- `llm-council-backend` and `llm-council-frontend` start successfully
- Warnings about unset env vars are OK if defaults exist, but can be fixed by setting `COUNCIL_MODELS` and `CHAIRMAN_MODEL` in `.env`

**Step 2: Confirm containers are healthy**

Run: `cd .worktrees/feature-router-per-conversation && docker compose ps`

Expected:
- backend: `Up ... (healthy)`
- frontend: `Up ...` (may show `health: starting` briefly)

---

### Task 2: Backend health endpoints

**Step 1: Check version endpoint**

Run: `curl -fsS http://localhost:8001/api/version`

Expected:
- JSON with `"version": "..."`

**Step 2: Check setup status**

Run: `curl -fsS http://localhost:8001/api/setup/status | python -m json.tool`

Expected:
- `"setup_required": false`
- `"web_search_enabled": true`
- `"duckduckgo_enabled": true`

---

### Task 3: UI availability

**Step 1: Ensure frontend serves HTML**

Run: `curl -fsSI http://localhost/ | sed -n '1,15p'`

Expected:
- `HTTP/1.1 200 OK`

---

### Task 4: API smoke “create conversation → send message (stream)”

**Step 1: Create a conversation with free OpenRouter models**

Run:

```bash
curl -fsS -X POST http://localhost:8001/api/conversations \
  -H 'Content-Type: application/json' \
  -d '{
    "models": [
      "meta-llama/llama-3.2-3b-instruct:free",
      "mistralai/mistral-7b-instruct:free"
    ],
    "chairman": "google/gemini-2.0-flash-exp:free",
    "execution_mode": "full",
    "router_type": "openrouter"
  }'
```

Expected:
- JSON containing `"id": "<uuid>"`

**Step 2: Send a streaming message with DuckDuckGo web search enabled**

Run:

```bash
CONV=<uuid-from-previous-step>
curl -fsS -N --max-time 45 \
  -X POST "http://localhost:8001/api/conversations/$CONV/message/stream" \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "Найди в вебе 3 свежие новости про OpenRouter и кратко перескажи. Укажи источники.",
    "web_search_provider": "duckduckgo"
  }' | sed -n '1,120p'
```

Expected:
- SSE event `{"type":"stage1_start"...}`
- SSE event `{"type":"tool_outputs"...}` (this powers the “Search context panel” in the UI)

