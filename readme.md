# Autonomous AI Worker — Day 1 (Ollama + Qwen3 8B)

Goal: an LLM that understands a task and **acts** — selecting tools, observing
results, looping, and verifying — instead of just answering.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- Qwen3 8B pulled: `ollama pull qwen3:8b`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # defaults are fine for a local Ollama

python data/make_sample_invoices.py
```

Make sure `ollama serve` is running in another terminal before proceeding.

## Run

```bash
python -m backend.main
```

Try:

```
Task> Process the latest Amazon invoice.
Task> Find the latest invoice from ABC Corp.
Task> Process the latest OpenAI invoice.
```

## Day 1 tools

- `search_files(query)` – list matching invoice files
- `read_file(filename)` – extract text from a PDF/TXT invoice
- `search_invoice(vendor?)` – check the internal system for duplicates
- `submit_invoice(vendor, amount, due_date)` – write into the internal system
- `verify_invoice(invoice_id)` – read back what was written

## Notes on the LLM

- Model: `qwen3:8b` via Ollama HTTP API (`/api/chat`).
- Thinking mode is disabled (`"think": false`) so the loop stays fast and the
  model emits tool calls directly instead of long reasoning blocks.
- If a specific run of `qwen3:8b` is unreliable with tool calling, try
  `qwen2.5:7b-instruct` or `qwen3:14b` — the code is model-agnostic; just change
  `OLLAMA_MODEL` in `.env`.




  # Autonomous AI Worker — Day 2 (Ollama + Qwen3 8B)

Goal-driven agent. The LLM *reasons* about what the user actually wants,
chooses tools to gather the missing information, and — for write tasks —
verifies its own write before declaring success.

## Architecture

    task
     ↓
    classifier  →  structured goal {task_type, vendor, month, latest, ...}
     ↓
    agent loop: OBSERVE → REASON → DECIDE → ACT → OBSERVE RESULT
     ↓           (guards: policy check + finalize gate)
    finalize_answer(summary, evidence)

Task types: READ · AGGREGATE · WRITE · VERIFY.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- `ollama pull qwen3:8b`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python data/make_sample_invoices.py
```

## Run

```bash
ollama serve   # in another terminal
python -m backend.main
```

Try each of these — the first three must NOT write anything:

```
Task> Find the latest Amazon invoice.
Task> Give me all July invoices.
Task> Check whether the September Amazon invoice was entered.
Task> Process the latest Amazon invoice.
```

## What changed since Day 1

- Task goals are classified before planning (`agent/classifier.py`).
- The planner no longer hardcodes a workflow.
- `submit_invoice` rejects duplicates at the tool level.
- `verify_invoice` returns `{verified, mismatches}` by comparing expected values.
- `finalize_answer` is the stop condition; for WRITE goals it is gated on a passing verification.
- A **policy guard** blocks `submit_invoice` unless the goal is a WRITE.
- Transient tool failures get bounded retries inside `_execute_tool`.
- `search_files` scans filenames *and* content and returns parsed invoice metadata.

## Notes on the LLM

- `qwen3:8b` via Ollama HTTP API (`/api/chat`); thinking mode disabled.
- If tool calling is unreliable on your build, switch `OLLAMA_MODEL` to
  `qwen2.5:7b-instruct` or `qwen3:14b`. Nothing else changes.


  ## Notes on the LLM

The agent uses **ChatGroq** (LangChain) with `llama-3.3-70b-versatile` by default.
Get an API key at https://console.groq.com/keys and put it in `.env` as
`GROQ_API_KEY`.

Recommended models (set `GROQ_MODEL` in `.env`):
- `llama-3.3-70b-versatile`  — best tool-calling reliability, default
- `qwen/qwen3-32b`           — Qwen on Groq
- `llama-3.1-8b-instant`     — fastest, may need a stricter prompt

All tool schemas in `agent/planner.py` are passed through `bind_tools()` unchanged,
so switching models is just an env-var change.