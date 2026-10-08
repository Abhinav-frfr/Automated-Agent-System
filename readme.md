# Autonomous AI Worker

An autonomous AI worker that can **understand a task, reason about what needs to be done, select and use tools, observe their results, recover from failures, and verify its actions** before declaring a task complete.

This project is being developed incrementally, with each development day introducing a new capability to the agent.

---

## Project Overview

Traditional LLM applications usually follow a simple pattern:

```text
User → LLM → Answer
```

This project aims to build a more autonomous workflow:

```text
                    ┌──────────────┐
                    │     Task     │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │ Goal / Task  │
                    │ Classifier   │
                    └──────┬───────┘
                           ↓
                 ┌─────────────────────┐
                 │      Agent Loop     │
                 │                     │
                 │ OBSERVE             │
                 │    ↓                │
                 │ REASON              │
                 │    ↓                │
                 │ DECIDE              │
                 │    ↓                │
                 │ ACT                 │
                 │    ↓                │
                 │ OBSERVE RESULT      │
                 │    ↓                │
                 │ Repeat if required  │
                 └─────────┬───────────┘
                           ↓
                    ┌──────────────┐
                    │ Verification │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │ Final Answer  │
                    └──────────────┘
```

The long-term goal is to move toward an **AI employee / AI worker architecture** capable of executing real-world tasks through tools and external systems rather than simply generating text.

---

# Key Capabilities

The agent is designed around several core capabilities:

- **Task understanding**
- **Goal classification**
- **Tool selection**
- **Multi-step planning**
- **Tool execution**
- **Observation of tool results**
- **Memory / context**
- **Failure recovery**
- **Retry mechanisms**
- **Policy enforcement**
- **Action verification**
- **Controlled finalization**

The project currently focuses on invoice-processing workflows as a controlled environment for developing these capabilities.

---

# Current Architecture

The current system consists of several major components:

```text
User Task
    │
    ▼
┌──────────────────┐
│ Task Classifier  │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────┐
│ Structured Goal            │
│                            │
│ task_type                  │
│ vendor                     │
│ month                      │
│ year                       │
│ latest                     │
│ invoice_number             │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│       Agent Planner        │
│                            │
│ OBSERVE → REASON → DECIDE  │
│              ↓             │
│             ACT            │
└────────────┬───────────────┘
             │
             ▼
        Tool Selection
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
  Search   Read     Submit
  Files    File     Invoice
     │       │        │
     └───────┼────────┘
             ▼
        Tool Result
             │
             ▼
        Agent observes
             │
             ▼
      More actions needed?
         /          \
       YES           NO
        │             │
        ▼             ▼
   Agent loop    Verification
                      │
                      ▼
              Final Answer
```

---

# Agent Goals

The agent currently supports four major task categories:

| Goal | Description |
|---|---|
| `READ` | Retrieve or inspect information |
| `AGGREGATE` | Collect information across multiple invoices |
| `WRITE` | Perform an action such as submitting an invoice |
| `VERIFY` | Check whether a particular state or action exists |

The classifier converts the user's natural-language request into a structured goal before the agent begins execution.

Example:

```text
User:
Find the latest Amazon invoice.
```

becomes approximately:

```json
{
  "task_type": "READ",
  "vendor": "Amazon",
  "latest": true
}
```

This structured goal is then used by the planner and policy layer.

---

# Tool System

The agent currently works with the following tools:

### `search_files(query)`

Searches the invoice data and returns relevant files and invoice metadata.

### `read_file(filename)`

Reads the contents of an invoice PDF/TXT file.

### `search_invoice(vendor?)`

Searches the internal invoice system to determine whether an invoice has already been entered.

### `submit_invoice(vendor, amount, due_date)`

Writes an invoice into the internal system.

This operation is protected by policy checks and duplicate detection.

### `verify_invoice(invoice_id)`

Reads the submitted invoice back from the internal system and compares it against the expected values.

### `finalize_answer(summary, evidence)`

Acts as the controlled termination point of the agent.

The agent should not simply stop because the LLM decides that it is finished.

---

# Safety and Reliability

A major focus of the project is making the agent **reliable rather than merely capable**.

The current system contains several safeguards.

## Policy Guard

The agent cannot arbitrarily perform write operations.

For example:

```text
READ goal
   ↓
submit_invoice()
   ↓
BLOCKED
```

`submit_invoice` is only allowed when the classified goal is a `WRITE` task.

---

## Duplicate Protection

The submission tool checks whether an invoice has already been entered.

Therefore:

```text
Submit Invoice
       ↓
Already exists?
   ┌───┴───┐
  YES      NO
   ↓        ↓
REJECTED  SUBMIT
DUPLICATE
```

This prevents repeated execution of the same write action.

---

## Verification Gate

For write tasks, the agent must verify its own action before finalizing.

```text
WRITE
  ↓
submit_invoice()
  ↓
verify_invoice()
  ↓
verified?
 ┌──────┴──────┐
YES           NO
 ↓             ↓
Finalize     Continue /
             Recover
```

This introduces a basic form of **post-action verification**.

---

## Bounded Retries

Transient tool failures are retried a limited number of times.

This prevents the agent from:

- immediately failing because of a temporary error
- retrying forever
- getting stuck in an infinite execution loop

---

# Technology Stack

### Language

- Python

### LLM / Inference

- Ollama
- Qwen3 8B
- ChatGroq
- Llama models
- Qwen models

### Agent Framework

- LangChain
- LangChain Groq integration

### Backend

- Python
- Modular tool architecture

### Data

- PDF/TXT invoices
- Internal JSON-based invoice database

---

# Project Structure

```text
Agentic Worker/
│
├── agent/
│   ├── classifier.py
│   ├── planner.py
│   └── memory.py
│
├── backend/
│   └── main.py
│
├── data/
│   ├── invoices/
│   └── internal_db.json
│
├── evaluation/
│
├── tests/
│
├── tools/
│   ├── file_tools.py
│   ├── invoice_tools.py
│   └── ...
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Setup

## Requirements

- Python 3.10+
- Ollama or Groq API access
- Git

---

## Option 1 — Local Ollama

Install Ollama and pull Qwen3:

```bash
ollama pull qwen3:8b
```

Create the virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Generate the sample invoice data:

```bash
python data/make_sample_invoices.py
```

Start Ollama:

```bash
ollama serve
```

Then run the worker:

```bash
python -m backend.main
```

---

# Option 2 — Groq

The project can also use ChatGroq through LangChain.

Create a Groq API key and add it to `.env`:

```env
GROQ_API_KEY=your_api_key_here
```

Set the model:

```env
GROQ_MODEL=llama-3.3-70b-versatile
```

The agent uses LangChain's tool-binding interface, allowing different supported models to be used without changing the overall tool architecture.

---

# Example Tasks

### READ

```text
Task> Find the latest Amazon invoice.
```

Expected behavior:

```text
Classify task
      ↓
Search Amazon invoices
      ↓
Determine latest invoice
      ↓
Read invoice if required
      ↓
Return result
```

---

### AGGREGATE

```text
Task> Give me all July invoices.
```

The agent should search the available invoice data and return all relevant July invoices without performing any write operation.

---

### VERIFY

```text
Task> Check whether the September Amazon invoice was entered.
```

The agent should inspect the internal system and determine whether the invoice has already been submitted.

---

### WRITE

```text
Task> Process the latest Amazon invoice.
```

The agent should:

```text
Find latest invoice
       ↓
Read invoice
       ↓
Extract required fields
       ↓
Check for duplicate
       ↓
Submit invoice
       ↓
Verify submission
       ↓
Finalize
```

---

# Development Progress

This project is intentionally being developed incrementally.

Each day focuses on introducing a specific capability or improving the reliability of an existing capability.

---

# Day 1 — Basic Autonomous Tool-Using Agent

## Goal

Build an LLM that does more than answer questions.

The first version introduced the fundamental agent loop:

```text
LLM
 ↓
Tool Call
 ↓
Tool Execution
 ↓
Observation
 ↓
LLM
 ↓
Tool Call
 ↓
...
```

The agent could:

- understand a task
- select tools
- execute tools
- observe results
- continue acting
- eventually produce an answer

## Day 1 Tools

```text
search_files(query)
read_file(filename)
search_invoice(vendor?)
submit_invoice(vendor, amount, due_date)
verify_invoice(invoice_id)
```

## Invoice Workflow

The initial agent was designed around invoice processing.

Example:

```text
Task> Process the latest Amazon invoice.
```

The agent could search for the relevant invoice, read it, check the internal system, submit it, and verify the result.

## Model

Day 1 used:

```text
Ollama
   ↓
Qwen3 8B
   ↓
/api/chat
```

Thinking mode was disabled:

```json
{
  "think": false
}
```

This was done to keep the execution loop fast and encourage direct tool calls rather than long reasoning outputs.

## Initial Limitations

The first version relied heavily on the planner's workflow logic.

This created an important limitation:

> The system could execute a workflow, but it did not yet have a strong understanding of the user's underlying goal.

This motivated the changes introduced on Day 2.

---

# Day 2 — Goal-Driven Agent

## Goal

Move from a workflow-oriented agent toward a **goal-driven agent**.

Instead of directly asking:

```text
"What workflow should I execute?"
```

the system first asks:

```text
"What does the user actually want?"
```

---

## Task Classification

A classifier was introduced before the agent loop.

```text
User Task
    ↓
Classifier
    ↓
Structured Goal
    ↓
Planner
    ↓
Agent Loop
```

Example:

```text
Task:
Find the latest Amazon invoice.
```

becomes:

```json
{
  "task_type": "READ",
  "vendor": "Amazon",
  "month": null,
  "year": null,
  "latest": true
}
```

---

## Four Goal Types

Day 2 introduced:

```text
READ
AGGREGATE
WRITE
VERIFY
```

This allows the system to distinguish between information retrieval and actions that modify state.

---

## Policy Guard

A policy guard was introduced to prevent unauthorized write actions.

For example:

```text
Goal = READ
        ↓
Agent attempts submit_invoice()
        ↓
Policy Guard
        ↓
BLOCK
```

This is important because an LLM should not be allowed to perform state-changing actions simply because it generated the corresponding tool call.

---

## Duplicate Protection

`submit_invoice()` now rejects duplicate submissions.

```text
submit_invoice()
       ↓
Check internal system
       ↓
Existing invoice?
   ┌───┴───┐
  YES      NO
   ↓        ↓
REJECT    WRITE
```

---

## Verification

`verify_invoice()` was upgraded to return:

```json
{
  "verified": true,
  "mismatches": []
}
```

The verification process compares the expected invoice values with the values actually stored.

---

## Finalization Gate

A `finalize_answer()` operation was introduced as the controlled termination point.

For WRITE tasks:

```text
WRITE
  ↓
Submit
  ↓
Verify
  ↓
Verification passed?
  ↓
Finalize
```

The agent cannot simply declare success without satisfying the required verification condition.

---

## Bounded Retries

Transient tool failures are handled using bounded retries inside:

```text
_execute_tool()
```

This provides basic failure recovery while preventing infinite loops.

---

## Improved File Search

`search_files()` was upgraded to search:

- filenames
- invoice contents
- parsed invoice metadata

This allows the agent to find relevant invoices based on their contents rather than relying only on filenames.

---

# Day 2 — Example Tasks

The following tasks were specifically used to test the new goal-driven architecture:

```text
Task> Find the latest Amazon invoice.
```

Expected:

```text
READ
```

No write should occur.

---

```text
Task> Give me all July invoices.
```

Expected:

```text
AGGREGATE
```

No write should occur.

---

```text
Task> Check whether the September Amazon invoice was entered.
```

Expected:

```text
VERIFY
```

No write should occur.

---

```text
Task> Process the latest Amazon invoice.
```

Expected:

```text
WRITE
```

The agent should submit and then verify the invoice.

---

# LLM Configuration

The project supports both local and hosted LLM inference.

## Ollama

Default local model:

```text
qwen3:8b
```

Other possible models:

```text
qwen2.5:7b-instruct
qwen3:14b
```

The model can be changed through:

```env
OLLAMA_MODEL=qwen3:8b
```

---

## Groq

The project can also use ChatGroq.

Example:

```env
GROQ_MODEL=llama-3.3-70b-versatile
```

Other supported model options can be configured through the environment file.

The planner passes the tool schemas through LangChain's:

```python
bind_tools()
```

This keeps the agent architecture relatively model-independent.

---

# Environment Variables

Never commit the actual `.env` file.

Use:

```text
.env
```

for local secrets and:

```text
.env.example
```

for the public configuration template.

Example:

```env
OLLAMA_MODEL=qwen3:8b
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
```

---

# Evaluation

The `evaluation/` directory is intended to measure the agent beyond simple answer accuracy.

Future evaluation areas include:

- Task completion rate
- Tool-selection accuracy
- Tool-call correctness
- Number of unnecessary tool calls
- Recovery from tool failures
- Duplicate prevention
- Verification success rate
- Policy violations
- Final-answer correctness
- End-to-end execution success

The goal is to evaluate the **agent's behavior**, not just the quality of generated text.

---

# Future Development

The project will progressively move toward more realistic autonomous-agent capabilities.

Planned areas include:

### Memory

Persistent memory for:

- previous tasks
- observations
- useful context
- user preferences
- execution history

### Better Planning

Move toward explicit planning for complex multi-step tasks.

### Tool Recovery

Allow the agent to:

```text
Detect failure
    ↓
Understand failure
    ↓
Choose alternative action
    ↓
Retry / recover
```

rather than simply retrying the same operation.

### Human Approval

Introduce approval gates for sensitive actions:

```text
Agent wants to perform action
            ↓
Is approval required?
       /           \
     YES            NO
      ↓              ↓
Ask human         Execute
      ↓
Approved?
 /       \
YES       NO
 ↓         ↓
Execute   Stop
```

### Browser / Computer Use

Extend the worker beyond local invoice files toward interacting with external applications and websites.

### API Tools

Allow the worker to interact with external APIs.

### Stronger Evaluation

Build a benchmark of tasks measuring:

- planning
- tool use
- recovery
- verification
- safety
- task completion

---

# Development Philosophy

The project is being developed around a simple principle:

> **An AI worker should not just know what to say. It should know what to do, be able to do it, and verify that it actually happened.**

The progression of the project therefore focuses on:

```text
LLM
 ↓
Tool Calling
 ↓
Agent Loop
 ↓
Goal Understanding
 ↓
Policy
 ↓
Verification
 ↓
Recovery
 ↓
Memory
 ↓
Real-World Actions
```

---

# Current Status

### Completed

- [x] Basic LLM tool-calling loop
- [x] Invoice search
- [x] Invoice file reading
- [x] Invoice submission
- [x] Invoice verification
- [x] Goal classification
- [x] READ / AGGREGATE / WRITE / VERIFY goals
- [x] Policy guard
- [x] Duplicate protection
- [x] Verification gate
- [x] Bounded retries
- [x] Improved invoice search
- [x] Ollama integration
- [x] ChatGroq integration
- [x] LangChain tool binding

### In Progress

- [ ] Better failure recovery
- [ ] Persistent memory
- [ ] Improved evaluation framework
- [ ] More complex planning
- [ ] Human approval workflows

### Planned

- [ ] Browser / computer use
- [ ] External API actions
- [ ] Long-term memory
- [ ] Agent performance benchmarks
- [ ] Multi-agent workflows
- [ ] More realistic AI-worker tasks

---

# Running the Project

After setup:

```bash
python -m backend.main
```

Then provide a task:

```text
Task> Find the latest Amazon invoice.
```

or:

```text
Task> Process the latest Amazon invoice.
```

The terminal shows the agent's execution process, including:

```text
[classify]
[goal]
[step 1]
[tool]
[result]
[step 2]
[tool]
[result]
...
[final]
```

This makes the agent's behavior observable during development and debugging.

---

# License

This project is currently developed as an experimental AI-agent engineering project for learning and experimentation.
