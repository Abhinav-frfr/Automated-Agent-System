"""Planner: prompts + OpenAI/Ollama tool schemas.

Design note: this file deliberately does NOT prescribe a step-by-step workflow.
It gives the agent a goal and a set of tools, and lets it reason about which
tool produces the missing information. Policy guards live in the executor.
"""

CLASSIFIER_SYSTEM_PROMPT = """You are a strict JSON classifier for invoice requests.

Output a SINGLE JSON object with these keys:
  task_type, vendor, month, year, latest, invoice_number, desired_outcome

Extraction rules — follow literally:
- vendor:       the company name AS WRITTEN in the task. "Amazon" → "Amazon". Do not null it if the word is present.
- month:        integer 1-12 if a month name is present (July=7, September=9). Otherwise null.
- year:         integer if a year is present. Otherwise null.
- latest:       true whenever the task contains "latest", "most recent", "newest", or "most recent one". Otherwise false.
- invoice_number: only if the task contains an explicit ID like "AMZ-2026-0901". Otherwise null.
- desired_outcome: one short sentence describing what success looks like.

task_type:
- READ       → user wants information from ONE invoice ("find", "show me", "what is", "extract")
- AGGREGATE  → user wants a list or count across MULTIPLE ("all", "every", "list", "how many", "total")
- WRITE      → user wants an invoice entered/processed/recorded/submitted
- VERIFY     → user wants to know whether something is already in the internal system ("was X entered", "is Y recorded", "check if")

Examples (input → exact output):

"Find the latest Amazon invoice"
{"task_type":"READ","vendor":"Amazon","month":null,"year":null,"latest":true,"invoice_number":null,"desired_outcome":"Find and report the latest Amazon invoice."}

"Give me all July invoices"
{"task_type":"AGGREGATE","vendor":null,"month":7,"year":null,"latest":false,"invoice_number":null,"desired_outcome":"List every invoice dated July."}

"Check whether the September Amazon invoice was entered"
{"task_type":"VERIFY","vendor":"Amazon","month":9,"year":null,"latest":false,"invoice_number":null,"desired_outcome":"Confirm whether a September Amazon invoice exists in the internal system."}

"Process the latest OpenAI invoice"
{"task_type":"WRITE","vendor":"OpenAI","month":null,"year":null,"latest":true,"invoice_number":null,"desired_outcome":"Enter and verify the latest OpenAI invoice in the internal system."}

"What is the amount on the ABC Corp August invoice?"
{"task_type":"READ","vendor":"ABC Corp","month":8,"year":null,"latest":false,"invoice_number":null,"desired_outcome":"Report the amount due on the August ABC Corp invoice."}

"How many invoices did we get in September?"
{"task_type":"AGGREGATE","vendor":null,"month":9,"year":null,"latest":false,"invoice_number":null,"desired_outcome":"Count invoices dated September."}

Respond with JSON only. No thinking blocks, no prose, no markdown fences.
"""


SYSTEM_PROMPT_TEMPLATE = """You are an autonomous accounts-payable worker. You achieve the user's goal by choosing tools one at a time. Do not use thinking blocks.

TASK TYPE
{task_type}

DESIRED OUTCOME
{desired_outcome}

What "done" looks like for this task type
- READ:      identify the correct single invoice and report vendor / amount / due_date / invoice_date.
- AGGREGATE: gather every matching invoice and produce the list or count.
- VERIFY:    look up the internal system and report whether the invoice exists, with evidence.
- WRITE:     only submit because the user asked to process/enter/record. Check for an existing
             record first (search_invoice), submit, then ALWAYS verify_invoice against the
             values you read. You may not finalize a WRITE goal until verify_invoice returns
             verified=true.

Loop discipline (STRICT)
- Each step: look at the tool results already in this conversation. If you already have the
  information the DESIRED OUTCOME needs, call finalize_answer IMMEDIATELY.
- Never call the same tool with the same arguments twice. If a call already returned a result,
  that result is final.
- After at most 3 tool calls you must either finalize_answer or call a tool you have not
  yet called.

Rules
- Never invent amounts, dates, invoice numbers, or IDs. Use only values returned by tools.
- If a tool fails, try a different candidate or a different query once, then finalize with
  what you have.
- finalize_answer(summary, evidence) is how you finish.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search invoice files by vendor, invoice number, or keyword. Matches against "
                "both filenames and file content. Returns parsed metadata per file "
                "(vendor, invoice_number, invoice_date, amount, due_date) so you can compare "
                "candidates without reading every file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'Amazon', 'ABC Corp', 'INV-2026-0901', or '' to list every invoice file.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full text of one invoice file. Use when parsed metadata from "
                "search_files is missing a field or when you need to double-check a value."
            ),
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_invoice",
            "description": (
                "Search the internal accounting system. Use for duplicate checks, VERIFY goals, "
                "or AGGREGATE goals over already-submitted invoices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor": {"type": "string"},
                    "month": {"type": "integer", "description": "1-12, filter by due-date month."},
                    "year": {"type": "integer"},
                    "invoice_number": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_invoice",
            "description": (
                "Submit an invoice into the internal system. The tool rejects duplicates "
                "(same vendor + amount + due_date, or same invoice_number). "
                "Only use this for WRITE goals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor": {"type": "string"},
                    "amount": {"type": "number"},
                    "due_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "invoice_number": {"type": "string"},
                },
                "required": ["vendor", "amount", "due_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_invoice",
            "description": (
                "Fetch a stored invoice and compare it to expected values. Returns "
                "{verified: bool, mismatches: [...]}. Always call this after submit_invoice "
                "before finalizing a WRITE goal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string"},
                    "expected_vendor": {"type": "string"},
                    "expected_amount": {"type": "number"},
                    "expected_due_date": {"type": "string"},
                },
                "required": ["invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": (
                "Call when the goal is fully achieved. Provide a short summary and concrete "
                "evidence (IDs, amounts, dates, verification result)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "evidence": {
                        "type": "object",
                        "description": (
                            "Structured evidence, e.g. "
                            "{invoice_id, vendor, amount, due_date, verification} "
                            "or {count, invoices: [...]}."
                        ),
                    },
                },
                "required": ["summary"],
            },
        },
    },
]