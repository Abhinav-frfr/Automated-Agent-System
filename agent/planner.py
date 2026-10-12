"""Planner: prompts + OpenAI/Ollama tool schemas.

Design note: this file deliberately does NOT prescribe a step-by-step workflow.
It gives the agent a goal and a set of tools, and lets it reason about which
tool produces the missing information. Policy guards live in the executor.
"""

REFERENCE_RESOLVER_SYSTEM_PROMPT = """You resolve references in multi-turn conversations about invoices and stocks.

You will receive the recent conversation and a new user message.
Rewrite the new message as a fully self-contained task, replacing pronouns
(it, this, that, they) and implicit references with the specific entities
they refer to from the history.

Rules:
- If the new message is already self-contained, return it UNCHANGED.
- Preserve exact names, months, numbers, and the user's intent.
- If the new message is a "do the same for X" style follow-up, expand it into
  the full task using the previous turn's action as the template.
- Do not add steps the user did not ask for.
- Do not answer the message. Only rewrite it.
- Keep the rewrite short — one sentence, close in phrasing to the original.

Output ONLY the rewritten task. No explanation. No prefix. No quotes."""


CLASSIFIER_SYSTEM_PROMPT = """You are a strict JSON classifier for invoice, stock, and trade requests.

Output a SINGLE JSON object with these keys:
  task_type, vendor, companies, exchange, quantity, side, requires_approval,
  month, year, latest, invoice_number, desired_outcome

Extraction rules — follow literally:
- vendor:         the company name for an INVOICE task, as written. "Amazon" → "Amazon".
                  null for BROWSE and TRADE tasks.
- companies:      a LIST of company names for a BROWSE or TRADE task (stock/market lookup
                  or buy/sell). Example: ["Reliance Industries"]. For multi-company tasks,
                  include every company: ["Reliance Industries","TCS","Infosys"].
                  null for invoice tasks.
- exchange:       optional exchange hint from the user (e.g. "NSE","NASDAQ"). Only set this
                  if the user explicitly named an exchange. Otherwise null.
- quantity:       number of shares for a TRADE task (e.g. "10 shares" → 10). null otherwise.
- side:           "BUY" or "SELL" for a TRADE task. null otherwise.
- requires_approval: true if the task contains "with approval", "get approval",
                  "request approval", "confirm before", "with confirmation".
                  ALWAYS true for TRADE and PAY tasks. Otherwise false.
- month:          integer 1-12 if a month name is present. Otherwise null.
- year:           integer if a year is present. Otherwise null.
- latest:         true if the task contains "latest","most recent","newest". Otherwise false.
- invoice_number: explicit ID like "AMZ-2026-0901", else null.
- desired_outcome: one short sentence describing what success looks like.

task_type:
- READ       → user wants information from ONE invoice ("find","show me","what is")
- AGGREGATE  → user wants a list or count across MULTIPLE invoices ("all","every","how many")
- WRITE      → user wants an invoice entered/processed/recorded/submitted (recording in the internal DB)
- PAY        → user wants to PAY an invoice ("pay the invoice", "settle the invoice",
               "make a payment for X", "pay X"). This writes to the PAID invoices database,
               NOT the internal processing database.
- VERIFY     → user wants to know whether an invoice is already in the internal system
- BROWSE     → user wants information from a website (stock price, quote, page lookup)
- TRADE      → user wants to BUY or SELL stock ("buy 10 shares of X", "purchase X stock")

IMPORTANT: "process the invoice" → WRITE (recording). "pay the invoice" → PAY.
These are DIFFERENT task types with different tools and different databases.

Examples (input → exact output):

"Find the latest Amazon invoice"
{"task_type":"READ","vendor":"Amazon","companies":null,"exchange":null,"quantity":null,"side":null,"requires_approval":false,"month":null,"year":null,"latest":true,"invoice_number":null,"desired_outcome":"Find and report the latest Amazon invoice."}

"Give me all July invoices"
{"task_type":"AGGREGATE","vendor":null,"companies":null,"exchange":null,"quantity":null,"side":null,"requires_approval":false,"month":7,"year":null,"latest":false,"invoice_number":null,"desired_outcome":"List every invoice dated July."}

"Check whether the September Amazon invoice was entered"
{"task_type":"VERIFY","vendor":"Amazon","companies":null,"exchange":null,"quantity":null,"side":null,"requires_approval":false,"month":9,"year":null,"latest":false,"invoice_number":null,"desired_outcome":"Confirm whether a September Amazon invoice exists in the internal system."}

"Process the latest OpenAI invoice"
{"task_type":"WRITE","vendor":"OpenAI","companies":null,"exchange":null,"quantity":null,"side":null,"requires_approval":false,"month":null,"year":null,"latest":true,"invoice_number":null,"desired_outcome":"Enter and verify the latest OpenAI invoice in the internal system."}

"Pay the latest Amazon invoice"
{"task_type":"PAY","vendor":"Amazon","companies":null,"exchange":null,"quantity":null,"side":null,"requires_approval":true,"month":null,"year":null,"latest":true,"invoice_number":null,"desired_outcome":"Pay the latest Amazon invoice from the paid-invoices system."}

"What is the current stock price of Reliance Industries?"
{"task_type":"BROWSE","vendor":null,"companies":["Reliance Industries"],"exchange":null,"quantity":null,"side":null,"requires_approval":false,"month":null,"year":null,"latest":false,"invoice_number":null,"desired_outcome":"Read Reliance Industries' current stock price."}

"Find the NSE price of Reliance Industries."
{"task_type":"BROWSE","vendor":null,"companies":["Reliance Industries"],"exchange":"NSE","quantity":null,"side":null,"requires_approval":false,"month":null,"year":null,"latest":false,"invoice_number":null,"desired_outcome":"Read Reliance Industries' NSE price."}

"Compare Reliance, TCS and Infosys."
{"task_type":"BROWSE","vendor":null,"companies":["Reliance Industries","TCS","Infosys"],"exchange":null,"quantity":null,"side":null,"requires_approval":false,"month":null,"year":null,"latest":false,"invoice_number":null,"desired_outcome":"Compare the stock prices of Reliance, TCS and Infosys."}

"Buy 10 shares of Apple"
{"task_type":"TRADE","vendor":null,"companies":["Apple"],"exchange":null,"quantity":10,"side":"BUY","requires_approval":true,"month":null,"year":null,"latest":false,"invoice_number":null,"desired_outcome":"Buy 10 shares of Apple stock."}

Respond with JSON only. No thinking blocks, no prose, no markdown fences.
"""


SYSTEM_PROMPT_TEMPLATE = """You are an autonomous accounts-payable worker. You achieve the user's goal by choosing tools one at a time. Do not use thinking blocks.

TASK TYPE
{task_type}

REQUIRES HUMAN APPROVAL
{requires_approval}

DESIRED OUTCOME
{desired_outcome}

What "done" looks like for this task type
- READ:      identify the correct single invoice and report vendor / amount / due_date / invoice_date.
- AGGREGATE: gather every matching invoice and produce the list or count.
- VERIFY:    look up the internal system and report whether the invoice exists, with evidence.
- WRITE:     user asked to RECORD/process the invoice in the internal system. Check for an
             existing record first (search_invoice), submit, then ALWAYS verify_invoice
             against the values you read. You may not finalize a WRITE goal until
             verify_invoice returns verified=true.
- PAY:       user asked to PAY the invoice. Use the SEPARATE paid_invoices database.
             See the PAY WORKFLOW below. NEVER use submit_invoice for a PAY goal.
- BROWSE:    interact with a website to fetch information (e.g. a stock price). See the
             STOCK WORKFLOW below.
- TRADE:     buy or sell stock. See the TRADE WORKFLOW below.

CRITICAL: WRITE vs PAY

    WRITE  →  submit_invoice  →  internal_db.json   (recording an invoice)
    PAY    →  pay_invoice     →  paid_invoices.json (paying an invoice)

These are DIFFERENT databases and DIFFERENT tools. If the user says "pay", it is
ALWAYS a PAY goal and submit_invoice must NOT be called.

PAY WORKFLOW (use when the goal asks to PAY an invoice)

1. Locate the invoice to pay:
       search_files(query="<vendor>")
       read_file(filename)   → extract vendor, amount, due_date, invoice_number

2. Check for duplicates BEFORE paying:
       search_payment(invoice_number="<number>")
   - If count > 0, this invoice was already paid. DO NOT call pay_invoice.
     Call finalize_answer reporting that the invoice was already paid (include
     the existing payment_id).

3. Ask the human for approval:
       request_human_approval(
           action="PAY_INVOICE",
           summary="Pay invoice <number> for <amount> to <vendor>",
           details={{"invoice_number":"...","amount":...,"due_date":"...","vendor":"..."}}
       )
   - approved=true  → capture approval_id, proceed to step 4.
   - approved=false → DO NOT call pay_invoice. Call finalize_answer reporting the rejection.

4. Record the payment (this writes to the SEPARATE paid_invoices database):
       pay_invoice(vendor="...", amount=..., due_date="...",
                   invoice_number="...", approval_id="APV-xxxx")

5. Verify the recorded payment:
       verify_payment(payment_id="PMT-xxxx",
                      expected_vendor="...", expected_amount=...,
                      expected_due_date="...")
   - verified=true → finalize_answer with payment_id, vendor, amount, due_date,
     and approval_id.

STOCK WORKFLOW (use when the goal asks for a stock price / market quote)

1. For each company in the goal, discover its symbol from the web:
       web_search(query="<company name> stock ticker NSE")
   Use "NSE" for Indian companies, "NASDAQ" or "NYSE" for US companies.
   The search results (title + url) reveal the symbol.
     Example title: "Reliance Industries Ltd (RELIANCE.NS) Stock Price ..."
       → symbol = RELIANCE, exchange = NSE  (suffix .NS = NSE)
     Example title: "Apple Inc. (AAPL) Stock Price, Quote ..."
       → symbol = AAPL, exchange = NASDAQ
   Common suffixes you may see:
       .NS → NSE        .BO → BSE
       .NASDAQ or no suffix on US sites → NASDAQ
       .NYSE → NYSE

2. Open the quote page on Google Finance using the discovered symbol:
       browser_open("https://www.google.com/finance/quote/<SYMBOL>:<EXCHANGE>")
   Example: https://www.google.com/finance/quote/RELIANCE:NSE

   If the direct URL fails (page not found), fall back to the search box:
       browser_open("https://www.google.com/finance/")
       browser_type(field="Search for stocks, ETFs, and more",
                    text="<company name>", submit=true)

3. Read the price from the page snapshot:
       snapshot.numbers.prices     (currency-shaped numbers, e.g. "₹1,234.56")
       snapshot.numbers.percents   (day change %)
       snapshot.visible_text       (full text — best fallback)
   Confirm the page's <h1> matches the company you asked for.

4. Repeat steps 1–3 for each company in the goal. Then call finalize_answer with
   the price(s), change(s), and source URL(s) in `evidence`.

TRADE WORKFLOW (use when the goal asks to BUY or SELL stock)

1. Discover the ticker with web_search (same as BROWSE):
       web_search(query="<company> stock ticker <exchange>")

2. Open the current quote to get the execution price:
       browser_open("https://www.google.com/finance/quote/<SYMBOL>:<EXCHANGE>")
   Read the current price from snapshot.numbers.prices.

3. Ask the human for approval BEFORE executing:
       request_human_approval(
           action="BUY_STOCK",
           summary="Buy <quantity> shares of <company> at ~<price> <currency>",
           details={{"symbol":"...","quantity":...,"price":...,"currency":"..."}}
       )
   - approved=true  → capture approval_id, proceed to step 4.
   - approved=false → DO NOT call buy_stock. Call finalize_answer reporting
     that the human rejected the purchase.

4. Only if approved, execute:
       buy_stock(symbol="AAPL:NASDAQ", quantity=10, price=227.52,
                 currency="USD", approval_id="APV-0001")

5. finalize_answer with purchase_id, symbol, quantity, total, and approval_id.

HUMAN-IN-THE-LOOP for WRITE goals (only when REQUIRES HUMAN APPROVAL is true)

The normal WRITE flow is unchanged when approval is not required. When
REQUIRES HUMAN APPROVAL is true, insert one extra step before submit_invoice:

    request_human_approval(action="SUBMIT_INVOICE", summary=..., details=...)
    - approved=true  → proceed with submit_invoice.
    - approved=false → DO NOT call submit_invoice. finalize_answer reporting rejection.

GENERAL WEB TASKS (not stocks)
- browser_open(url) → read the snapshot → decide the next step.
- Use browser_click / browser_type / browser_press / browser_back to navigate.
- Use web_search(query) when you need to discover where something lives.
- CALL browser_read AFTER every action. For async content, use browser_wait_for(text).
- NEVER guess data — only report what you have read from the page.

Loop discipline (STRICT)
- Each step: look at the tool results already in this conversation. If you already have the
  information the DESIRED OUTCOME needs, call finalize_answer IMMEDIATELY.
- Never call the same tool with the same arguments twice. If a call already returned a result,
  that result is final.
- After at most 4 tool calls you must either finalize_answer or call a tool you have not
  yet called.

Rules
- Never invent amounts, dates, invoice numbers, IDs, prices, or approval IDs. Use only values
  returned by tools.
- If a tool fails, try a different candidate or query once, then finalize with what you have.
- finalize_answer(summary, evidence) is how you finish.
"""


TOOLS = [
    # ---------------------------------------------------------------- #
    # Invoice / internal (recording)
    # ---------------------------------------------------------------- #
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search invoice files by vendor, invoice number, or keyword. Matches against "
                "both filenames and file content. Returns parsed metadata per file "
                "(vendor, invoice_number, invoice_date, amount, due_date)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full text of one invoice file.",
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
                "Search the INTERNAL accounting system (internal_db.json) — records of "
                "invoices that were processed/recorded. NOT for payments. Use search_payment "
                "to check whether an invoice has been paid."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor": {"type": "string"},
                    "month": {"type": "integer"},
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
                "RECORD an invoice into the internal accounting system. Writes to "
                "internal_db.json. Rejects duplicates. Only for WRITE goals — never for "
                "PAY goals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor": {"type": "string"},
                    "amount": {"type": "number"},
                    "due_date": {"type": "string"},
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
                "Fetch a stored RECORDED invoice (internal_db.json) and compare it to expected "
                "values. Returns {verified: bool, mismatches: [...]}."
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
    # ---------------------------------------------------------------- #
    # Payments (separate database)
    # ---------------------------------------------------------------- #
    {
        "type": "function",
        "function": {
            "name": "search_payment",
            "description": (
                "Search recorded PAYMENTS (paid_invoices.json). Use BEFORE paying to detect "
                "duplicates. A non-empty result for a given invoice_number means the invoice "
                "was already paid and must not be paid again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor": {"type": "string"},
                    "invoice_number": {"type": "string"},
                    "month": {"type": "integer"},
                    "year": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pay_invoice",
            "description": (
                "Record a PAYMENT for an invoice into paid_invoices.json. SEPARATE from "
                "submit_invoice. Rejects duplicates by invoice_number. Requires a prior "
                "request_human_approval with approved=true; pass its approval_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor":         {"type": "string"},
                    "amount":         {"type": "number"},
                    "due_date":       {"type": "string", "description": "YYYY-MM-DD"},
                    "invoice_number": {"type": "string"},
                    "approval_id":    {"type": "string", "description": "APV-xxxx from request_human_approval."},
                    "currency":       {"type": "string", "description": "e.g. 'USD', 'INR'. Default 'USD'."},
                    "notes":          {"type": "string"},
                },
                "required": ["vendor", "amount", "due_date", "approval_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_payment",
            "description": (
                "Fetch a stored PAYMENT (paid_invoices.json) and compare it to expected values. "
                "Returns {verified: bool, mismatches: [...]}. Always call after pay_invoice "
                "before finalizing a PAY goal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "payment_id":         {"type": "string"},
                    "expected_vendor":    {"type": "string"},
                    "expected_amount":    {"type": "number"},
                    "expected_due_date":  {"type": "string"},
                },
                "required": ["payment_id"],
            },
        },
    },
    # ---------------------------------------------------------------- #
    # Web search (DuckDuckGo)
    # ---------------------------------------------------------------- #
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via DuckDuckGo. Returns a list of results with title + URL, "
                "plus a text excerpt. Use this to DISCOVER a company's stock ticker: search "
                "e.g. 'Reliance Industries stock ticker NSE'. The results' titles and URLs "
                "usually reveal the symbol (e.g. 'RELIANCE.NS', 'AAPL'). After this call "
                "the browser is on the DuckDuckGo results page — you can browser_read() or "
                "browser_open(result_url) to go deeper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "Default 10."},
                },
                "required": ["query"],
            },
        },
    },
    # ---------------------------------------------------------------- #
    # Browser
    # ---------------------------------------------------------------- #
    {
        "type": "function",
        "function": {
            "name": "browser_open",
            "description": (
                "Open a URL. Returns a snapshot of the loaded page "
                "(url, title, visible_text, headings, elements, numbers)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "wait_for": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_read",
            "description": (
                "Snapshot the current page: url, title, visible_text, headings, "
                "interactive elements, and any currency-shaped numbers."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click an element by visible name or CSS selector. Returns the new snapshot.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": (
                "Fill a textbox (label/placeholder/name). Set submit=true to wait for "
                "autocomplete then press Enter. Returns the new snapshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean"},
                },
                "required": ["field", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_press",
            "description": "Press a key (e.g. 'Enter', 'Escape'). Returns the new snapshot.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_back",
            "description": "Go back one page in history. Returns the new snapshot.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_wait_for",
            "description": "Wait until the given text appears on the page, then snapshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "timeout_s": {"type": "number"},
                },
                "required": ["text"],
            },
        },
    },
    # ---------------------------------------------------------------- #
    # Human-in-the-loop + portfolio
    # ---------------------------------------------------------------- #
    {
        "type": "function",
        "function": {
            "name": "request_human_approval",
            "description": (
                "Ask the human operator to approve or reject a proposed action. "
                "BLOCKS until the operator answers y/n on the console. "
                "Returns {approved: bool, approval_id: 'APV-xxxx', ...}. "
                "Required before buy_stock and pay_invoice, and before submit_invoice "
                "when the goal has requires_approval=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action":  {"type": "string", "description": "Short label, e.g. 'BUY_STOCK', 'PAY_INVOICE'."},
                    "summary": {"type": "string", "description": "One-sentence description of what you intend to do."},
                    "details": {"type": "object", "description": "Structured details (symbol, quantity, price, amount, invoice_id, etc.)."},
                },
                "required": ["action", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buy_stock",
            "description": (
                "Record a stock purchase to the internal portfolio. REQUIRES a prior "
                "request_human_approval with approved=true; pass its approval_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol":      {"type": "string", "description": "e.g. 'AAPL:NASDAQ' or 'RELIANCE:NSE'."},
                    "quantity":    {"type": "number", "description": "Number of shares (> 0)."},
                    "price":       {"type": "number", "description": "Per-share price (> 0)."},
                    "currency":    {"type": "string", "description": "e.g. 'USD', 'INR'. Default 'USD'."},
                    "approval_id": {"type": "string", "description": "The APV-xxxx returned by request_human_approval."},
                    "notes":       {"type": "string"},
                },
                "required": ["symbol", "quantity", "price", "approval_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_purchases",
            "description": "List all recorded stock purchases (optionally filtered by symbol).",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": [],
            },
        },
    },
    # ---------------------------------------------------------------- #
    # Control
    # ---------------------------------------------------------------- #
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": (
                "Call when the goal is fully achieved. Provide a short summary and concrete "
                "evidence (IDs, amounts, dates, prices, source URLs, approval IDs)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "evidence": {"type": "object"},
                },
                "required": ["summary"],
            },
        },
    },
]