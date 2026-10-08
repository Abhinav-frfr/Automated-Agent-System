"""Agent loop using ChatGroq (LangChain).

Same external interface as before:
    agent = Agent()
    result = agent.run("Find the latest Amazon invoice")

Internals:
    - memory stores OpenAI-style dicts (unchanged)
    - messages are converted to LangChain objects at the LLM boundary
    - tool schemas in planner.TOOLS are passed straight to bind_tools()
    - finalize_answer is intercepted before tool execution
    - Groq's tool_use_failed errors are recovered by remapping the invented
      tool name to finalize_answer (see _try_recover_tool_use_failure)
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq

from agent.classifier import classify
from agent.memory import Memory
from agent.planner import SYSTEM_PROMPT_TEMPLATE, TOOLS
from tools.company_api import search_invoice, submit_invoice, verify_invoice
from tools.file_tools import read_file, search_files

load_dotenv()

TOOL_REGISTRY = {
    "search_files": search_files,
    "read_file": read_file,
    "search_invoice": search_invoice,
    "submit_invoice": submit_invoice,
    "verify_invoice": verify_invoice,
}

FINALIZE = "finalize_answer"
_THINK_RE = re.compile(r" thinking.*?", re.DOTALL | re.IGNORECASE)

_TRANSIENT = (TimeoutError, ConnectionError)

DEBUG_PATH = Path(__file__).resolve().parent.parent / "data" / "debug_messages.json"


class Agent:
    # Groq workaround: some models wrap the final answer in a tool call with
    # an invented name (commentary, json, output, ...). We remap to finalize.
    _TOOL_ALIASES = {
        "commentary": FINALIZE,
        "final": FINALIZE,
        "final_answer": FINALIZE,
        "finalize": FINALIZE,
        "respond": FINALIZE,
        "response": FINALIZE,
        "answer": FINALIZE,
        "json": FINALIZE,
        "output": FINALIZE,
        "result": FINALIZE,
        "data": FINALIZE,
        "note": FINALIZE,
    }

    def __init__(
        self,
        model: str | None = None,
        max_steps: int = 10,
        verbose: bool = True,
        tool_retries: int = 2,
        debug: bool = True,
        api_key: str | None = None,
        temperature: float = 0.1,
    ):
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.max_steps = max_steps
        self.verbose = verbose
        self.tool_retries = tool_retries
        self.debug = debug
        self.temperature = temperature

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Put it in .env or pass api_key=..."
            )

        self._llm = ChatGroq(
            model=self.model,
            api_key=key,
            temperature=temperature,
            max_retries=3,
            timeout=60,
        )

        self.memory = Memory()
        self._goal: dict = {}
        self._tool_history: list[dict] = []
        self._sig_counts: Counter = Counter()

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------ #
    # Message conversion (dict memory <-> LangChain messages)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(content or "")

    def _to_lc_messages(self, messages: list[dict]) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                out.append(SystemMessage(content=m.get("content", "")))
            elif role == "user":
                out.append(HumanMessage(content=m.get("content", "")))
            elif role == "assistant":
                content = _THINK_RE.sub(
                    "", self._coerce_content(m.get("content", ""))
                ).strip()
                raw_tcs = m.get("tool_calls") or []
                lc_tcs = []
                for i, tc in enumerate(raw_tcs):
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError:
                            args = {}
                    lc_tcs.append({
                        "id": tc.get("id") or f"call_{i}",
                        "name": fn.get("name", ""),
                        "args": args,
                    })
                if lc_tcs:
                    out.append(AIMessage(content=content, tool_calls=lc_tcs))
                else:
                    out.append(AIMessage(content=content))
            elif role == "tool":
                out.append(ToolMessage(
                    content=self._coerce_content(m.get("content", "")),
                    tool_call_id=m.get("tool_call_id") or "call_0",
                    name=m.get("tool_name", ""),
                ))
        return out

    def _from_lc_response(self, response: AIMessage) -> dict:
        content = _THINK_RE.sub("", self._coerce_content(response.content)).strip()

        tool_calls: list[dict] = []
        for i, tc in enumerate(getattr(response, "tool_calls", None) or []):
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            tool_calls.append({
                "id": tc_id or f"call_{i}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args if isinstance(args, dict) else {},
                },
            })

        return {"content": content, "tool_calls": tool_calls}

    # ------------------------------------------------------------------ #
    # LLM call — same signature as before
    # ------------------------------------------------------------------ #
    def _llm_chat(self, messages, tools=None, force_json: bool = False) -> dict:
        lc_messages = self._to_lc_messages(messages)

        if self.debug:
            try:
                DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
                DEBUG_PATH.write_text(
                    json.dumps(
                        [
                            {
                                "role": type(m).__name__,
                                "content": self._coerce_content(
                                    getattr(m, "content", "")
                                ),
                                **(
                                    {
                                        "tool_calls": [
                                            {"name": tc.get("name"), "args": tc.get("args")}
                                            for tc in (getattr(m, "tool_calls", None) or [])
                                        ]
                                    }
                                    if getattr(m, "tool_calls", None)
                                    else {}
                                ),
                            }
                            for m in lc_messages
                        ],
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        if tools is not None:
            llm = self._llm.bind_tools(tools)
        elif force_json:
            llm = self._llm.bind(response_format={"type": "json_object"})
        else:
            llm = self._llm

        try:
            response = llm.invoke(lc_messages)
        except Exception as e:
            recovered = self._try_recover_tool_use_failure(e)
            if recovered is not None:
                self._log(
                    f"    ⚠ recovered from Groq tool_use_failed → "
                    f"{recovered['tool_calls'][0]['function']['name']}"
                )
                return recovered
            raise RuntimeError(
                f"ChatGroq request failed: {type(e).__name__}: {e}"
            ) from e

        return self._from_lc_response(response)

    # ------------------------------------------------------------------ #
    # Groq tool_use_failed recovery
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_failed_generation(raw: str) -> tuple[str, dict] | None:
        if not raw:
            return None
        cleaned = re.sub(r"<\|.*?\|>", "", raw).strip()
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not m:
                return None
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None

        if not isinstance(obj, dict):
            return None

        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        return name, args

    def _try_recover_tool_use_failure(self, exc: Exception) -> dict | None:
        # Extract parsed error body from the SDK exception.
        body = getattr(exc, "body", None)
        if body is None:
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    body = resp.json()
                except Exception:
                    body = None
        if not isinstance(body, dict):
            return None

        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if not isinstance(err, dict) or err.get("code") != "tool_use_failed":
            return None

        parsed = self._parse_failed_generation(err.get("failed_generation") or "")
        if not parsed:
            return None
        name, args = parsed

        # Route 1: known alias (commentary, json, output, ...)
        mapped = self._TOOL_ALIASES.get(name.lower())

        # Route 2: matches a real tool name
        if mapped is None and name in TOOL_REGISTRY:
            mapped = name

        # Route 3: shape-based — arguments look like a finalize payload
        if mapped is None and isinstance(args, dict):
            if "summary" in args or "evidence" in args:
                mapped = FINALIZE

        if mapped is None:
            return None

        return {
            "content": "",
            "tool_calls": [{
                "id": "call_recovered",
                "type": "function",
                "function": {"name": mapped, "arguments": args},
            }],
        }

    # ------------------------------------------------------------------ #
    # Policy + finalize gate
    # ------------------------------------------------------------------ #
    def _policy_check(self, name: str, args: dict) -> dict | None:
        goal = self._goal or {}
        if name == "submit_invoice" and goal.get("task_type") != "WRITE":
            return {
                "error": (
                    f"Policy violation: submit_invoice is only allowed for WRITE goals "
                    f"(current task_type={goal.get('task_type')!r})."
                ),
                "blocked_by": "policy",
            }
        return None

    def _finalize_gate(self) -> dict | None:
        goal = self._goal or {}
        if goal.get("task_type") == "WRITE":
            ok = any(
                h["name"] == "verify_invoice"
                and isinstance(h.get("result"), dict)
                and h["result"].get("verified") is True
                for h in self._tool_history
            )
            if not ok:
                return {
                    "error": (
                        "Cannot finalize a WRITE goal before a verify_invoice call returns "
                        "verified=true. Call verify_invoice with the expected values."
                    ),
                    "blocked_by": "finalize_gate",
                }
        return None

    # ------------------------------------------------------------------ #
    # Tool execution with loop-breaker + bounded retries
    # ------------------------------------------------------------------ #
    def _execute_tool(self, name: str, args) -> dict:
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return {"error": f"Could not parse arguments: {args!r}"}
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {"error": f"Arguments must be an object, got {type(args).__name__}"}

        sig = f"{name}::{json.dumps(args, sort_keys=True, default=str)}"
        self._sig_counts[sig] += 1
        n = self._sig_counts[sig]
        if n >= 2:
            self._log(f"    ✗ loop-breaker: {name} called {n}× with identical args")
            return {
                "error": "REPEATED_CALL",
                "message": (
                    f"You have already called {name} with these exact arguments. "
                    f"The result will not change. Call finalize_answer NOW using the "
                    f"data you already have, or pick a DIFFERENT tool or DIFFERENT arguments."
                ),
                "repeat_count": n,
            }

        violation = self._policy_check(name, args)
        if violation:
            self._log(f"    ✗ policy: {violation['error']}")
            return violation

        func = TOOL_REGISTRY.get(name)
        if func is None:
            return {"error": f"Unknown tool: {name}"}

        attempt = 0
        while True:
            try:
                return func(**args)
            except _TRANSIENT as e:
                attempt += 1
                if attempt > self.tool_retries:
                    return {
                        "error": f"{type(e).__name__}: {e}",
                        "retries_exhausted": True,
                    }
                self._log(
                    f"    (transient error: {e}; retry {attempt}/{self.tool_retries})"
                )
                time.sleep(0.3 * attempt)
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, task: str) -> dict:
        self._log(f"[classify] {task}")
        goal = classify(self._llm_chat, task)
        self._log(f"[goal]     {json.dumps(goal)}")
        self._goal = goal
        self._tool_history = []
        self._sig_counts = Counter()

        self.memory.reset()
        self.memory.add(
            "system",
            SYSTEM_PROMPT_TEMPLATE.format(
                task_type=goal.get("task_type", "READ"),
                desired_outcome=goal.get("desired_outcome") or task,
            ),
        )
        self.memory.add("user", task)

        for step in range(1, self.max_steps + 1):
            msg = self._llm_chat(self.memory.get(), tools=TOOLS)
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            assistant_record: dict = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_record["tool_calls"] = tool_calls
            self.memory.add_raw(assistant_record)

            if not tool_calls:
                text = content.strip()
                return {
                    "status": "completed_no_finalize" if text else "incomplete",
                    "summary": text or "(model produced no output)",
                    "goal": goal,
                    "evidence": None,
                }

            for tc in tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                args = fn.get("arguments", {})

                # ---- finalize_answer is a control tool ----
                if name == FINALIZE:
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"summary": args}
                    args = args or {}
                    evidence = args.get("evidence", {})
                    if isinstance(evidence, str):
                        try:
                            evidence = json.loads(evidence)
                        except json.JSONDecodeError:
                            evidence = {"raw": evidence}

                    blocked = self._finalize_gate()
                    if blocked:
                        self._log(
                            f"\n[step {step}] ✗ finalize blocked: {blocked['error']}"
                        )
                        self.memory.add_raw({
                            "role": "tool",
                            "tool_name": FINALIZE,
                            "tool_call_id": tc.get("id") or "call_finalize",
                            "content": json.dumps(blocked),
                        })
                        continue

                    self._log(f"\n[step {step}] → finalize_answer(...)")
                    return {
                        "status": "completed",
                        "summary": (args.get("summary") or "").strip(),
                        "goal": goal,
                        "evidence": evidence,
                    }

                # ---- Ordinary tool ----
                self._log(f"\n[step {step}] → {name}({json.dumps(args, default=str)})")
                result = self._execute_tool(name, args)
                preview = json.dumps(result, default=str)
                self._log(
                    f"[step {step}] ← {preview[:400]}{'...' if len(preview) > 400 else ''}"
                )

                self._tool_history.append(
                    {"name": name, "args": args, "result": result}
                )

                note = ""
                if step >= 3:
                    note = (
                        f"\n\n[coordinator] step {step}/{self.max_steps}. "
                        f"If the DESIRED OUTCOME is achievable with the data above, "
                        f"call finalize_answer NOW. Do not repeat any tool you have "
                        f"already called."
                    )

                self.memory.add_raw({
                    "role": "tool",
                    "tool_name": name,
                    "tool_call_id": tc.get("id") or f"call_{step}",
                    "content": preview + note,
                })

        return {
            "status": "max_steps",
            "summary": "Max steps reached without finalizing.",
            "goal": goal,
            "evidence": None,
        }