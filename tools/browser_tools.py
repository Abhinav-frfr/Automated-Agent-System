"""Browser automation tools backed by Playwright.

Includes:
- browser_open / read / click / type / press / back / wait_for / wait
- web_search(query) — DuckDuckGo search returning structured results

All browser actions share a persistent session per task; browser_reset() tears
it down. This is the only place in the agent that touches Playwright.
"""

from __future__ import annotations

import re
import threading
from urllib.parse import parse_qs, quote_plus, unquote, urlparse


_PRICE_RE = re.compile(
    r"(?:(?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP|¥|JPY)\s?)"
    r"\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE,
)
_PCT_RE = re.compile(r"[+\-]?\d+(?:\.\d+)?\s?%")


class BrowserManager:
    def __init__(self, headless: bool = False, default_timeout_ms: int = 12000):
        self.headless = headless
        self.default_timeout_ms = default_timeout_ms
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

    # ---------------- lifecycle ---------------- #
    def _ensure_page(self):
        if self._page is not None:
            return self._page
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        # headless=False avoids Google / DDG anti-bot blocks.
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            viewport={"width": 1400, "height": 900},
            locale="en-IN",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(self.default_timeout_ms)
        return self._page

    def close(self):
        for obj in (self._page, self._context, self._browser):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass
        self._pw = self._browser = self._context = self._page = None

    # ---------------- consent handling ---------------- #
    def _dismiss_consent(self, page):
        candidates = [
            "Accept all", "I agree", "Accept", "Got it", "OK",
            "Agree to all", "Accept All",
        ]
        for label in candidates:
            try:
                btn = page.get_by_role("button", name=label, exact=False).first
                if btn.is_visible(timeout=800):
                    btn.click(timeout=1500)
                    page.wait_for_timeout(400)
                    return True
            except Exception:
                continue
        return False

    # ---------------- actions ---------------- #
    def open(self, url: str, wait_for: str | None = None) -> dict:
        try:
            page = self._ensure_page()
            page.goto(url, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            self._dismiss_consent(page)
            if wait_for:
                try:
                    page.wait_for_selector(
                        f"text={wait_for}", timeout=self.default_timeout_ms
                    )
                except Exception:
                    pass
            return self.read()
        except Exception as e:
            return {"error": f"browser_open failed: {type(e).__name__}: {e}"}

    def read(self) -> dict:
        try:
            page = self._ensure_page()
            url = page.url
            try:
                title = page.title()
            except Exception:
                title = ""
            try:
                text = page.locator("body").first.inner_text(timeout=3000)
            except Exception:
                text = ""
            if len(text) > 4000:
                text = text[:4000] + "\n...[truncated]"

            return {
                "url": url,
                "title": title,
                "visible_text": text,
                "headings": self._extract_headings(page),
                "elements": self._extract_elements(page),
                "numbers": self._extract_numbers(text),
            }
        except Exception as e:
            return {"error": f"browser_read failed: {type(e).__name__}: {e}"}

    def click(self, target: str) -> dict:
        try:
            page = self._ensure_page()
        except Exception as e:
            return {"error": f"browser_click: no page yet ({e})"}

        attempts = [
            lambda: page.get_by_role("button", name=target, exact=False).first.click(timeout=3500),
            lambda: page.get_by_role("link",   name=target, exact=False).first.click(timeout=3500),
            lambda: page.get_by_role("option", name=target, exact=False).first.click(timeout=3500),
            lambda: page.get_by_role("menuitem", name=target, exact=False).first.click(timeout=3500),
            lambda: page.get_by_role("tab", name=target, exact=False).first.click(timeout=3500),
            lambda: page.get_by_text(target, exact=False).first.click(timeout=3500),
            lambda: page.locator(target).first.click(timeout=3500),
        ]
        last_err = None
        for fn in attempts:
            try:
                fn()
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                return self.read()
            except Exception as e:
                last_err = e
                continue
        return {"error": f"browser_click: could not click {target!r} ({type(last_err).__name__})"}

    def type(self, field: str, text: str, submit: bool = False) -> dict:
        try:
            page = self._ensure_page()
        except Exception as e:
            return {"error": f"browser_type: no page yet ({e})"}

        attempts = [
            lambda: page.get_by_role("textbox", name=field, exact=False).first,
            lambda: page.get_by_role("combobox", name=field, exact=False).first,
            lambda: page.get_by_role("searchbox", name=field, exact=False).first,
            lambda: page.get_by_placeholder(field, exact=False).first,
            lambda: page.locator(f"input[aria-label='{field}']").first,
            lambda: page.locator(f"input[name='{field}']").first,
            lambda: page.locator(f"input[placeholder*='{field}' i]").first,
        ]
        last_err = None
        for fn in attempts:
            try:
                loc = fn()
                loc.click(timeout=3000)
                loc.fill("")
                loc.type(text, delay=40)
                if submit:
                    page.wait_for_timeout(700)
                    loc.press("Enter")
                    try:
                        page.wait_for_load_state("networkidle", timeout=6000)
                    except Exception:
                        pass
                return self.read()
            except Exception as e:
                last_err = e
                continue
        return {"error": f"browser_type: could not type into {field!r} ({type(last_err).__name__})"}

    def press(self, key: str) -> dict:
        try:
            page = self._ensure_page()
            page.keyboard.press(key)
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            return self.read()
        except Exception as e:
            return {"error": f"browser_press failed: {type(e).__name__}: {e}"}

    def back(self) -> dict:
        try:
            page = self._ensure_page()
            page.go_back(wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            return self.read()
        except Exception as e:
            return {"error": f"browser_back failed: {type(e).__name__}: {e}"}

    def wait_for(self, text: str, timeout_s: float = 10.0) -> dict:
        try:
            page = self._ensure_page()
            page.wait_for_selector(f"text={text}", timeout=int(timeout_s * 1000))
            return self.read()
        except Exception as e:
            return {"error": f"browser_wait_for: text {text!r} not seen ({type(e).__name__})"}

    def wait(self, seconds: float = 1.0) -> dict:
        try:
            page = self._ensure_page()
            page.wait_for_timeout(int(seconds * 1000))
            return self.read()
        except Exception as e:
            return {"error": f"browser_wait failed: {type(e).__name__}: {e}"}

    # ---------------- web search ---------------- #
    def web_search(self, query: str, max_results: int = 10) -> dict:
        """Search the web via DuckDuckGo and return structured results."""
        try:
            page = self._ensure_page()
            url = f"https://duckduckgo.com/?q={quote_plus(query)}&ia=web"
            page.goto(url, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=6000)
            except Exception:
                pass
            self._dismiss_consent(page)
            page.wait_for_timeout(500)

            results = self._extract_search_results(page, max_results=max_results)
            snapshot = self.read()

            return {
                "query": query,
                "engine": "duckduckgo",
                "url": snapshot.get("url"),
                "count": len(results),
                "results": results,
                "visible_text_excerpt": (snapshot.get("visible_text") or "")[:2000],
            }
        except Exception as e:
            return {"error": f"web_search failed: {type(e).__name__}: {e}"}

    @staticmethod
    def _extract_search_results(page, max_results: int = 10) -> list[dict]:
        out: list[dict] = []
        seen_urls: set[str] = set()

        selectors = [
            "a[data-testid='result-title-a']",
            "article h2 a",
            "article h3 a",
            ".result__a",
            ".result__title a",
            "h2 a",
            "h3 a",
        ]

        def _unwrap(href: str) -> str:
            if "duckduckgo.com/l/" in href or "duckduckgo.com/y.js" in href:
                try:
                    qs = parse_qs(urlparse(href).query)
                    if "uddg" in qs:
                        return unquote(qs["uddg"][0])
                except Exception:
                    pass
            return href

        for sel in selectors:
            try:
                loc = page.locator(sel)
                count = min(loc.count(), max_results * 2)
                for i in range(count):
                    if len(out) >= max_results:
                        break
                    try:
                        el = loc.nth(i)
                        title = (el.inner_text() or "").strip()
                        href = el.get_attribute("href") or ""
                        if not title or not href:
                            continue
                        if href.startswith("/") or href.startswith("#"):
                            continue
                        href = _unwrap(href)
                        if "duckduckgo.com" in href:
                            continue
                        if href in seen_urls:
                            continue
                        seen_urls.add(href)
                        out.append({"title": title, "url": href})
                    except Exception:
                        continue
            except Exception:
                continue
            if out:
                break
        return out

    # ---------------- snapshot extraction ---------------- #
    @staticmethod
    def _extract_headings(page) -> list[str]:
        out: list[str] = []
        for sel in ("h1", "h2", "h3"):
            try:
                for el in page.locator(sel).all()[:15]:
                    t = (el.inner_text() or "").strip()
                    if t and len(t) < 200:
                        out.append(t)
            except Exception:
                continue
        return out

    @staticmethod
    def _extract_elements(page) -> list[dict]:
        raw: list[dict] = []

        def collect(selector: str, role: str):
            try:
                for el in page.locator(selector).all()[:30]:
                    try:
                        t = (el.inner_text() or "").strip()
                        if not t:
                            t = (
                                el.get_attribute("aria-label")
                                or el.get_attribute("placeholder")
                                or el.get_attribute("name")
                                or el.get_attribute("value")
                                or ""
                            ).strip()
                        if t and len(t) < 120:
                            raw.append({"role": role, "name": t})
                    except Exception:
                        continue
            except Exception:
                pass

        collect("a", "link")
        collect("button, input[type='submit'], input[type='button']", "button")
        collect(
            "input[type='text'], input[type='search'], input:not([type]), textarea",
            "textbox",
        )
        collect("[role='option'], [role='menuitem'], [role='tab']", "option")

        seen = set()
        out = []
        for e in raw:
            key = (e["role"], e["name"])
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    @staticmethod
    def _extract_numbers(text: str) -> dict:
        return {
            "prices": list(dict.fromkeys(_PRICE_RE.findall(text)))[:20],
            "percents": list(dict.fromkeys(_PCT_RE.findall(text)))[:20],
        }


# ---------------- module singleton ---------------- #
_manager: BrowserManager | None = None
_lock = threading.Lock()


def _get_manager() -> BrowserManager:
    global _manager
    with _lock:
        if _manager is None:
            _manager = BrowserManager()
        return _manager


# ---------------- public tools ---------------- #

def browser_open(url: str, wait_for: str | None = None) -> dict:
    """Open a URL. Optional `wait_for` text blocks until that text appears. Returns a page snapshot."""
    return _get_manager().open(url, wait_for=wait_for)


def browser_read() -> dict:
    """Snapshot the current page: url, title, visible_text, headings, elements, numbers."""
    return _get_manager().read()


def browser_click(target: str) -> dict:
    """Click an element by visible name (button/link/option) or CSS selector. Returns the new snapshot."""
    return _get_manager().click(target)


def browser_type(field: str, text: str, submit: bool = False) -> dict:
    """Fill a textbox (label/placeholder/name). submit=True waits for autocomplete then presses Enter. Returns the new snapshot."""
    return _get_manager().type(field, text, submit=submit)


def browser_press(key: str) -> dict:
    """Press a keyboard key, e.g. 'Enter', 'Escape', 'Tab'. Returns the new snapshot."""
    return _get_manager().press(key)


def browser_back() -> dict:
    """Go back one page in history. Returns the new snapshot."""
    return _get_manager().back()


def browser_wait_for(text: str, timeout_s: float = 10.0) -> dict:
    """Wait until `text` appears on the page, then return a snapshot."""
    return _get_manager().wait_for(text, timeout_s=timeout_s)


def browser_wait(seconds: float = 1.0) -> dict:
    """Sleep N seconds, then return a snapshot."""
    return _get_manager().wait(seconds)


def web_search(query: str, max_results: int = 10) -> dict:
    """Search the web via DuckDuckGo. Returns:
       {query, engine, url, count, results: [{title, url}, ...], visible_text_excerpt}.

       Use this to DISCOVER a company's stock ticker from search results
       (titles/URLs usually reveal it, e.g. "RELIANCE.NS" or "AAPL"). After
       this call the browser is on the DuckDuckGo results page — you can
       browser_read() or browser_open(result_url) to go deeper."""
    return _get_manager().web_search(query, max_results=max_results)


def browser_reset() -> dict:
    """Tear down the current browser session. Called at the start of each task."""
    global _manager
    with _lock:
        if _manager is not None:
            _manager.close()
        _manager = None
    return {"ok": True}