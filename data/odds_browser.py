"""
Fetching a JSON endpoint through a real browser.

WHY THIS EXISTS. Two of the books answer a browser and refuse a script.
PrizePicks sits behind Cloudflare and has always done so; DraftKings served
this machine 367 KB on a Sunday morning and 403'd the same URL, from the same
IP, with the same headers, an hour and twenty requests later. In both cases
the data is visibly there - paste the URL into the address bar and it loads.

WHERE THE LINE IS, BECAUSE THIS IS EXACTLY THE PLACE IT GETS CROSSED.

What this does: launches an ordinary Chromium, navigates to the URL, and
reads what comes back. It is a browser. That is the entire trick.

What this deliberately does NOT do, and what the rest of this project has
refused throughout:

  - no undetected-chromedriver, no stealth or evasion plugins
  - no navigator.webdriver patching or fingerprint spoofing
  - no canvas/WebGL noise, no timezone or locale faking
  - no proxy rotation, no residential exit nodes
  - no CAPTCHA solving, no Cloudflare/Datadome challenge bypass
  - no credential use, no paywall circumvention

The distinction being drawn: sending `User-Agent: Chrome` from a Python
script is a claim that isn't true. Driving Chrome is not a claim at all. One
is a lie about what you are, the other is being the thing. This project has
declined the first all along and that has not changed.

It is still possible for a book to refuse this. Cloudflare can and does
identify headless browsers, and if it says no, that is the answer - there is
no second attempt wearing a different hat. Set NFLSCHOLAR_BROWSER_HEADED=1 to
run it visibly, which is a normal browser doing a normal thing and is more
likely to be served, at the cost of a window appearing.

Requests go out at the same rate a person clicking would produce, and every
caller of this is behind a 30-minute cache.

OPTIONAL DEPENDENCY. Playwright is not in requirements.txt because it pulls
a browser binary and most of this app does not need one. Without it the
fallback reports itself unavailable and the manual save-from-browser path
still works exactly as before:

    pip install playwright
    playwright install chromium
"""
import contextlib
import json
import os

BROWSER_TIMEOUT_MS = 30000

# Set to 1 to watch it work, or when a book refuses the headless one.
HEADED_ENV = 'NFLSCHOLAR_BROWSER_HEADED'
# Point at a Chrome/Chromium binary directly, for a machine where the
# installed Playwright and the available browser build disagree on version.
BROWSER_PATH_ENV = 'NFLSCHOLAR_BROWSER_PATH'


def browser_available():
    """Can we drive a browser at all? Returns (bool, explanation)."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        return False, ("Playwright isn't installed, so the browser fallback is off. "
                       "`pip install playwright` then `playwright install chromium` "
                       "turns it on. Until then, saving the JSON from your own "
                       "browser does the same job by hand.")
    return True, ''


def _launch(playwright):
    """
    A plain Chromium, preferring the Chrome already on the machine.

    Order matters here and is deliberate: `channel='chrome'` drives the real
    Google Chrome the user already has installed, which is the closest thing
    to "the browser I opened the link in myself" and needs no extra download.
    Playwright's bundled Chromium is the fallback, and
    NFLSCHOLAR_BROWSER_PATH overrides both for a machine where the bundled
    build and the installed Playwright disagree about versions.

    One non-default argument: --no-sandbox, needed to run as root inside a
    container and irrelevant on a normal desktop.

    NOTHING HERE HIDES THAT THE BROWSER IS AUTOMATED. An earlier version of
    this passed --disable-blink-features=AutomationControlled, which turns
    off the navigator.webdriver flag - and the docstring at the top of this
    file lists "no navigator.webdriver patching" as a thing this module does
    not do. Shipping both would have made that claim false, and a stated
    principle that quietly does not hold is worse than not stating it. The
    flag is gone; the claim is true.

    If a book refuses an automated browser, that is a real answer. Run it
    headed (NFLSCHOLAR_BROWSER_HEADED=1) or save the JSON by hand.
    """
    headed = os.environ.get(HEADED_ENV, '').strip() in ('1', 'true', 'yes')
    args = ['--no-sandbox']
    attempts = []
    override = os.environ.get(BROWSER_PATH_ENV, '').strip()
    if override:
        attempts.append({'executable_path': override})
    attempts.append({'channel': 'chrome'})
    attempts.append({})

    last = None
    for kwargs in attempts:
        try:
            return playwright.chromium.launch(headless=not headed, args=args, **kwargs)
        except Exception as exc:
            last = exc
    raise last


def _read_json(page, url):
    """Navigate and pull the JSON body out, whatever wrapper it arrives in."""
    response = page.goto(url, timeout=BROWSER_TIMEOUT_MS, wait_until='domcontentloaded')
    status = response.status if response else 0
    if status and status >= 400:
        return None, f"{url} returned {status} to a browser too."
    # A JSON endpoint viewed in Chromium renders inside a <pre>; some builds
    # add a JSON viewer around it. innerText of body covers both, and
    # falling back to content() covers neither being present.
    try:
        text = page.evaluate("() => document.body ? document.body.innerText : ''")
    except Exception:
        text = ''
    if not text.strip():
        text = page.content()
    text = text.strip()
    if not text:
        return None, f"{url} came back empty in the browser."
    try:
        return json.loads(text), None
    except ValueError:
        head = text[:160].replace('\n', ' ')
        return None, (f"{url} loaded but wasn't JSON - most likely a challenge or "
                      f"consent page: {head!r}")


@contextlib.contextmanager
def browser_session():
    """
    One browser for several URLs.

    Worth having because startup is a second or two and DraftKings needs
    eight calls; paying that once instead of eight times is the difference
    between a usable refresh and one you avoid pressing.

    Yields a `get(url) -> (payload, error)` function, or None when Playwright
    isn't installed.
    """
    ok, _why = browser_available()
    if not ok:
        yield None
        return

    from playwright.sync_api import sync_playwright
    playwright = browser = None
    try:
        playwright = sync_playwright().start()
        browser = _launch(playwright)
        context = browser.new_context()
        page = context.new_page()

        def get(url):
            try:
                return _read_json(page, url)
            except Exception as exc:
                return None, f"Browser couldn't load {url}: {exc}"

        yield get
    except Exception as exc:
        def failed(url, _exc=exc):
            return None, f"Couldn't start a browser: {_exc}"
        yield failed
    finally:
        with contextlib.suppress(Exception):
            if browser:
                browser.close()
        with contextlib.suppress(Exception):
            if playwright:
                playwright.stop()


def fetch_json_via_browser(url):
    """One URL through a browser. Returns (payload, error)."""
    ok, why = browser_available()
    if not ok:
        return None, why
    with browser_session() as get:
        if get is None:
            return None, why
        return get(url)
