#!/usr/bin/env python3
"""
Unpack a browser HAR capture into a readable, greppable directory tree, with
credentials stripped.

WHY NOT JUST COMMIT THE HAR: a capture of a rich web app runs to hundreds of
megabytes, and essentially all of that is player headshots, fonts, and media.
GitHub refuses any single file over 100MB, and no reader - human or model -
can work through a 320MB single-line JSON blob. Splitting it into one file
per response, in folders by kind, turns it into something you can actually
open, search and diff.

WHAT IT KEEPS: everything textual, because on a reference site the
non-JSON assets are often the interesting part.
  api/   JSON responses      - the data: rankings, projections, ADP, state
  js/    scripts             - the CLIENT-SIDE LOGIC. A draft tool computes
                               tiers, value and availability in the browser,
                               so the bundle frequently spells out formulas
                               and field names the API alone doesn't reveal.
  css/   stylesheets         - the design system: palette, type scale, spacing
  html/  documents           - server-rendered markup and embedded state
  other/ anything else textual it didn't recognise

WHAT IT SKIPS: images, fonts, audio, video, and analytics/error-reporting
endpoints. These are the bulk of the bytes and none of them are readable as
text. Nothing is hidden, though - EVERY request in the capture, kept or
skipped, is listed in index.json with its URL, type and size, so if a skipped
asset turns out to matter you can go pull that one deliberately.

CREDENTIALS: a raw HAR carries every cookie and Authorization header the
browser sent, so committing one publishes a live login session for the
captured site. This copies no headers and no cookies, and redacts
credential-looking query parameters from every URL it records.

POST request bodies ARE recorded, field-redacted, because for a POST API the
body is the question - a draft assistant's response means nothing without the
roster state and pick number that were sent to get it. Credential-looking
keys are blanked at any nesting depth, and auth/login endpoints are skipped
whole rather than field-redacted, since nothing in a login body is worth
keeping and one near-miss on the pattern would leak a password.

Usage
    python tools/har_extract.py capture.har -o reference/ffa
    python tools/har_extract.py capture.har --list          # inventory only
    python tools/har_extract.py capture.har -o out --include api,js
    python tools/har_extract.py capture.har -o out --max-file-mb 40
"""
import argparse
import base64
import hashlib
import json
import os
import re
import sys
from collections import Counter
from urllib.parse import urlsplit

# Binary asset types. Not readable as text and they are essentially all of
# the bytes in a large capture.
_BINARY_MIME = re.compile(r'^(image|font|audio|video)/|application/(octet-stream|zip|pdf|wasm)'
                          r'|application/vnd\.ms-fontobject', re.I)
_BINARY_EXT = re.compile(r'\.(png|jpe?g|gif|webp|avif|svg|ico|bmp|woff2?|ttf|otf|eot|'
                         r'mp4|webm|mp3|wav|zip|gz|pdf|wasm)(\?|$)', re.I)

# Telemetry. Often the largest JSON in a capture and never informative.
_JUNK_URL = re.compile(
    r'google-analytics|googletagmanager|doubleclick|googlesyndication|'
    r'segment\.(io|com)|sentry|bugsnag|datadoghq|hotjar|mixpanel|amplitude|'
    r'intercom|facebook\.(com|net)|fullstory|clarity\.ms|launchdarkly|'
    r'cloudflareinsights|newrelic|optimizely|adservice|scorecardresearch',
    re.I)

_SECRET_PARAM = re.compile(
    r'^(token|access_token|id_token|refresh_token|auth|authorization|key|'
    r'api_?key|apikey|session|sessionid|sid|jwt|password|passwd|secret|'
    r'signature|sig|code|client_secret|email|user_?id)$', re.I)

CATEGORIES = ('api', 'js', 'css', 'html', 'other')


def _redact_url(url):
    """Blank credential-looking query parameter values, keep the path intact."""
    if '?' not in url:
        return url
    base, _, query = url.partition('?')
    parts = []
    for chunk in query.split('&'):
        name, sep, _value = chunk.partition('=')
        parts.append(f"{name}={sep and 'REDACTED'}" if _SECRET_PARAM.match(name) else chunk)
    return f"{base}?{'&'.join(parts)}"


def _classify(url, mime):
    """Which folder this response belongs in, or None to skip it."""
    if _JUNK_URL.search(url):
        return None
    if _BINARY_MIME.search(mime or '') or _BINARY_EXT.search(url):
        return None
    mime = (mime or '').lower()
    if 'json' in mime or re.search(r'/api/|/graphql|/v\d+/|\.json(\?|$)', url, re.I):
        return 'api'
    if 'javascript' in mime or 'ecmascript' in mime or re.search(r'\.m?jsx?(\?|$)', url, re.I):
        return 'js'
    if 'css' in mime or re.search(r'\.css(\?|$)', url, re.I):
        return 'css'
    if 'html' in mime or 'xml' in mime:
        return 'html'
    if mime.startswith('text/'):
        return 'other'
    return None


def _body_text(response):
    content = response.get('content') or {}
    text = content.get('text')
    if not text:
        return None
    if content.get('encoding') == 'base64':
        try:
            text = base64.b64decode(text).decode('utf-8', errors='replace')
        except Exception:
            return None
    return text


def _filename_for(url, category, used, url_seen):
    """
    A readable, unique, filesystem-safe name derived from the URL path.

    TWO different collisions have to be handled, and conflating them loses
    data. Hashing the URL alone (the first version of this) disambiguated
    /api/sos?table=overall from /api/sos?table=rushing correctly - but every
    REPEAT call to one endpoint hashes identically, so a draft assistant
    polled fifteen times through a draft wrote fifteen responses over the
    top of each other and only the last survived. Those repeats are the
    single most interesting thing in a draft capture: they're the same
    endpoint answering as your roster fills up.

    So: different URL sharing a basename gets a URL hash, the SAME URL hit
    again gets a sequential number in capture order. Ordinals rather than
    content hashes because the order is the meaning - _001 through _015 is
    the draft unfolding. Both are deterministic, so re-running the extractor
    on one capture always produces the same filenames.
    """
    parts = urlsplit(url)
    stem = os.path.basename(parts.path) or (parts.path.strip('/').replace('/', '_')) or 'index'
    stem = re.sub(r'[^A-Za-z0-9._-]', '_', stem)[:70] or 'response'
    ext = {'api': '.json', 'js': '.js', 'css': '.css', 'html': '.html', 'other': '.txt'}[category]
    if not stem.lower().endswith(ext):
        stem += ext
    root = stem[:-len(ext)]

    owner = url_seen.get(stem)
    if owner is not None and owner != url:
        # A different endpoint already claimed this basename.
        root = f"{root}_{hashlib.sha1(url.encode()).hexdigest()[:6]}"
        stem = root + ext
        url_seen.setdefault(stem, url)
    else:
        url_seen.setdefault(stem, url)

    candidate = stem
    ordinal = 1
    while candidate in used:
        ordinal += 1
        candidate = f"{root}_{ordinal:03d}{ext}"
    used.add(candidate)
    return candidate


# Keys whose values are credentials, redacted wherever they appear in a
# captured request body (at any nesting depth).
_SECRET_KEY = re.compile(
    r'(pass|passwd|password|token|secret|auth|apikey|api_key|jwt|session|'
    r'cookie|signature|otp|pin|ssn|card|cvv)', re.I)

# Endpoints whose request bodies are credentials by definition. Skipped
# whole rather than field-redacted.
_AUTH_URL = re.compile(r'/(login|signin|sign-in|auth|oauth|token|register|'
                       r'signup|sign-up|password|session)s?(/|\?|$)', re.I)


def _redact_structure(value):
    """Recursively blank credential-looking keys in a decoded JSON body."""
    if isinstance(value, dict):
        return {k: ('REDACTED' if _SECRET_KEY.search(str(k)) else _redact_structure(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_structure(v) for v in value]
    return value


def _request_body(request, max_chars=20000):
    """
    The POST body that produced a response, with credential fields blanked.

    Worth capturing despite the extra care needed: for a POST API the body IS
    the question. A draft assistant's response is meaningless without knowing
    the roster state and pick number that were sent to get it - you'd see the
    answer with no idea what was asked. Auth endpoints are skipped entirely
    rather than field-redacted, since there is nothing in a login body worth
    keeping and a near-miss on the redaction pattern would leak a password.
    """
    url = request.get('url') or ''
    if _AUTH_URL.search(url):
        return {'skipped': 'auth endpoint'}
    post = request.get('postData') or {}
    text = post.get('text')
    if not text:
        return None
    try:
        return _redact_structure(json.loads(text))
    except Exception:
        # Not JSON (form-encoded, or truncated by the browser) - keep a
        # bounded snippet only if it carries no credential-looking keys.
        if _SECRET_KEY.search(text[:2000]):
            return {'skipped': 'non-JSON body containing credential-like keys'}
        return {'raw': text[:max_chars]}


def _pretty(text, category):
    """Pretty-print JSON so it's diffable and readable; leave everything else."""
    if category != 'api':
        return text
    try:
        return json.dumps(json.loads(text), indent=1, ensure_ascii=False)
    except Exception:
        return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('har', help='Path to the .har saved from DevTools')
    ap.add_argument('-o', '--out', default=None, help='Output DIRECTORY (default: <name>_unpacked)')
    ap.add_argument('--list', action='store_true', help='Inventory only, write nothing')
    ap.add_argument('--include', default=','.join(CATEGORIES),
                    help=f"Comma-separated categories to write ({', '.join(CATEGORIES)})")
    ap.add_argument('--only', default=None, help='Keep only URLs containing one of these substrings')
    ap.add_argument('--max-file-mb', type=float, default=60.0,
                    help='Skip any single response larger than this (GitHub rejects >100MB)')
    args = ap.parse_args()

    if not os.path.exists(args.har):
        sys.exit(f"No such file: {args.har}")
    include = {c.strip() for c in args.include.split(',') if c.strip()}
    only = [t.strip().lower() for t in args.only.split(',') if t.strip()] if args.only else None

    original_mb = os.path.getsize(args.har) / 1e6
    print(f"Reading {args.har} ({original_mb:,.0f} MB) — this can take a minute on a large capture.")
    with open(args.har, 'r', encoding='utf-8', errors='replace') as fh:
        har = json.load(fh)
    entries = (har.get('log') or {}).get('entries') or []
    print(f"{len(entries):,} requests in capture.\n")

    out_dir = args.out or (os.path.splitext(args.har)[0] + '_unpacked')
    manifest, used_names, url_seen, kept_bytes = [], set(), {}, Counter()
    counts = Counter()
    written = 0

    for entry in entries:
        request = entry.get('request') or {}
        response = entry.get('response') or {}
        url = request.get('url') or ''
        mime = ((response.get('content') or {}).get('mimeType')) or ''
        size = (response.get('content') or {}).get('size') or 0
        category = _classify(url, mime)
        record = {
            'url': _redact_url(url),
            'method': request.get('method', 'GET'),
            'status': response.get('status'),
            'mime': mime,
            'bytes': size,
            'category': category or 'skipped',
            'file': None,
        }

        if category is None:
            record['reason'] = 'binary asset or analytics endpoint'
            counts['skipped'] += 1
            manifest.append(record)
            continue
        if only and not any(token in url.lower() for token in only):
            record['reason'] = 'filtered out by --only'
            counts['filtered'] += 1
            manifest.append(record)
            continue
        if category not in include:
            record['reason'] = f'category {category} not in --include'
            counts['excluded'] += 1
            manifest.append(record)
            continue

        body = _body_text(response)
        if not body or not body.strip():
            record['reason'] = 'no response body captured (re-save with "Save all as HAR with content")'
            counts['empty'] += 1
            manifest.append(record)
            continue
        if len(body) / 1e6 > args.max_file_mb:
            record['reason'] = f'response larger than --max-file-mb ({len(body)/1e6:.0f} MB)'
            counts['oversize'] += 1
            manifest.append(record)
            continue

        counts[category] += 1
        kept_bytes[category] += len(body)
        if not args.list:
            folder = os.path.join(out_dir, category)
            os.makedirs(folder, exist_ok=True)
            name = _filename_for(url, category, used_names, url_seen)
            with open(os.path.join(folder, name), 'w', encoding='utf-8') as fh:
                fh.write(_pretty(body, category))
            record['file'] = f"{category}/{name}"
            written += 1
            body_sent = _request_body(request)
            if body_sent is not None:
                record['request_body'] = body_sent
        manifest.append(record)

    print(f"{'CATEGORY':<10} {'FILES':>7} {'SIZE':>12}")
    for category in CATEGORIES:
        if counts[category]:
            print(f"{category:<10} {counts[category]:>7,} {kept_bytes[category]/1e6:>10,.1f} MB")
    for label in ('skipped', 'filtered', 'excluded', 'empty', 'oversize'):
        if counts[label]:
            print(f"{label:<10} {counts[label]:>7,}")

    total_mb = sum(kept_bytes.values()) / 1e6
    print(f"\nKept {total_mb:,.1f} MB of text out of {original_mb:,.0f} MB "
          f"({total_mb / max(original_mb, 1e-9) * 100:.1f}%).")

    if args.list:
        print("\n--list: nothing written. Largest responses that WOULD be kept:")
        for record in sorted([m for m in manifest if m['category'] in CATEGORIES],
                             key=lambda m: -(m['bytes'] or 0))[:25]:
            print(f"  {(m := record)['bytes'] or 0:>12,}  {m['category']:<5} {m['url'][:100]}")
        return

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'index.json'), 'w', encoding='utf-8') as fh:
        json.dump({'source_har': os.path.basename(args.har),
                   'source_mb': round(original_mb, 1),
                   'requests_total': len(entries),
                   'entries': manifest}, fh, indent=1)

    print(f"Wrote {written:,} files to {out_dir}/")
    print(f"index.json lists all {len(entries):,} requests including skipped ones, "
          "so nothing is lost track of.")
    print("No headers or cookies copied. Credential-like query parameters are "
          "redacted, POST bodies are field-redacted, auth endpoints skipped whole.")


if __name__ == '__main__':
    main()
