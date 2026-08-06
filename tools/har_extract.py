#!/usr/bin/env python3
"""
Shrink a browser HAR capture down to just the API responses, with
credentials stripped.

WHY: a HAR saved from DevTools is almost entirely payload you don't want -
images, fonts, JS bundles, CSS, tracking pixels. The JSON the page actually
fetched (rankings, projections, ADP, draft state) is typically 1-3% of the
file. Filtering to that usually takes a 40MB capture under 1MB, which is
small enough to commit and read.

AND: a raw HAR contains every cookie and Authorization header the browser
sent, which means it contains your live session for that site. Committing
one to a git repo publishes your login. This script never copies headers,
cookies, or POST bodies, and redacts credential-looking query parameters
from the URLs it keeps - so what comes out is the data, not your account.

Usage
    python tools/har_extract.py capture.har
    python tools/har_extract.py capture.har -o reference/ffa_api.json
    python tools/har_extract.py capture.har --list        # just show what's in it
    python tools/har_extract.py capture.har --max-mb 8    # raise the size cap

If the result is still too big, use --list to see which URLs carry the bulk,
then --only to keep just those:
    python tools/har_extract.py capture.har --only rankings,players,adp
"""
import argparse
import base64
import json
import os
import re
import sys

# Response types worth keeping. Anything else is page furniture.
_WANTED_MIME = re.compile(r'json|javascript/json|text/plain', re.I)
_WANTED_URL = re.compile(r'/api/|/graphql|/v\d+/|\.json(\?|$)', re.I)

# Never keep these, even when they claim to be JSON - analytics and error
# reporting are pure noise and often the largest JSON in a capture.
_JUNK_URL = re.compile(
    r'google-analytics|googletagmanager|doubleclick|segment\.(io|com)|'
    r'sentry|bugsnag|datadoghq|hotjar|mixpanel|amplitude|intercom|'
    r'facebook\.com|fullstory|clarity\.ms|launchdarkly|cloudflareinsights',
    re.I)

# Query parameters that carry credentials. Their values are replaced rather
# than the whole URL dropped, because the path itself is often the most
# useful thing in the capture.
_SECRET_PARAM = re.compile(
    r'^(token|access_token|id_token|refresh_token|auth|authorization|key|'
    r'api_?key|apikey|session|sessionid|sid|jwt|password|passwd|secret|'
    r'signature|sig|code|client_secret)$', re.I)


def _redact_url(url):
    """Blank out credential-looking query parameter values."""
    if '?' not in url:
        return url
    base, _, query = url.partition('?')
    parts = []
    for chunk in query.split('&'):
        name, sep, _value = chunk.partition('=')
        parts.append(f"{name}={sep and 'REDACTED'}" if _SECRET_PARAM.match(name) else chunk)
    return f"{base}?{'&'.join(parts)}"


def _body_text(response):
    """Decode a HAR entry's response body, handling base64 encoding."""
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


def _looks_useful(url, mime):
    if _JUNK_URL.search(url):
        return False
    return bool(_WANTED_MIME.search(mime or '')) or bool(_WANTED_URL.search(url))


def extract(har_path, only=None):
    """Return [{'url','method','status','bytes','body'}] for the API responses."""
    with open(har_path, 'r', encoding='utf-8', errors='replace') as fh:
        har = json.load(fh)

    entries = (har.get('log') or {}).get('entries') or []
    kept, seen = [], set()
    for entry in entries:
        request = entry.get('request') or {}
        response = entry.get('response') or {}
        url = request.get('url') or ''
        mime = ((response.get('content') or {}).get('mimeType')) or ''
        if not _looks_useful(url, mime):
            continue
        if only and not any(token.lower() in url.lower() for token in only):
            continue
        body = _body_text(response)
        if not body or not body.strip():
            continue
        # Identical repeated polls are common (a draft board refreshing);
        # one copy of each distinct body is enough.
        fingerprint = (url.split('?')[0], len(body), body[:200])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append({
            'url': _redact_url(url),
            'method': request.get('method', 'GET'),
            'status': response.get('status'),
            'mime': mime,
            'bytes': len(body),
            'body': body,
        })
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('har', help='Path to the .har file saved from DevTools')
    ap.add_argument('-o', '--out', default=None,
                    help='Output path (default: alongside the input, as <name>_api.json)')
    ap.add_argument('--list', action='store_true',
                    help='Only print what was found, write nothing')
    ap.add_argument('--only', default=None,
                    help='Comma-separated substrings; keep only URLs containing one of them')
    ap.add_argument('--max-mb', type=float, default=5.0,
                    help='Warn above this output size in MB (default 5)')
    args = ap.parse_args()

    if not os.path.exists(args.har):
        sys.exit(f"No such file: {args.har}")

    only = [t.strip() for t in args.only.split(',') if t.strip()] if args.only else None
    original_mb = os.path.getsize(args.har) / 1e6
    kept = extract(args.har, only=only)

    if not kept:
        sys.exit("No API-looking responses found. Re-save the HAR with the "
                 "'Fetch/XHR' filter cleared, and make sure you used "
                 "'Save all as HAR with content'.")

    kept.sort(key=lambda e: -e['bytes'])
    print(f"Input:  {args.har}  ({original_mb:.1f} MB)")
    print(f"Kept:   {len(kept)} API responses\n")
    print(f"{'BYTES':>10}  {'STATUS':>6}  URL")
    for entry in kept[:40]:
        url = entry['url']
        print(f"{entry['bytes']:>10,}  {str(entry['status']):>6}  {url[:110]}")
    if len(kept) > 40:
        print(f"{'':>10}  {'':>6}  ... and {len(kept) - 40} more")

    if args.list:
        return

    out_path = args.out or os.path.splitext(args.har)[0] + '_api.json'
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    payload = {'source_har': os.path.basename(args.har), 'responses': kept}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=1)

    out_mb = os.path.getsize(out_path) / 1e6
    print(f"\nWrote {out_path}  ({out_mb:.2f} MB, {out_mb / original_mb * 100:.1f}% of original)")
    print("Headers, cookies and POST bodies were not copied; credential-like "
          "query parameters were redacted.")
    if out_mb > args.max_mb:
        print(f"\nStill above {args.max_mb} MB. Narrow it with --only, e.g.:")
        hosts = sorted({e['url'].split('/')[2] for e in kept if '//' in e['url']})
        print(f"  python {sys.argv[0]} {args.har} --only <substring>")
        print(f"  hosts seen: {', '.join(hosts[:8])}")


if __name__ == '__main__':
    main()
