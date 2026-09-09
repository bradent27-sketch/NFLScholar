"""Flatten hand-saved Ourlads ``.mhtml`` archive pages to plain ``.html``.

Browsers' "Save page as -> Web Page, Single File" writes a MIME multipart
(``.mhtml``) container: the HTML document plus every CSS/image resource, the
HTML part quoted-printable-encoded. scripts/import_ourlads_historical.py only
globs ``*.htm*`` and regex-scans raw text, so it silently skips ``.mhtml`` and
would choke on the ``=3D`` / soft-break encoding anyway.

This extracts the single ``text/html`` part from each ``.mhtml``, decodes it,
and writes a sibling ``<same name>.html`` the importer then picks up. Idempotent
(skips a target that already exists unless --force). Leaves the ``.mhtml``
originals in place.

    python scripts/convert_ourlads_mhtml.py \
        --src "external_data/OurLads Historical Depth Charts - Week Before Season 0901"
    python scripts/import_ourlads_historical.py
"""
from __future__ import annotations

import argparse
import email
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_SRC = os.path.join(
    ROOT, "external_data", "OurLads Historical Depth Charts - Week Before Season 0901"
)


def convert_one(path: str, force: bool = False) -> dict:
    out = {"src": os.path.basename(path), "dst": None, "status": "", "bytes": 0}
    dst = re.sub(r"\.mhtml$", ".html", path, flags=re.I)
    if dst == path:
        out["status"] = "skip (not .mhtml)"
        return out
    out["dst"] = os.path.basename(dst)
    if os.path.exists(dst) and not force:
        out["status"] = "exists (skip)"
        return out
    with open(path, "rb") as fh:
        msg = email.message_from_binary_file(fh)
    html_parts = [p for p in msg.walk() if p.get_content_type() == "text/html"]
    if not html_parts:
        out["status"] = "NO text/html PART"
        return out
    # The saved document is the first (usually only) text/html part; later ones
    # would be inlined iframes, which Ourlads archive pages do not use.
    payload = html_parts[0].get_payload(decode=True)
    if payload is None:
        out["status"] = "empty payload"
        return out
    text = payload.decode("utf-8", errors="replace")
    if "ctl00_phContent_dcTBody" not in text:
        out["status"] = "WARN: no depth-chart tbody in decoded html"
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    out["bytes"] = len(text)
    out["status"] = out["status"] or "ok"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--force", action="store_true", help="overwrite existing .html")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "**", "*.mhtml"), recursive=True))
    print(f"found {len(files)} .mhtml under {args.src}\n")
    if not files:
        return 2

    n_ok = n_skip = n_warn = 0
    for path in files:
        r = convert_one(path, force=args.force)
        flag = "  ok " if r["status"] in ("ok", "exists (skip)") else "WARN "
        if r["status"].startswith("WARN") or "NO " in r["status"] or "empty" in r["status"]:
            n_warn += 1
        elif r["status"] == "exists (skip)":
            n_skip += 1
        else:
            n_ok += 1
        print(f"{flag}{r['src']}  ->  {r['dst']}   [{r['status']}, {r['bytes']} chars]")

    print(f"\nconverted {n_ok}, skipped {n_skip} existing, {n_warn} warnings")
    return 1 if n_warn else 0


if __name__ == "__main__":
    sys.exit(main())
