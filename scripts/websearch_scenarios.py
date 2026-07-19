#!/usr/bin/env python3
"""Web-search auto-detection harness (Tahap 2, Bagian B).

Goal: measure how accurately the router AUTO-detects when a chat turn needs a
web search — WITHOUT the manual toggle. The manual web button is left OFF for
every scenario so what we measure is purely the automatic path (intent
classification + contextual follow-up). The result decides whether the manual
toggle is still needed, can become an optional mode, or be removed.

How it works:
  1. Log in as admin, create a throwaway session per scenario group.
  2. Send each scenario prompt to /api/chat_stream with mode=chat and the
     manual web toggle OFF (no use_web / allow_web_search).
  3. Read the router_decisions row back via the admin observability endpoint
     and compare `search_enabled` (did web actually get enabled?) against the
     scenario's expected label `should_search`.
  4. Print a precision/recall report + the misclassified cases + a
     recommendation for the manual toggle.

Usage:
  python scripts/websearch_scenarios.py \
      --base-url http://127.0.0.1:7000 \
      --user admin --password '<admin-password>'

Exit code is always 0 (this is a measurement tool, not a gate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Labeled scenario dataset (ID + EN). `should_search=True` means a correct
# router SHOULD have enabled web search automatically for this turn.
#
# `followup_of` chains a turn onto the previous one in the SAME session to test
# contextual web follow-ups (the second turn refers back to the first).
# ---------------------------------------------------------------------------
SCENARIOS = [
    # ── Group 1: fresh / factual → SHOULD search ──────────────────────────
    {"group": "fresh_factual", "prompt": "berita terbaru gempa hari ini", "should_search": True},
    {"group": "fresh_factual", "prompt": "harga bitcoin sekarang berapa", "should_search": True},
    {"group": "fresh_factual", "prompt": "cuaca Jakarta besok gimana", "should_search": True},
    {"group": "fresh_factual", "prompt": "siapa yang menang pertandingan tadi malam", "should_search": True},
    {"group": "fresh_factual", "prompt": "kurs dolar ke rupiah hari ini", "should_search": True},
    {"group": "fresh_factual", "prompt": "what is the latest news about the election", "should_search": True},
    {"group": "fresh_factual", "prompt": "current stock price of Apple", "should_search": True},
    {"group": "fresh_factual", "prompt": "who is the president of Argentina right now", "should_search": True},

    # ── Group 2: stable knowledge → should NOT search ─────────────────────
    {"group": "stable_knowledge", "prompt": "apa itu fotosintesis", "should_search": False},
    {"group": "stable_knowledge", "prompt": "jelaskan cara kerja rekursi dalam pemrograman", "should_search": False},
    {"group": "stable_knowledge", "prompt": "tuliskan puisi singkat tentang laut", "should_search": False},
    {"group": "stable_knowledge", "prompt": "bagaimana cara membuat kopi yang enak", "should_search": False},
    {"group": "stable_knowledge", "prompt": "what is the capital of France", "should_search": False},
    {"group": "stable_knowledge", "prompt": "explain the difference between TCP and UDP", "should_search": False},
    {"group": "stable_knowledge", "prompt": "write a haiku about autumn", "should_search": False},
    {"group": "stable_knowledge", "prompt": "halo apa kabar", "should_search": False},

    # ── Group 2b: English "who/what is X right now" → SHOULD search ────────
    {"group": "en_fresh_who_is", "prompt": "who is the president of Argentina right now", "should_search": True},
    {"group": "en_fresh_who_is", "prompt": "who is the prime minister of the UK currently", "should_search": True},
    {"group": "en_fresh_who_is", "prompt": "what is the price of gold today", "should_search": True},
    {"group": "en_fresh_who_is", "prompt": "how much is one euro in dollars right now", "should_search": True},
    # Stable "who/what is" WITHOUT a freshness marker → should NOT search.
    {"group": "en_stable_who_is", "prompt": "who wrote Hamlet", "should_search": False},
    {"group": "en_stable_who_is", "prompt": "what is the capital of Japan", "should_search": False},
    {"group": "en_stable_who_is", "prompt": "who is Albert Einstein", "should_search": False},

    # ── Group 2c: more Indonesian fresh / stable coverage ─────────────────
    {"group": "id_fresh_extra", "prompt": "jadwal pertandingan bola malam ini", "should_search": True},
    {"group": "id_fresh_extra", "prompt": "berapa harga emas antam hari ini", "should_search": True},
    {"group": "id_fresh_extra", "prompt": "cari di internet tentang lowongan kerja", "should_search": True},
    {"group": "id_stable_extra", "prompt": "apa perbedaan RAM dan ROM", "should_search": False},
    {"group": "id_stable_extra", "prompt": "jelaskan apa itu inflasi", "should_search": False},
    {"group": "id_stable_extra", "prompt": "cara memasak rendang yang empuk", "should_search": False},

    # ── Group 3: contextual follow-up → second turn SHOULD search ─────────
    # These require carrying prior-turn topic/source across turns (anaphora).
    # Keyword-only detection cannot resolve them; they are measured SEPARATELY
    # and are the target of the Fase C contextual-state work, not Fase B.
    {"group": "contextual_followup", "prompt": "berita terbaru soal harga emas", "should_search": True, "chain_start": True},
    {"group": "contextual_followup", "prompt": "kalau yang di Amerika bagaimana?", "should_search": True, "followup": True, "contextual": True},

    {"group": "contextual_followup_en", "prompt": "what is the latest news on AI regulation", "should_search": True, "chain_start": True},
    {"group": "contextual_followup_en", "prompt": "and what about in Europe?", "should_search": True, "followup": True, "contextual": True},

    # ── Group 4: ambiguous / boundary ─────────────────────────────────────
    {"group": "boundary", "prompt": "kabar terbaru tentang AI", "should_search": True},
    {"group": "boundary", "prompt": "resep nasi goreng sederhana", "should_search": False},
    {"group": "boundary", "prompt": "info terkini tentang startup teknologi", "should_search": True},
    {"group": "boundary", "prompt": "ceritakan sejarah Kerajaan Majapahit", "should_search": False},
    {"group": "boundary", "prompt": "trending topic di media sosial hari ini", "should_search": True},
    {"group": "boundary", "prompt": "tips belajar bahasa Inggris", "should_search": False},
]


class Client:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.cookie = None

    def _req(self, method: str, path: str, *, json_body=None, form=None, timeout=120):
        url = self.base + path
        headers = {}
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        elif form is not None:
            boundary = "----harness%d" % int(time.time() * 1000)
            parts = []
            for k, v in form.items():
                parts.append(f"--{boundary}")
                parts.append(f'Content-Disposition: form-data; name="{k}"')
                parts.append("")
                parts.append(str(v))
            parts.append(f"--{boundary}--")
            parts.append("")
            data = "\r\n".join(parts).encode()
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        if self.cookie:
            headers["Cookie"] = self.cookie
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            sc = resp.headers.get_all("Set-Cookie") or []
            if sc:
                self.cookie = "; ".join(c.split(";")[0] for c in sc)
            return resp.getcode(), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def login(self, user: str, password: str):
        code, body = self._req("POST", "/api/auth/login",
                               json_body={"username": user, "password": password})
        if code != 200:
            raise SystemExit(f"Login failed ({code}): {body[:200]!r}")

    def create_session(self, name: str, model: str, endpoint_id: str) -> str:
        code, body = self._req("POST", "/api/session", form={
            "name": name, "model": model,
            "endpoint_id": endpoint_id, "skip_validation": "true",
        })
        if code != 200:
            raise SystemExit(f"Session create failed ({code}): {body[:200]!r}")
        return json.loads(body)["id"]

    def send(self, session: str, message: str):
        # Manual web toggle intentionally OFF (no use_web / allow_web_search).
        code, _ = self._req("POST", "/api/chat_stream", form={
            "message": message, "session": session, "mode": "chat",
        }, timeout=180)
        return code

    def last_decision(self, session: str):
        code, body = self._req("GET",
                               f"/api/admin/router-decisions?limit=1&session_id={session}")
        if code != 200:
            return None
        rows = json.loads(body).get("decisions", [])
        return rows[0] if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:7000")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--model", default="hy3-free")
    ap.add_argument("--endpoint-id", default="56f0cd90")
    ap.add_argument("--json-out", default=None, help="Optional path to write raw results JSON")
    args = ap.parse_args()

    c = Client(args.base_url)
    c.login(args.user, args.password)

    results = []
    # Group scenarios so contextual follow-ups share a session with their start.
    chain_session = {}
    for i, sc in enumerate(SCENARIOS):
        group = sc["group"]
        if sc.get("followup") and group in chain_session:
            session = chain_session[group]
        else:
            session = c.create_session(f"[harness] {group} {i}", args.model, args.endpoint_id)
            if sc.get("chain_start"):
                chain_session[group] = session

        code = c.send(session, sc["prompt"])
        time.sleep(0.5)
        dec = c.last_decision(session) or {}
        got = bool(dec.get("search_enabled"))
        exp = bool(sc["should_search"])
        results.append({
            "group": group,
            "prompt": sc["prompt"],
            "expected": exp,
            "got_search_enabled": got,
            "contextual": bool(sc.get("contextual")),
            "search_trigger": dec.get("search_trigger"),
            "intent_category": dec.get("intent_category"),
            "effective_mode": dec.get("effective_mode"),
            "exit_path": dec.get("exit_path"),
            "correct": got == exp,
            "http": code,
        })
        mark = "OK " if got == exp else "XX "
        print(f"{mark}[{group:22}] exp={int(exp)} got={int(got)} trig={dec.get('search_trigger')!s:18} :: {sc['prompt'][:55]}")

    # ── Metrics ───────────────────────────────────────────────────────────
    # Decision thresholds for the manual-toggle call (agreed in planning).
    THRESH_RECALL = 0.90
    THRESH_PRECISION = 0.95

    def _metrics(subset):
        tp = sum(1 for r in subset if r["expected"] and r["got_search_enabled"])
        fp = sum(1 for r in subset if not r["expected"] and r["got_search_enabled"])
        fn = sum(1 for r in subset if r["expected"] and not r["got_search_enabled"])
        tn = sum(1 for r in subset if not r["expected"] and not r["got_search_enabled"])
        n = len(subset)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        accuracy = (tp + tn) / n if n else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": n,
                "precision": precision, "recall": recall,
                "accuracy": accuracy, "f1": f1}

    # Contextual follow-ups need cross-turn state (Fase C); they are NOT part
    # of the keyword-targetable set the toggle decision is based on.
    non_ctx = [r for r in results if not r.get("contextual")]
    ctx = [r for r in results if r.get("contextual")]
    overall = _metrics(results)
    keyword = _metrics(non_ctx)
    contextual = _metrics(ctx) if ctx else None

    def _print_block(title, m):
        print(f"\n{title}")
        print(f"  Total    : {m['total']}")
        print(f"  Confusion: TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
        print(f"  Precision: {m['precision']:.2f}")
        print(f"  Recall   : {m['recall']:.2f}")
        print(f"  Accuracy : {m['accuracy']:.2f}")
        print(f"  F1       : {m['f1']:.2f}")

    print("\n" + "=" * 70)
    print("WEB-SEARCH AUTO-DETECTION REPORT (manual toggle OFF)")
    print("=" * 70)
    _print_block("OVERALL (all scenarios)", overall)
    _print_block("KEYWORD-TARGETABLE (excludes contextual follow-ups)", keyword)
    if contextual:
        _print_block("CONTEXTUAL FOLLOW-UPS (needs Fase C cross-turn state)", contextual)

    fns = [r for r in results if r["expected"] and not r["got_search_enabled"]]
    fps = [r for r in results if not r["expected"] and r["got_search_enabled"]]
    if fns:
        print(f"\nFALSE NEGATIVES ({len(fns)}) — should have searched but didn't:")
        for r in fns:
            tag = " (contextual)" if r.get("contextual") else ""
            print(f"  - [{r['group']}] {r['prompt']}{tag}")
    if fps:
        print(f"\nFALSE POSITIVES ({len(fps)}) — searched but shouldn't have:")
        for r in fps:
            print(f"  - [{r['group']}] {r['prompt']}")

    # ── Recommendation (threshold-based, on keyword-targetable subset) ─────
    meets = (keyword["recall"] >= THRESH_RECALL
             and keyword["precision"] >= THRESH_PRECISION)
    print("\n" + "-" * 70)
    print("RECOMMENDATION (manual web toggle):")
    print(f"  Thresholds: recall>={THRESH_RECALL:.2f} AND precision>={THRESH_PRECISION:.2f}")
    print(f"  Keyword-targetable: recall={keyword['recall']:.2f} "
          f"precision={keyword['precision']:.2f} → "
          f"{'MEETS' if meets else 'BELOW'} thresholds")
    if meets:
        print("  → Auto-detect is reliable for explicit web-worthy phrasing.")
        print("    Make the manual toggle an OPTIONAL power-user mode (default off);")
        print("    keep it available to FORCE web on ambiguous/contextual turns")
        print("    until Fase C closes the cross-turn follow-up gap.")
    else:
        print("  → KEEP the manual toggle prominent; auto-detect still misses too")
        print("    much. Revisit after tightening intent patterns.")
    print("-" * 70)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({
                "thresholds": {"recall": THRESH_RECALL, "precision": THRESH_PRECISION},
                "meets_thresholds": meets,
                "metrics_overall": overall,
                "metrics_keyword_targetable": keyword,
                "metrics_contextual": contextual,
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nRaw results written to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
