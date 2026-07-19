#!/usr/bin/env python3
"""Knowledge-routing accuracy harness (Tahap 3).

Goal: measure how accurately the router picks the KNOWLEDGE SOURCE that should
ground each answer — personal_recall (persistent memory), past_chat (earlier
conversations), personal_docs (uploaded files / RAG), web (external/fresh), or
none (the model's own knowledge). Observability-first: this measures the
classification recorded in router_decisions.knowledge_source; it does not force
retrieval.

How it works:
  1. Log in as admin, create a throwaway session per scenario group.
  2. Send each labeled prompt to /api/chat_stream (mode=chat, manual web OFF).
  3. Read the router_decisions row back via the admin observability endpoint
     and compare `knowledge_source` against the scenario's expected label.
  4. Print a per-class confusion matrix + macro precision/recall/F1 + the
     misrouted cases + a recommendation.

Usage:
  python scripts/knowledge_scenarios.py \
      --base-url http://127.0.0.1:7000 \
      --user admin --password '<admin-password>'

Exit code is always 0 (measurement tool, not a gate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


CLASSES = ["personal_recall", "past_chat", "personal_docs", "web", "none"]

# ---------------------------------------------------------------------------
# Labeled dataset (ID + EN). `expected` is the knowledge source a correct
# router should record. `followup` chains onto the previous turn in the SAME
# session (so context-dependent phrasings can be exercised).
# ---------------------------------------------------------------------------
SCENARIOS = [
    # ── personal_recall: the user's persistent memory ─────────────────────
    {"group": "personal_recall", "prompt": "what did I tell you about my sister?", "expected": "personal_recall"},
    {"group": "personal_recall", "prompt": "do you remember my favorite color?", "expected": "personal_recall"},
    {"group": "personal_recall", "prompt": "what do you know about me?", "expected": "personal_recall"},
    {"group": "personal_recall", "prompt": "what is my birthday?", "expected": "personal_recall"},
    {"group": "personal_recall", "prompt": "apa yang pernah aku bilang ke kamu soal pekerjaanku?", "expected": "personal_recall"},
    {"group": "personal_recall", "prompt": "kamu masih inget alamat aku?", "expected": "personal_recall"},
    {"group": "personal_recall", "prompt": "apa nama saya?", "expected": "personal_recall"},

    # ── past_chat: earlier conversations ──────────────────────────────────
    {"group": "past_chat", "prompt": "what did we discuss yesterday?", "expected": "past_chat"},
    {"group": "past_chat", "prompt": "in our previous conversation you mentioned a plan", "expected": "past_chat"},
    {"group": "past_chat", "prompt": "you told me earlier about the config", "expected": "past_chat"},
    {"group": "past_chat", "prompt": "obrolan kita kemarin soal apa?", "expected": "past_chat"},
    {"group": "past_chat", "prompt": "apa yang kita bahas waktu itu?", "expected": "past_chat"},
    {"group": "past_chat", "prompt": "kamu bilang tadi soal harga", "expected": "past_chat"},

    # ── personal_docs: uploaded files / RAG ───────────────────────────────
    {"group": "personal_docs", "prompt": "according to the document I uploaded, what is the deadline?", "expected": "personal_docs"},
    {"group": "personal_docs", "prompt": "in my report, summarize section 2", "expected": "personal_docs"},
    {"group": "personal_docs", "prompt": "based on the file I shared earlier", "expected": "personal_docs"},
    {"group": "personal_docs", "prompt": "menurut dokumen yang aku upload, berapa totalnya?", "expected": "personal_docs"},
    {"group": "personal_docs", "prompt": "di file laporan itu ada angka berapa?", "expected": "personal_docs"},

    # ── web: external / explicitly-web phrasing ───────────────────────────
    {"group": "web", "prompt": "search the web for python 3.14 release notes", "expected": "web"},
    {"group": "web", "prompt": "cari di internet harga emas hari ini", "expected": "web"},
    {"group": "web", "prompt": "what is the latest news about the election", "expected": "web"},
    {"group": "web", "prompt": "current stock price of Apple", "expected": "web"},

    # ── none: answer from the model's own knowledge ───────────────────────
    {"group": "none", "prompt": "what is the capital of France?", "expected": "none"},
    {"group": "none", "prompt": "explain how recursion works", "expected": "none"},
    {"group": "none", "prompt": "how do I upload a document?", "expected": "none"},
    {"group": "none", "prompt": "write a haiku about autumn", "expected": "none"},
    {"group": "none", "prompt": "apa itu fotosintesis?", "expected": "none"},
    {"group": "none", "prompt": "halo apa kabar", "expected": "none"},
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
    chain_session = {}
    for i, sc in enumerate(SCENARIOS):
        group = sc["group"]
        if sc.get("followup") and group in chain_session:
            session = chain_session[group]
        else:
            session = c.create_session(f"[harness-kn] {group} {i}", args.model, args.endpoint_id)
            if sc.get("chain_start"):
                chain_session[group] = session

        code = c.send(session, sc["prompt"])
        time.sleep(0.5)
        dec = c.last_decision(session) or {}
        got = dec.get("knowledge_source") or "none"
        exp = sc["expected"]
        results.append({
            "group": group,
            "prompt": sc["prompt"],
            "expected": exp,
            "got": got,
            "knowledge_retrieval": dec.get("knowledge_retrieval"),
            "knowledge_reason": dec.get("knowledge_reason"),
            "intent_category": dec.get("intent_category"),
            "exit_path": dec.get("exit_path"),
            "correct": got == exp,
            "http": code,
        })
        mark = "OK " if got == exp else "XX "
        print(f"{mark}[{group:16}] exp={exp:16} got={got:16} :: {sc['prompt'][:50]}")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0.0

    # ── Per-class confusion + macro metrics ───────────────────────────────
    labels = CLASSES
    tp = {k: 0 for k in labels}
    fp = {k: 0 for k in labels}
    fn = {k: 0 for k in labels}
    for r in results:
        e, g = r["expected"], r["got"]
        if e == g:
            tp[e] = tp.get(e, 0) + 1
        else:
            fp[g] = fp.get(g, 0) + 1
            fn[e] = fn.get(e, 0) + 1

    def prf(k):
        p = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) else 0.0
        rc = tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) else 0.0
        f = (2 * p * rc / (p + rc)) if (p + rc) else 0.0
        return p, rc, f

    print("\n" + "=" * 74)
    print("KNOWLEDGE-ROUTING ACCURACY REPORT (manual web toggle OFF)")
    print("=" * 74)
    print(f"Total scenarios : {total}")
    print(f"Overall accuracy: {accuracy:.2f} ({correct}/{total})")
    print("\nPer-class metrics:")
    print(f"  {'class':16} {'support':>7} {'prec':>6} {'recall':>7} {'f1':>6}")
    support = {k: sum(1 for r in results if r['expected'] == k) for k in labels}
    macro_p = macro_r = macro_f = 0.0
    counted = 0
    for k in labels:
        p, rc, f = prf(k)
        print(f"  {k:16} {support[k]:>7} {p:>6.2f} {rc:>7.2f} {f:>6.2f}")
        if support[k]:
            macro_p += p
            macro_r += rc
            macro_f += f
            counted += 1
    if counted:
        print(f"  {'MACRO AVG':16} {'':>7} {macro_p/counted:>6.2f} {macro_r/counted:>7.2f} {macro_f/counted:>6.2f}")

    misrouted = [r for r in results if not r["correct"]]
    if misrouted:
        print(f"\nMISROUTED ({len(misrouted)}):")
        for r in misrouted:
            print(f"  - [{r['group']}] exp={r['expected']} got={r['got']} :: {r['prompt']}")

    # ── Recommendation ────────────────────────────────────────────────────
    print("\n" + "-" * 74)
    print("RECOMMENDATION (knowledge routing):")
    macro_recall = (macro_r / counted) if counted else 0.0
    if accuracy >= 0.85 and macro_recall >= 0.8:
        print("  Knowledge routing is reliable → safe to start LIGHTLY nudging tool")
        print("  availability toward the detected source in a later stage.")
    elif accuracy < 0.6:
        print("  Knowledge routing misroutes often → keep observability-only; refine")
        print("  patterns (esp. Indonesian) before influencing retrieval.")
    else:
        print("  Knowledge routing is partial → keep observability-only for now;")
        print("  expand patterns for weak classes before acting on the signal.")
    print("-" * 74)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({
                "accuracy": accuracy,
                "total": total,
                "support": support,
                "per_class": {k: dict(zip(("precision", "recall", "f1"), prf(k))) for k in labels},
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nRaw results written to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
