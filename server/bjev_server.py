"""Jev's decision contract over a llama.cpp server.

One read answers every question in the schema: the questions go out as a
numbered list, the assistant turn is prefilled, and each answer is a single
token whose logprobs are restricted to that question's labels and renormalised.
Newlines separate the answers because the tokenizer packs bare letters together
and the positions stop lining up.
"""

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
UPSTREAM = None
TOP_LOGPROBS = 20
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "123456789"
EXTENSIONS = ("instructions", "samples", "think", "max_questions")


class SchemaError(ValueError):
    pass


def upstream(path, body, timeout=600):
    req = urllib.request.Request(
        UPSTREAM.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SchemaError(f"upstream {e.code}: {e.read()[:300].decode('replace')}") from None


def parse_schema(value):
    if not isinstance(value, dict) or not isinstance(value.get("questions"), list):
        raise SchemaError("schema: needs a questions array")
    out, seen = [], set()
    for q in value["questions"]:
        qid = str(q.get("id", "")).strip()
        if not qid or ":" in qid or "\n" in qid:
            raise SchemaError(f"question id {qid!r} must be non-empty, no ':' or newline")
        if qid in seen:
            raise SchemaError(f"duplicate question id {qid!r}")
        seen.add(qid)
        kind = q.get("type")
        if kind in ("noul", "bool", "boolean"):
            kind = "noul"
            crit = q.get("criteria") or {}
            choices = [("yes", crit.get("true")), ("no", crit.get("false"))]
            labels = ["Y", "N"]
        elif kind == "choice":
            opts = q.get("options") or q.get("criteria") or []
            if isinstance(opts, dict):
                opts = [{"name": k, "description": v} for k, v in opts.items()]
            choices = [
                (o["name"], o.get("description")) if isinstance(o, dict) else (str(o), None)
                for o in opts
            ]
            if len(choices) > len(LETTERS):
                raise SchemaError(f"question {qid!r}: at most {len(LETTERS)} options")
            labels = list(LETTERS[: len(choices)])
        elif kind == "score":
            levels = q.get("levels") or q.get("criteria") or []
            choices = [(str(x), None) for x in levels]
            if len(choices) > len(DIGITS):
                raise SchemaError(f"question {qid!r}: at most {len(DIGITS)} levels")
            labels = list(DIGITS[: len(choices)])
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        if len(choices) < 2:
            raise SchemaError(f"question {qid!r}: needs at least two answers")
        out.append({"id": qid, "type": kind, "instructions": q.get("instructions") or qid,
                    "choices": choices, "labels": labels})
    if not out:
        raise SchemaError("schema: needs at least one question")
    limit = int(value.get("max_questions") or ARGS.max_questions)
    if len(out) > limit:
        raise SchemaError(f"{len(out)} questions exceeds max_questions {limit}")
    return {"questions": out, "instructions": value.get("instructions"),
            "think": bool(value.get("think"))}


def system_text(schema):
    lines = []
    if schema["instructions"]:
        lines.append(str(schema["instructions"]).strip())
    lines.append(
        "Answer every question. Reply with one answer per line, in order, "
        "using only the letters shown. No numbering, no explanation."
    )
    for i, q in enumerate(schema["questions"], 1):
        opts = ", ".join(
            f"{lab}={name}" + (f" ({desc})" if desc else "")
            for lab, (name, desc) in zip(q["labels"], q["choices"])
        )
        lines.append(f"{i}. {q['instructions']} [{opts}]")
    return "\n".join(lines)


def build_prompt(schema, state_text):
    body = {"messages": [
        {"role": "system", "content": system_text(schema)},
        {"role": "user", "content": state_text},
    ]}
    if not schema["think"]:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    prompt = upstream("/apply-template", body)["prompt"]
    if not schema["think"]:
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def read(schema, state_text):
    prompt = build_prompt(schema, state_text)
    n = len(schema["questions"])
    d = upstream("/completion", {
        "prompt": prompt,
        "n_predict": n * 4 + 8,
        "n_probs": TOP_LOGPROBS,
        "temperature": 0,
        "cache_prompt": True,
        "stop": [],
    })
    probs = d.get("completion_probabilities") or []
    answers, taken = {}, []
    for entry in probs:
        tok = entry.get("token", "")
        if not tok.strip():
            continue
        taken.append(entry)
        if len(taken) == n:
            break
    if len(taken) != n:
        raise SchemaError(
            f"model produced {len(taken)} answers for {n} questions; "
            "the answer format did not hold"
        )
    for q, entry in zip(schema["questions"], taken):
        top = {}
        for cand in entry.get("top_logprobs") or entry.get("probs") or []:
            t = (cand.get("token") or "").strip().upper()
            p = cand.get("prob")
            if p is None:
                p = pow(2.718281828459045, cand.get("logprob", -99))
            if t and t not in top:
                top[t] = p
        raw = {lab: top.get(lab, 0.0) for lab in q["labels"]}
        mass = sum(raw.values())
        if mass > 0:
            norm = {lab: v / mass for lab, v in raw.items()}
        else:
            picked = (entry.get("token") or "").strip().upper()[:1]
            hit = {lab: (1.0 if lab == picked else 0.0) for lab in q["labels"]}
            total = sum(hit.values())
            norm = hit if total else {lab: 1.0 / len(q["labels"]) for lab in q["labels"]}
        answers[q["id"]] = {"norm": norm, "label_mass": min(1.0, mass),
                            "emitted": (entry.get("token") or "").strip()}
    return answers, d.get("timings") or {}


def jev_answer(q, scored):
    norm = scored["norm"]
    top = max(norm, key=norm.get)
    names = [name for name, _ in q["choices"]]
    by_name = {name: norm[lab] for lab, name in zip(q["labels"], names)}
    if q["type"] == "noul":
        return {"type": "noul", "noul": by_name.get("yes", 0.0)}
    if q["type"] == "choice":
        return {"type": "choice", "choice": names[q["labels"].index(top)],
                "probabilities": by_name, "confidence": norm[top]}
    score = sum((i + 1) * norm[lab] for i, lab in enumerate(q["labels"]))
    return {"type": "score", "score": score,
            "legend": {str(i): name for i, name in enumerate(names)},
            "probabilities": {str(i): norm[lab] for i, lab in enumerate(q["labels"])},
            "confidence": norm[top]}


def decide(body):
    schema = parse_schema(body)
    state = body.get("state")
    state_text = state if isinstance(state, str) else json.dumps(state or {})
    scored, timings = read(schema, state_text)
    answers, diag = {}, {}
    for q in schema["questions"]:
        s = scored[q["id"]]
        answers[q["id"]] = jev_answer(q, s)
        diag[q["id"]] = {"label_mass": s["label_mass"], "emitted": s["emitted"]}
    return {
        "model": ARGS.model,
        "answers": answers,
        "usage": {"input_tokens": timings.get("prompt_n"),
                  "output_tokens": timings.get("predicted_n")},
        "diagnostics": {
            "questions": diag,
            "timing": {"prompt_ms": timings.get("prompt_ms"),
                       "predicted_ms": timings.get("predicted_ms"),
                       "reads": 1},
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, payload):
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _auth_ok(self):
        if not ARGS.api_key:
            return True
        return self.headers.get("authorization", "") == f"Bearer {ARGS.api_key}"

    def do_GET(self):
        if self.path == "/health":
            try:
                upstream("/apply-template", {"messages": [{"role": "user", "content": "x"}]}, 10)
            except Exception as e:
                return self._json(503, {"status": "upstream unavailable", "detail": str(e)[:200]})
            return self._json(200, {"status": "ok", "model": ARGS.model})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list",
                                    "data": [{"id": ARGS.model, "object": "model"}]})
        return self._json(404, {"error": {"message": "unknown route"}})

    def do_POST(self):
        if not self._auth_ok():
            return self._json(401, {"error": {"message": "bad or missing bearer token"}})
        length = int(self.headers.get("content-length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError as e:
            return self._json(400, {"error": {"message": f"bad JSON: {e}"}})
        route = self.path.rstrip("/")
        if route == "/v1/systemone":
            try:
                return self._json(200, decide(body))
            except SchemaError as e:
                return self._json(400, {"error": {"message": str(e),
                                                  "type": "invalid_request_error"}})
        if route == "/v1/raw/completion":
            try:
                return self._json(200, upstream("/completion", body))
            except SchemaError as e:
                return self._json(502, {"error": {"message": str(e)}})
        return self._json(404, {"error": {"message": "unknown route"}})


def main():
    global ARGS, UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.environ.get("LLAMA_URL", "http://127.0.0.1:8010"))
    ap.add_argument("--model", default=os.environ.get("SERVED_NAME", "bonsai"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--max-questions", type=int, default=20)
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ARGS = ap.parse_args()
    UPSTREAM = ARGS.upstream
    print(f"bjev on {ARGS.host}:{ARGS.port} -> {UPSTREAM} ({ARGS.model})", flush=True)

    class Server(ThreadingHTTPServer):
        request_queue_size = 256
        daemon_threads = True

    Server((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
