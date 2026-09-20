"""Runs the decision server against a fake llama.cpp. No model, no GPU."""

import json
import math
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bjev_server as S  # noqa: E402

REPLY = {"tokens": [], "probs": {}}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/apply-template":
            out = {"prompt": "<|im_start|>assistant\n"}
        else:
            comp = []
            for tok in REPLY["tokens"]:
                top = [{"token": t, "prob": p} for t, p in REPLY["probs"].get(tok, {}).items()]
                comp.append({"token": tok, "top_logprobs": top})
            out = {"content": "".join(REPLY["tokens"]),
                   "completion_probabilities": comp,
                   "timings": {"prompt_n": 42, "predicted_n": len(comp),
                               "prompt_ms": 10.0, "predicted_ms": 20.0}}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    fake = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    S.ARGS = type("A", (), {"model": "bonsai", "max_questions": 20, "api_key": ""})()
    S.UPSTREAM = f"http://127.0.0.1:{fake.server_address[1]}"

    app = ThreadingHTTPServer(("127.0.0.1", 0), S.Handler)
    threading.Thread(target=app.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{app.server_address[1]}"
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{'' if cond else '  ' + str(detail)}")
        ok = ok and cond

    schema = {"questions": [
        {"id": "urgent", "type": "noul", "instructions": "Urgent?"},
        {"id": "team", "type": "choice", "instructions": "Which team?",
         "criteria": {"billing": None, "outage": None, "feature": None}},
        {"id": "tone", "type": "score", "instructions": "How angry?",
         "criteria": ["calm", "annoyed", "furious"]},
    ], "state": {"ticket": "dashboard blank"}}

    REPLY["tokens"] = ["Y", "\n", "B", "\n", "2", "\n"]
    REPLY["probs"] = {"Y": {"Y": 0.8, "N": 0.2}, "B": {"A": 0.1, "B": 0.7, "C": 0.2},
                      "2": {"1": 0.2, "2": 0.6, "3": 0.2}}
    code, d = post(base, "/v1/systemone", schema)
    check("three questions answered in one read", code == 200 and len(d.get("answers", {})) == 3, d)
    a = d["answers"]
    check("noul reports p(yes)", abs(a["urgent"]["noul"] - 0.8) < 1e-6, a["urgent"])
    check("choice maps the label to its option name", a["team"]["choice"] == "outage", a["team"])
    check("choice probabilities are named and normalised",
          abs(sum(a["team"]["probabilities"].values()) - 1.0) < 1e-6, a["team"])
    check("score is the expected level, not the argmax index",
          abs(a["tone"]["score"] - 2.0) < 1e-6, a["tone"])
    check("one read per decision", d["diagnostics"]["timing"]["reads"] == 1)

    # labels outside the returned top-k leave no mass, so the emitted token decides
    REPLY["probs"]["Y"] = {"maybe": 0.9}
    code, d = post(base, "/v1/systemone", schema)
    check("a label-free distribution falls back to the emitted token",
          code == 200 and d["answers"]["urgent"]["noul"] == 1.0
          and d["diagnostics"]["questions"]["urgent"]["label_mass"] == 0.0,
          d.get("diagnostics"))
    REPLY["probs"]["Y"] = {"Y": 0.8, "N": 0.2}

    # the count assertion is the one that catches a broken answer format
    REPLY["tokens"] = ["Y", "\n", "B", "\n"]
    code, d = post(base, "/v1/systemone", schema)
    check("a short answer list is refused, not silently scored",
          code == 400 and "did not hold" in d["error"]["message"], d)
    REPLY["tokens"] = ["Y", "\n", "B", "\n", "2", "\n"]

    code, d = post(base, "/v1/systemone", {"questions": []})
    check("empty schema refused", code == 400, d)
    code, d = post(base, "/v1/systemone", {"questions": [
        {"id": "a", "type": "choice", "criteria": {"only": None}}]})
    check("a single-option choice refused", code == 400, d)
    code, d = post(base, "/v1/systemone", {"questions": [
        {"id": "x:y", "type": "noul"}]})
    check("a colon in an id refused", code == 400, d)
    code, d = post(base, "/v1/systemone", {"questions": [
        {"id": f"q{i}", "type": "noul"} for i in range(25)]})
    check("too many questions refused", code == 400, d)

    S.ARGS.api_key = "secret"
    code, _ = post(base, "/v1/systemone", schema)
    check("bearer token enforced when set", code == 401)
    S.ARGS.api_key = ""

    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
