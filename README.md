# bjev-spark

Ternary Bonsai 27B on a DGX Spark, answering Jev's decision contract.

## Run

```bash
cp .env.example .env      # point MODELS_DIR at the directory holding the .gguf
docker compose up -d
scripts/smoke.sh
```

`POST /v1/systemone` takes `{"state", "questions"}` and answers in Jev's shapes.
Every question in a schema is answered by one read.

## Why this is not djev-spark

| | djev-spark | bjev-spark |
|---|---|---|
| weights | NVFP4 safetensors | GGUF, ternary or 1-bit |
| runtime | patched vLLM | PrismML's llama.cpp fork |
| read | seeded diffusion canvas, one forward for every question | autoregressive, one answer token per question |
| label ceiling | exact logprobs per label id | whatever reaches the top 20 |

The quantisations are readable only by PrismML's fork, which is why the runtime
differs. The read differs because Bonsai is autoregressive: answers are decoded
in sequence rather than filled into a canvas, so a schema's cost grows with the
number of questions.

## Choosing a quantisation

| | Q1_0 (1-bit) | PQ2_0 (ternary) |
|---|---|---|
| size | 3.53 GiB | 6.66 GiB |
| tg128 | 49.0 t/s | 29.9 t/s |
| yes/no battery | 86% | 92% |
| four-way battery | 88% | 95% |
| arithmetic tier | 50-70% | 80-90% |

The gap between them is arithmetic. Use the 1-bit build for
classification-shaped work and the ternary build when anything numeric is
involved or you intend to threshold on confidence.

## Layout

| path | |
|---|---|
| `Dockerfile` | builds the fork at a pinned ref for sm_121, then a runtime stage |
| `entrypoint.sh` | memory guard, llama-server, decision server with a restart loop |
| `server/bjev_server.py` | the decision contract |
| `server/test_bjev_server.py` | runs against a fake llama.cpp, no model needed |
| `scripts/smoke.sh` | two decisions against a running container |

## Notes

Answers are newline separated because the tokenizer packs bare letters into one
token and the answer positions stop lining up. The server refuses a reply whose
answer count does not match the schema rather than scoring a partial one.

Four-way questions calibrate better than yes/no: a binary question leaves the
probability mass nowhere to go, and the model will be confidently wrong. Prefer
four options with an abstain threshold around 0.85.

Concurrency does not help. Prefix caching is the optimisation, and concurrent
requests land on separate slots that each recompute the shared prefix, so
`--parallel 1` with a stable system prompt beats fanning out.

To reload the decision server without reloading the weights, copy the file in
and `pkill -f bjev_server.py`.
