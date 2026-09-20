#!/usr/bin/env bash
# Two decisions against a running bjev, one per answer type.
set -uo pipefail
BASE=${1:-http://127.0.0.1:8011}
AUTH=()
[ -n "${API_KEY:-}" ] && AUTH=(-H "authorization: Bearer ${API_KEY}")

echo "== health"
curl -sf "${AUTH[@]}" "$BASE/health" || { echo "unhealthy"; exit 1; }
echo

echo "== ticket triage"
curl -s "${AUTH[@]}" -X POST "$BASE/v1/systemone" -H 'content-type: application/json' -d '{
  "instructions": "Triage the support ticket.",
  "state": {"ticket": "Dashboard blank after login since 9am, demo at noon."},
  "questions": [
    {"id": "urgent", "type": "noul", "instructions": "Does this need a reply within the hour?"},
    {"id": "team", "type": "choice", "instructions": "Which team owns it?",
     "criteria": {"billing": null, "outage": null, "feature": null}},
    {"id": "tone", "type": "score", "instructions": "How angry is the customer?",
     "criteria": ["calm", "annoyed", "furious"]}
  ]}' | python3 -m json.tool
echo

echo "== four-way factual, the form that calibrates best"
curl -s "${AUTH[@]}" -X POST "$BASE/v1/systemone" -H 'content-type: application/json' -d '{
  "state": {},
  "questions": [
    {"id": "capital", "type": "choice", "instructions": "What is the capital of Australia?",
     "criteria": {"Sydney": null, "Canberra": null, "Melbourne": null, "Perth": null}}
  ]}' | python3 -m json.tool
