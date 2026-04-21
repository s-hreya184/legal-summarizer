import requests
import json
import re

OLLAMA_URL  = "http://localhost:11434/api/generate"
MAX_SINGLE  = 6000   

def call_llm(prompt: str, timeout: int = 180) -> str:
    """POST to Ollama and return the response string."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    # near-deterministic for extraction
                    "temperature": 0.05,  
                    "top_p": 0.9,
                    # increase context window if model supports it
                    "num_ctx": 8192,    
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Make sure it is running at http://localhost:11434")
    except requests.exceptions.Timeout:
        raise RuntimeError("LLM request timed out. The filtered text may still be too long — try lowering min_score.")
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")


# JSON extractor
def extract_json(text: str) -> str | None:
    """Robustly extract the first valid JSON object from LLM output."""
    clean = re.sub(r"```(?:json)?", "", text).strip()

    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Substring search fallback
    for i, ch in enumerate(clean):
        if ch == "{":
            for j in range(len(clean), i, -1):
                try:
                    json.loads(clean[i:j])
                    return clean[i:j]
                except json.JSONDecodeError:
                    continue
    return None


# Prompt builder
_EXTRACTION_PROMPT = """You are a strict Indian health insurance policy analyzer.

TASK: Extract ONLY information explicitly present in the policy clauses below.
These clauses have already been pre-filtered from a full policy document to contain
only the sections relevant to exclusions, waiting periods, co-payments, and limits.

RULES:
- Extract ONLY what is explicitly written. Do NOT invent or infer.
- Return empty lists [] when a category has no matches in the text.
- DO NOT mention suicide, self-harm, or mental health unless explicitly written.
- Every item must be traceable to an actual sentence in the text.

RISK SCORE GUIDE (0–100):
  0–30  : Few exclusions, short waiting periods, low co-pay — policy is claimant-friendly
  31–60 : Moderate exclusions or waiting periods — some financial exposure
  61–80 : Many exclusions, long waiting periods, or high co-pay — significant risk
  81–100: Extensive exclusions, multiple co-pays, very long waiting periods — high rejection risk

OUTPUT: Respond with ONLY a valid JSON object. No preamble, no explanation, no markdown fences.

{{
  "risk_score": <integer 0-100>,
  "waiting_periods": [
    {{"condition": "<name>", "duration": "<e.g. 2 years>", "impact": "<plain English consequence>"}}
  ],
  "exclusions": [
    {{"item": "<excluded item>", "impact": "<what the policyholder must pay themselves>"}}
  ],
  "co_payment": [
    {{"percentage": "<e.g. 20%>", "condition": "<when it applies>", "impact": "<cost consequence>"}}
  ],
  "hidden_limits": [
    {{"limit": "<description>", "applies_to": "<treatment or scenario>", "impact": "<consequence>"}}
  ],
  "danger_alerts": [
    {{"severity": "<Critical|High|Medium>", "message": "<plain language warning>"}}
  ]
}}

Policy Clauses:
{text}
"""


def _parse_result(raw: str) -> dict | None:
    """Parse and validate LLM output into a clean dict."""
    json_string = extract_json(raw)
    if not json_string:
        print("WARNING: No JSON found in LLM output.")
        print("RAW PREVIEW:", raw[:400])
        return None
    try:
        parsed = json.loads(json_string)
        defaults = {
            "risk_score": 0,
            "waiting_periods": [],
            "exclusions": [],
            "co_payment": [],
            "hidden_limits": [],
            "danger_alerts": [],
        }
        defaults.update(parsed)
        # Clamp risk score
        defaults["risk_score"] = max(0, min(100, int(defaults["risk_score"])))
        return defaults
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON PARSE FAILED: {e}")
        return None


def _merge_results(a: dict, b: dict) -> dict:
    """Merge two extraction results. Risk score = max of both."""
    merged = {
        "risk_score": max(a.get("risk_score", 0), b.get("risk_score", 0)),
        "waiting_periods": a.get("waiting_periods", []) + b.get("waiting_periods", []),
        "exclusions":      a.get("exclusions", [])      + b.get("exclusions", []),
        "co_payment":      a.get("co_payment", [])      + b.get("co_payment", []),
        "hidden_limits":   a.get("hidden_limits", [])   + b.get("hidden_limits", []),
        "danger_alerts":   a.get("danger_alerts", [])   + b.get("danger_alerts", []),
    }
    return merged


# Public API
def insurance_decoder(filtered_text: str) -> dict | None:
    """
    Analyse pre-filtered policy text with 1–2 LLM calls.

    If filtered_text <= MAX_SINGLE chars: 1 call.
    If filtered_text >  MAX_SINGLE chars: split into 2 halves at a paragraph
    boundary, run 2 calls, merge results.

    Returns the same dict shape as before so app.py needs no changes.
    """
    text = filtered_text.strip()

    if not text:
        return None

    if len(text) <= MAX_SINGLE:
        # Single call
        raw = call_llm(_EXTRACTION_PROMPT.format(text=text))
        return _parse_result(raw)

    else:
        # Two-call split
        # Find a paragraph boundary near the midpoint
        mid = len(text) // 2
        split_at = text.rfind("\n\n", mid - 500, mid + 500)
        if split_at == -1:
            split_at = text.rfind("\n", mid - 200, mid + 200)
        if split_at == -1:
            split_at = mid

        part_a = text[:split_at].strip()
        part_b = text[split_at:].strip()

        print(f"  [llm] Filtered text too long ({len(text)} chars), splitting into 2 calls.")

        raw_a = call_llm(_EXTRACTION_PROMPT.format(text=part_a))
        result_a = _parse_result(raw_a) or {}

        raw_b = call_llm(_EXTRACTION_PROMPT.format(text=part_b))
        result_b = _parse_result(raw_b) or {}

        if not result_a and not result_b:
            return None

        if not result_a:
            return result_b
        if not result_b:
            return result_a

        return _merge_results(result_a, result_b)