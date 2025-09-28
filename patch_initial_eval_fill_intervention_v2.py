
# patch_initial_eval_fill_intervention_v2.py
# Self-contained filler for mini_intervention_items.csv using Ollama HTTP API.
# - Uses your query2_llm/query_llm if available in globals()
# - Otherwise falls back to direct HTTP calls to OLLAMA /api/generate
# - Also defines its own wait_for_ollama()

import os, time, csv, requests, pandas as pd

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME  = os.environ.get("MODEL_NAME",  "llama3")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
NUM_PREDICT = int(os.environ.get("MAX_TOKENS", "128"))

_session = requests.Session()

def wait_for_ollama(timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = _session.get(f"{OLLAMA_HOST}/api/version", timeout=5)
            if r.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False

def _gen_ollama(prompt: str) -> str:
    # Direct call to /api/generate with sane defaults.
    try:
        r = _session.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": NUM_PREDICT,
                }
            },
            timeout=120
        )
        r.raise_for_status()
        js = r.json()
        return (js.get("response") or "").strip()
    except Exception as e:
        return "unsure"

def _choose_generator():
    # If the user's notebook defined helpers, use them
    g = globals()
    if "query2_llm" in g and callable(g["query2_llm"]):
        return g["query2_llm"]
    if "query_llm" in g and callable(g["query_llm"]):
        return g["query_llm"]
    return _gen_ollama

def run_language_intervention_eval(items_csv="mini_intervention_items.csv",
                                   out_csv="mini_intervention_eval_template_filled.csv",
                                   sleep_s=0.2):
    if not wait_for_ollama(30):
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_HOST}. Start the Ollama server first.")

    gen = _choose_generator()

    df = pd.read_csv(items_csv)

    def _call(prompt):
        try:
            return str(gen(prompt)).strip(), ""
        except Exception as e:
            return "unsure", str(e)

    rows = []
    for _, r in df.iterrows():
        base_out, base_err = _call(r["baseline_prompt"])
        intr_out, intr_err = _call(r["intervention_prompt"])
        rows.append({
            "item_id": r["item_id"],
            "condition": r["condition"],
            "answer_key": r["answer_key"],
            "baseline_response": base_out,
            "intervention_response": intr_out,
            "baseline_fluency_1to5": "",
            "intervention_fluency_1to5": "",
            "baseline_error": base_err,
            "intervention_error": intr_err
        })
        time.sleep(sleep_s)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["item_id","condition","answer_key",
                      "baseline_response","intervention_response",
                      "baseline_fluency_1to5","intervention_fluency_1to5",
                      "baseline_error","intervention_error"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[done] Wrote {out_csv} with {len(rows)} rows.")
