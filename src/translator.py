#!/usr/bin/env python3
"""
Translate a Jupyter/Colab .ipynb notebook from Hungarian to English using the OpenAI API.

Design goals:
- Keep the notebook JSON structure and all fields intact.
- Only modify nb["cells"][i]["source"] for each cell.
- Batch multiple cells per API call (dynamic batching by token estimate).
- Use Structured Outputs (JSON Schema) to reliably parse the model response.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import logging 

from openai import OpenAI


# -----------------------------
# Global settings (edit here)
# -----------------------------

# Fastest/cheapest for testing:
MODEL = "gpt-4.1-nano"
MODEL = "gpt-5-mini"

# More capable (commented out):
# MODEL = "gpt-4.1"          # higher quality translation/reasoning
# MODEL = "gpt-5-mini"       # strong, typically pricier than nano
# MODEL = "gpt-5.2"          # strongest, typically most expensive

# Context window (tokens). Used only for batching heuristic.
# gpt-4.1-nano has ~1,047,576 token context (per docs). If you switch models, update this.
MODEL_CONTEXT_TOKENS = 1_047_576
# MODEL_CONTEXT_TOKENS = 400_000 # GPT-5, gpt-5-mini

# How much of context to use for INPUT (prompt + payload). Keep conservative.
INPUT_CONTEXT_FRACTION = 0.60

# Output cap. Keep high enough to avoid truncation, but not insane.
# Note: some models have a hard max output (e.g., 32,768). If your SDK/model rejects large values, lower this.
MAX_OUTPUT_TOKENS = 16_384

# Primitive token estimator: tokens ≈ chars / 4 (good enough for conservative batching).
def estimate_tokens_from_chars(n_chars: int) -> int:
    return max(1, n_chars // 4)


# Single prompt with cell_type discriminator:
INSTRUCTION_PROMPT = """You are translating notebook cells from Hungarian to English.

You will receive a JSON object with an array named "items".
Each item has:
- "index": the cell index in the notebook
- "cell_type": either "markdown" or "code"
- "source": an array of strings (lines/fragments) representing the cell source

TASK:
Return a JSON object that matches the provided schema, with "translations" containing one entry per input item.

Rules:
1) Preserve the notebook cell order by returning one translation per item (use the same index).
2) Only translate Hungarian content to English.
3) Keep numbers, math, file paths, URLs, and code syntax intact.
4) For markdown cells:
   - Preserve Markdown formatting (headings, links, lists, tables).
   - Translate code inside fenced code blocks (``` ... ```) exactly as for code cells (see below).
5) For code cells:
   - Translate only human-language parts: comments (# ...), docstrings, and user-facing strings (e.g. text printed to users),
     as long as changing them does not break code.
   - Avoid renaming well-known library aliases like: np, pd, plt, tf, torch, sklearn, cv2 (keep these unchanged).
   - Translate hungarian variable names to English equivalent. Use the same translation for the same Hungarian word across code blocks.
   - Never produce invalid Python identifiers; follow Python naming rules.

Output requirements:
- Output MUST be valid JSON matching the given schema.
- Do not include any extra keys or commentary outside the JSON.
"""


# JSON Schema for Structured Outputs
TRANSLATION_SCHEMA: Dict[str, Any] = {
    "name": "NotebookCellTranslations",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "translated_source": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["index", "translated_source"],
                },
            }
        },
        "required": ["translations"],
    },
}


# -----------------------------
# Core logic
# -----------------------------

def load_notebook(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(nb: Dict[str, Any], path: str) -> None:
    # Keep as minimal as possible; note that JSON serialization will still reformat.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, separators=(",", ":"), indent=2)


def normalize_source_to_list(source: Any) -> List[str]:
    if source is None:
        return []
    if isinstance(source, list):
        # Ensure all elements are strings
        return ["" if s is None else str(s) for s in source]
    if isinstance(source, str):
        # Wrap single string as a one-element list
        return [source]
    # Unexpected type; coerce
    return [str(source)]


def build_payload_items(nb: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    """
    Returns:
    - items: list of payload items (index, cell_type, source[list[str]])
    - original_source_type: mapping index -> "list" or "str" (to restore exact type if desired)
    """
    items: List[Dict[str, Any]] = []
    original_source_type: Dict[int, str] = {}

    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError("Invalid notebook: nb['cells'] is not a list")

    for i, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        cell_type = cell.get("cell_type", "")
        if cell_type not in ("markdown", "code"):
            # Skip other cell types (raw, etc.) but keep them unchanged
            continue

        src = cell.get("source", [])
        original_source_type[i] = "list" if isinstance(src, list) else "str" if isinstance(src, str) else "other"
        src_list = normalize_source_to_list(src)

        # Optional: skip empty sources
        if not any(s.strip() for s in src_list):
            continue

        items.append(
            {
                "index": i,
                "cell_type": cell_type,
                "source": src_list,
            }
        )

    return items, original_source_type


def approximate_input_tokens(prompt: str, payload_obj: Dict[str, Any]) -> int:
    payload_json = json.dumps(payload_obj, ensure_ascii=False, separators=(",", ":"))
    # Rough input = prompt + payload
    return estimate_tokens_from_chars(len(prompt) + len(payload_json))


def translate_batch(client: OpenAI, batch_items: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    Returns mapping: cell_index -> translated_source (list[str])
    """
    payload = {"items": batch_items}

    logging.info("---------------- modell request ----------------")
    logging.info(f"first_block: {batch_items[0]}")
    logging.info(f"last_block: {batch_items[-1]}")
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": INSTRUCTION_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                **TRANSLATION_SCHEMA,
            }
        },
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    # dump resp to file for debug
    with open("debug.openai.resp.json", "w", encoding="utf-8") as f:
        f.write(f"{resp}\n")

    try:
        logging.info(f"""Model output: 
            id:{resp.id}
            error:{resp.error}
            metadata:{resp.metadata}
            model:{resp.model}
            object:{resp.object}
            reasoning:{resp.reasoning}
            usage:{resp.usage}
        """)

        message = resp.output[0]
        logging.info(f"""Message0:
            id:{message.id}
            status:{message.status}
            type:{message.type}
        """)
    except (IndexError, KeyError):
        logging.error("Model output JSON has unexpected structure")

    # parse text from json
    data = json.loads(resp.output_text)

    translations = data.get("translations", [])
    if not isinstance(translations, list):
        raise RuntimeError("Model output JSON missing 'translations' list.")

    out: Dict[int, List[str]] = {}
    for t in translations:
        if not isinstance(t, dict):
            continue
        idx = t.get("index")
        ts = t.get("translated_source")
        if isinstance(idx, int) and isinstance(ts, list) and all(isinstance(x, str) for x in ts):
            out[idx] = ts

    # Validate coverage
    expected = {it["index"] for it in batch_items}
    got = set(out.keys())
    missing = expected - got
    extra = got - expected
    if missing or extra:
        #raise RuntimeError(f"Translation batch mismatch. missing={sorted(missing)} extra={sorted(extra)}")
        logging.error(f"Translation batch mismatch. missing={sorted(missing)} extra={sorted(extra)}")

    return out


def dynamic_batches(items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Create batches of items based on estimated tokens up to INPUT_CONTEXT_FRACTION of context window.
    """
    target_input_tokens = int(MODEL_CONTEXT_TOKENS * INPUT_CONTEXT_FRACTION)

    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    # We'll account for prompt overhead + JSON wrapper overhead by testing each append.
    for it in items:
        if not current:
            current = [it]
            continue

        tentative = current + [it]
        tok = approximate_input_tokens(INSTRUCTION_PROMPT, {"items": tentative})
        if tok <= target_input_tokens:
            current = tentative
        else:
            batches.append(current)
            current = [it]

    if current:
        batches.append(current)

    return batches


def apply_translations(nb: Dict[str, Any], translations: Dict[int, List[str]], original_source_type: Dict[int, str]) -> None:
    cells = nb.get("cells", [])
    for idx, translated_source in translations.items():
        if idx < 0 or idx >= len(cells):
            continue
        cell = cells[idx]
        if not isinstance(cell, dict):
            continue

        # Restore original type if it was a string.
        # Note: restoring a string means joining list elements. This may slightly change segmentation,
        # but keeps the content correct. If you prefer always list, you can remove this branch.
        src_type = original_source_type.get(idx, "list")
        if src_type == "str":
            cell["source"] = "".join(translated_source)
        else:
            cell["source"] = translated_source


def translate_notebook(nb: Dict[str, Any], client: OpenAI, max_retries: int = 2) -> Dict[str, Any]:
    nb_out = copy.deepcopy(nb)

    items, original_source_type = build_payload_items(nb_out)
    if not items:
        return nb_out

    batches = dynamic_batches(items)

    all_translations: Dict[int, List[str]] = {}
    for b_idx, batch in enumerate(batches, start=1):
        attempt = 0
        while True:
            attempt += 1
            try:
                logging.info(f"Translating batch {b_idx} of {len(batches)}")
                result = translate_batch(client, batch)
                all_translations.update(result)
                break
            except Exception as e:
                raise
                if attempt > max_retries:
                    raise
                # Backoff + shrink batch if needed
                time.sleep(0.8 * attempt)

                # If batch is large, split it to reduce risk (output limit / occasional mismatch)
                if len(batch) > 1:
                    mid = len(batch) // 2
                    left = batch[:mid]
                    right = batch[mid:]
                    # Translate halves sequentially with retries
                    left_res = translate_batch(client, left)
                    right_res = translate_batch(client, right)
                    all_translations.update(left_res)
                    all_translations.update(right_res)
                    break
                else:
                    # Single item failed; just retry
                    continue

    apply_translations(nb_out, all_translations, original_source_type)
    return nb_out


# -----------------------------
# CLI
# -----------------------------

def derive_output_path(input_path: str) -> str:
    if input_path.lower().endswith(".ipynb"):
        return input_path[:-6] + "_en.ipynb"
    return input_path + "_en.ipynb"


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Translate .ipynb (Hungarian -> English) using OpenAI Structured Outputs.")
    parser.add_argument("input", help="Path to input .ipynb")
    parser.add_argument("-o", "--output", help="Path to output .ipynb (default: *_en.ipynb)")
    args = parser.parse_args()

    # Load API key from environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key)

    nb = load_notebook(args.input)
    out_path = args.output or derive_output_path(args.input)

    nb_translated = translate_notebook(nb, client)

    save_notebook(nb_translated, out_path)
    print(f"Saved translated notebook to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
