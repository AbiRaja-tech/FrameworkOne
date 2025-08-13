# commonsense_agent.py
import os
import time
import logging
import argparse, json, torch, re
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from fewshot import EXAMPLES, CATEGORY_KEYWORDS

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = """You are a travel-planning policy generator.
Given a natural-language trip brief, reply with a single, strict JSON object
containing the knobs/constraints that downstream planners use.
No prose. No code fences. JSON only.

The JSON shape should match the assistant examples provided in context.
"""

# ------------------------
# Logging / timing helpers
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("commonsense")

def _now(): return time.perf_counter()
def _timed(label: str, t0: float) -> float:
    dt = _now() - t0
    log.info(f"{label} in {dt:.2f}s")
    return _now()

# ------------------------
# Routing helpers
# ------------------------
def score_category(query: str, keywords: List[str]) -> int:
    q = query.lower()
    return sum(1 for kw in keywords if kw in q)

def route_category(query: str, min_score: int = 1) -> List[str]:
    ranked = sorted(
        ((cat, score_category(query, kws)) for cat, kws in CATEGORY_KEYWORDS.items()),
        key=lambda x: x[1],
        reverse=True
    )
    top = [cat for cat, s in ranked if s >= min_score]
    return top or [cat for cat, _ in ranked[:3]]

def select_examples(categories: List[str], k_per_cat: int = 1) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for cat in categories:
        ids = {e["id"] for e in EXAMPLES if e.get("category") == cat}
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for _id in ids:
            u = next((e for e in EXAMPLES if e["id"] == _id and e["role"] == "user"), None)
            a = next((e for e in EXAMPLES if e["id"] == _id and e["role"] == "assistant"), None)
            if u and a:
                pairs.append((u, a))
        for (u, a) in pairs[:k_per_cat]:
            selected.extend([u, a])
    return selected

# ------------------------
# Prompt builder
# ------------------------
def build_messages(user_request: str, fewshot_msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in fewshot_msgs:
        content = m["content"]
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        msgs.append({"role": m["role"], "content": content})
    msgs.append({"role": "user", "content": user_request})
    return msgs

# ------------------------
# Model call (4-bit on NVIDIA)
# ------------------------
def _gpu_index_from_device_str(device_str: str) -> int:
    # "cuda:0" -> 0 ; "cuda:1" -> 1 ; otherwise -1
    try:
        if device_str.startswith("cuda:"):
            return int(device_str.split(":")[1])
    except Exception:
        pass
    return -1

def call_deepseek(messages: List[Dict[str, str]], device_str: str = "cuda:0", max_new_tokens: int = 256) -> str:
    """
    Loads DeepSeek 7B in 4-bit (bitsandbytes) on the requested CUDA device so it fits in 8GB VRAM.
    Returns RAW model text (no JSON parsing).
    """
    # Strongly recommended when you want to target GPU1:
    # Example: CUDA_VISIBLE_DEVICES=1 python ... --device cuda:0
    # Inside the process, cuda:0 == physical GPU1.

    gpu_idx = _gpu_index_from_device_str(device_str)
    using_cuda = gpu_idx >= 0

    if using_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Install CUDA PyTorch + NVIDIA driver, or pass --device cpu.")

    # Speed knobs (safe on Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if using_cuda:
        log.info(f"Requested device: {device_str}")
        log.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        log.info(f"torch.version.cuda: {torch.version.cuda}")
        log.info(f"Active GPU [{gpu_idx}] name: {torch.cuda.get_device_name(gpu_idx)}")

    # 4-bit quant config (fits 7B into ~5–6 GB VRAM)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # fp16 compute on NVIDIA
    )

    # Tokenizer
    t = _now()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    t = _timed("Loaded tokenizer", t)

    # Quantized model directly on the chosen CUDA device
    # device_map expects a dict; "" maps the whole model to that device index.
    t = _now()
    device_map = {"": gpu_idx} if using_cuda else {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map=device_map,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    t = _timed("Loaded 4-bit model", t)

    # pad_token hygiene
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build chat prompt
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "".join(f"{m['role'].upper()}: {m['content']}\n" for m in messages) + "ASSISTANT:"
    t = _timed("Built prompt", t)

    # Show prompt (comment out if too verbose)
    log.info("==== PROMPT START ====")
    log.info(prompt)
    log.info("==== PROMPT END ====")

    # Some transformers versions prefer int device id here; -1 means CPU
    pipe_device = gpu_idx if using_cuda else -1

    t = _now()
    textgen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    t = _timed("Initialized generation pipeline", t)

    gen_t0 = _now()
    out = textgen(prompt)[0]["generated_text"]
    if using_cuda:
        torch.cuda.synchronize(gpu_idx)
    log.info(f"Generation finished in {_now()-gen_t0:.2f}s")

    # Return RAW model text
    return out[len(prompt):].strip() if out.startswith(prompt) else out.strip()

# ------------------------
# JSON extraction helper
# ------------------------
def extract_json(raw_text: str) -> Dict[str, Any]:
    """Extract JSON from raw model output."""
    try:
        # Try to find JSON in the text
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # Clean the JSON string by removing any trailing non-printable characters
            json_str = json_str.rstrip('\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f')
            return json.loads(json_str)
        else:
            # If no JSON found, try to parse the entire text
            return json.loads(raw_text.strip())
    except json.JSONDecodeError as e:
        # Try to fix incomplete JSON by adding missing closing braces
        print(f"DEBUG: JSON decode error at position {e.pos}")
        print(f"DEBUG: Character at error position: {repr(raw_text[e.pos]) if e.pos < len(raw_text) else 'END'}")
        print(f"DEBUG: Last 10 chars before error: {repr(raw_text[max(0, e.pos-10):e.pos])}")
        print(f"DEBUG: First 10 chars after error: {repr(raw_text[e.pos:e.pos+10])}")
        
        # Try to fix incomplete JSON
        try:
            # Find the last opening brace
            last_open = raw_text.rfind('{')
            if last_open != -1:
                # Count braces to see if we need to close them
                open_count = raw_text[last_open:].count('{')
                close_count = raw_text[last_open:].count('}')
                
                if open_count > close_count:
                    # Add missing closing braces
                    missing_braces = open_count - close_count
                    fixed_json = raw_text[last_open:] + '}' * missing_braces
                    print(f"DEBUG: Attempting to fix JSON by adding {missing_braces} closing braces")
                    return json.loads(fixed_json)
                elif close_count > open_count:
                    # Remove extra closing braces
                    extra_braces = close_count - open_count
                    # Find the position of the last valid closing brace
                    brace_positions = []
                    pos = last_open
                    brace_count = 0
                    for i, char in enumerate(raw_text[last_open:], last_open):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # This is the last valid closing brace
                                fixed_json = raw_text[last_open:i+1]
                                print(f"DEBUG: Attempting to fix JSON by removing extra characters after position {i+1}")
                                return json.loads(fixed_json)
        except Exception as fix_error:
            print(f"DEBUG: Failed to fix JSON: {fix_error}")
        
        raise ValueError(f"Failed to parse JSON: {e}")

# ------------------------
# CLI
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="*", help="User trip brief")
    ap.add_argument("--k", type=int, default=1, help="few-shot pairs per category")
    ap.add_argument("--device", type=str, default="cuda:0", help="cuda:0 | cuda:1 | cpu")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--hide-prompt", action="store_true", help="Hide prompt in logs")
    args = ap.parse_args()

    user_request = " ".join(args.request).strip() or (
        "4-day London trip for a single 32-year-old male, loves history and museums, medium budget, prefers walking and Tube."
    )

    t0 = _now()
    cats = route_category(user_request)
    log.info(f"[router] categories → {cats}")
    fewshots = select_examples(cats, k_per_cat=args.k)
    log.info(f"[router] selected examples → {[(m['id'], m['role']) for m in fewshots]}")
    t0 = _timed("Routing + example selection", t0)

    messages = build_messages(user_request, fewshots)
    t0 = _timed("Built messages", t0)

    if args.hide_prompt:
        logging.getLogger("commonsense").setLevel(logging.WARNING)

    raw = call_deepseek(messages, device_str=args.device, max_new_tokens=args.max_new_tokens)
    t0 = _timed("Model call", t0)

    print(raw)

if __name__ == "__main__":
    main()