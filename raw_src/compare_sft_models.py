import os
import random
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_A_DIR = "outputs/sft"           
MODEL_B_DIR = "outputs/sft_filtered"   

OUT_PATH = Path("demo/compare_sft.txt")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEN_KWARGS = dict(
    max_new_tokens=64,
    do_sample=True,
    top_p=0.9,
    temperature=1.0,
)

BASE_SEED = 42
random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(BASE_SEED)

PROMPTS = [
    "Write a short engaging tweet about science and technology:\n",
    "Write a short engaging tweet about AI safety:\n",
    "Write a short engaging tweet about space exploration:\n",
    "Write a short engaging tweet about renewable energy:\n",
    "Write a short engaging tweet about cybersecurity:\n",
]

N_SAMPLES_PER_PROMPT = 3  


def load_model(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return tok, model

print("Loading models...")
tok_a, model_a = load_model(MODEL_A_DIR)
tok_b, model_b = load_model(MODEL_B_DIR)
print("✅ Models loaded on:", DEVICE)

@torch.no_grad()
def generate_text(tokenizer, model, prompt: str, seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(DEVICE)
    out = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **GEN_KWARGS)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


lines = []
lines.append("=== SFT MODEL COMPARISON ===\n")
lines.append(f"MODEL_A (raw SFT)     : {MODEL_A_DIR}\n")
lines.append(f"MODEL_B (filtered SFT): {MODEL_B_DIR}\n")
lines.append(f"DEVICE: {DEVICE}\n")
lines.append("-" * 80 + "\n\n")

for pi, prompt in enumerate(PROMPTS):
    lines.append("=" * 80 + "\n")
    lines.append(f"PROMPT {pi}: {prompt.strip()}\n")
    lines.append("-" * 80 + "\n")

    for si in range(N_SAMPLES_PER_PROMPT):
        seed = BASE_SEED + pi * 100 + si

        out_a = generate_text(tok_a, model_a, prompt, seed=seed)
        out_b = generate_text(tok_b, model_b, prompt, seed=seed)

        lines.append(f"\n[SAMPLE {si}] seed={seed}\n")
        lines.append("RAW SFT:\n")
        lines.append(out_a + "\n")
        lines.append("-" * 80 + "\n")
        lines.append("FILTERED SFT:\n")
        lines.append(out_b + "\n")
        lines.append("\n")

print(" Comparison done. Writing:", OUT_PATH)

OUT_PATH.write_text("".join(lines), encoding="utf-8")
print(" Saved:", OUT_PATH.resolve())
