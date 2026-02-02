import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def gen(model_dir, prompts, max_new_tokens=40):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    outs = []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        outs.append(text)
    return outs

prompts = [
    "Write a short engaging tweet about science and technology:\n",
    "Write a short engaging tweet about AI safety:\n",
    "Write a short engaging tweet about space exploration:\n",
]

sft = gen("outputs/sft", prompts)
ppo = gen("outputs/ppo", prompts)

print("\n" + "="*80)
for i, p in enumerate(prompts):
    print(f"\nPROMPT {i}: {p.strip()}")
    print("-"*80)
    print("SFT:\n", sft[i])
    print("-"*80)
    print("PPO:\n", ppo[i])
    print("="*80)
