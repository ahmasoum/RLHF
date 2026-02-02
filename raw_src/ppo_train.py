import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.tensorboard import SummaryWriter
import warnings
import math

warnings.filterwarnings("ignore")


SFT_DIR = "outputs/sft_filtered"
DATA_DIR = "data/processed_filtered"
OUT_DIR = "outputs/ppo_filtered"
LOG_DIR = "outputs/logs/ppo_filtered"


writer = SummaryWriter(log_dir=LOG_DIR)


tokenizer = AutoTokenizer.from_pretrained(SFT_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


ds = load_dataset("json", data_files={"train": f"{DATA_DIR}/train.jsonl"})["train"]

def clip_reward(r, lo=0.0, hi=6.0):
    try:
        r = float(r)
    except Exception:
        r = 0.0
    return max(lo, min(hi, r))

def shaped_reward(r):
    r = clip_reward(r)
    return math.tanh(r / 2.0) 


BATCH_SIZE = 8
MINI_BATCH_SIZE = 2
NUM_SAMPLES = 2000 

config = PPOConfig(
    learning_rate=2e-6,   
    batch_size=BATCH_SIZE,
    mini_batch_size=MINI_BATCH_SIZE,
    gradient_accumulation_steps=1,
)


model = AutoModelForCausalLMWithValueHead.from_pretrained(SFT_DIR)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(SFT_DIR)

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
device = ppo_trainer.accelerator.device
print("PPO device:", device)


gen_kwargs = dict(
    max_new_tokens=12,
    do_sample=True,
    top_k=0,
    top_p=0.9,
    temperature=1.0,
    pad_token_id=tokenizer.eos_token_id,
)


queries_buf = []
responses_buf = []
rewards_buf = []

global_step = 0

data_iter = iter(ds.shuffle(seed=42).select(range(NUM_SAMPLES)))

while True:
    try:
        ex = next(data_iter)
    except StopIteration:
        break

    prompt = ex.get("prompt", "")
    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
        continue

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids_2d = enc["input_ids"].to(device)        
    attn_mask_2d = enc["attention_mask"].to(device)   
    q_len = input_ids_2d.shape[-1]

    with torch.no_grad():
        full_ids_2d = model.generate(
            input_ids=input_ids_2d,
            attention_mask=attn_mask_2d,
            **gen_kwargs
        )  

    query_1d = input_ids_2d[0]                
    response_1d = full_ids_2d[0][q_len:]   

    if response_1d.numel() < 1:
        continue

    r = shaped_reward(ex.get("reward", 0.0))  

    queries_buf.append(query_1d)
    responses_buf.append(response_1d)
    rewards_buf.append(r)

    if len(queries_buf) == BATCH_SIZE:
        scores = [torch.tensor(rv, device=device) for rv in rewards_buf]

        stats = ppo_trainer.step(queries_buf, responses_buf, scores)

        reward_mean = sum(rewards_buf) / len(rewards_buf)
        writer.add_scalar("ppo/reward_mean", float(reward_mean), global_step)
        writer.add_scalar("ppo/kl", float(stats.get("objective/kl", 0.0)), global_step)
        writer.add_scalar("ppo/entropy", float(stats.get("objective/entropy", 0.0)), global_step)

        for k in ["ppo/loss/policy", "ppo/loss/value", "ppo/returns/mean", "ppo/advantages/mean"]:
            if k in stats:
                try:
                    writer.add_scalar(k, float(stats[k]), global_step)
                except Exception:
                    pass

        if global_step % 10 == 0:
            gen_text = tokenizer.decode(responses_buf[0], skip_special_tokens=True)
            print(
                f"ppo_step={global_step} reward_mean={reward_mean:.3f} "
                f"kl={stats.get('objective/kl', 0.0):.4f} | gen='{gen_text[:80]}'"
            )

        queries_buf, responses_buf, rewards_buf = [], [], []
        global_step += 1

writer.close()

ppo_trainer.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("✅ PPO saved to:", OUT_DIR)
print("✅ PPO TensorBoard logs in:", LOG_DIR)
