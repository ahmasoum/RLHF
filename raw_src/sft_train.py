
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)

try:
    from transformers import EarlyStoppingCallback
    HAS_EARLY_STOP = True
except Exception:
    HAS_EARLY_STOP = False

from trl import SFTTrainer

MODEL_NAME = "distilgpt2"
DATA_DIR = "data/processed"
OUTPUT_DIR = "outputs/sft"
TB_DIR = "outputs/logs/sft"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


ds = load_dataset(
    "json",
    data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "test":  f"{DATA_DIR}/test.jsonl",
    },
)

def add_completion(ex):
    ex["completion"] = ex["text"]
    return ex

def build_text_field(ex):
    p = ex.get("prompt", "")
    c = ex.get("completion", "")
    if not isinstance(p, str):
        p = ""
    if not isinstance(c, str):
        c = ""
    ex["text"] = (p + c).strip()
    return ex

ds = ds.map(add_completion, load_from_cache_file=False)
ds = ds.map(build_text_field, load_from_cache_file=False)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

writer = SummaryWriter(log_dir=TB_DIR)

def token_accuracy_from_texts(model, tokenizer, texts, device, max_seq_len=256):
    clean = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if len(clean) == 0:
        return 0.0

    model.eval()
    enc = tokenizer(
        clean,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    preds = logits[:, :-1].argmax(dim=-1)
    gold = labels[:, 1:]
    mask = attention_mask[:, 1:].bool()

    correct = (preds == gold) & mask
    denom = mask.sum().item()
    return float(correct.sum().item() / denom) if denom > 0 else 0.0


class AccuracyCallback(TrainerCallback):
    def __init__(self, writer, eval_dataset, tokenizer,
                 every_steps=5, max_samples=32, max_seq_len=256):
        self.writer = writer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.every_steps = every_steps
        self.max_samples = max_samples
        self.max_seq_len = max_seq_len

    def on_log(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        step = int(state.global_step)
        if step == 0 or (step % self.every_steps != 0):
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        n = min(self.max_samples, len(self.eval_dataset))
        batch = self.eval_dataset.select(range(n))
        texts = list(batch["text"])

        acc = token_accuracy_from_texts(model, self.tokenizer, texts, device, self.max_seq_len)

        self.writer.add_scalar("eval/token_accuracy", acc, step)
        self.writer.flush()
        print(f"[eval/token_accuracy] step={step} acc={acc:.4f}")


class EvalLossCallback(TrainerCallback):
    def __init__(self, writer):
        self.writer = writer

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        step = int(state.global_step)
        if "eval_loss" in metrics:
            loss = float(metrics["eval_loss"])
            self.writer.add_scalar("eval/eval_loss", loss, step)
            self.writer.flush()
            print(f"[eval/eval_loss] step={step} loss={loss:.4f}")


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=5e-5,

    fp16=torch.cuda.is_available(),

    logging_steps=5,
    logging_first_step=True,

    save_steps=500,
    save_total_limit=2,

    eval_strategy="steps",
    eval_steps=100,

    report_to="tensorboard",
    logging_dir=TB_DIR,
)

BEST_MODEL_ENABLED = False
try:
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "eval_loss"
    training_args.greater_is_better = False
    BEST_MODEL_ENABLED = True
except Exception:
    BEST_MODEL_ENABLED = False

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    dataset_text_field="text",
    max_seq_length=256,
)

trainer.add_callback(AccuracyCallback(writer, ds["test"], tokenizer, every_steps=5, max_samples=32, max_seq_len=256))
trainer.add_callback(EvalLossCallback(writer))

EARLY_STOP_ENABLED = False
if HAS_EARLY_STOP:
    try:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
        EARLY_STOP_ENABLED = True
    except Exception:
        EARLY_STOP_ENABLED = False

print("✅ Starting training...")
print(f"✅ Best-model-at-end enabled: {BEST_MODEL_ENABLED}")
print(f"✅ Early stopping enabled     : {EARLY_STOP_ENABLED}")

trainer.train()
print("✅ Training finished.")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
writer.close()

print("✅ Saved model to:", OUTPUT_DIR)
print("✅ TensorBoard logs:", TB_DIR)
