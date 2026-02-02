import random
import warnings
warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from torch.utils.tensorboard import SummaryWriter
from trl import SFTTrainer


MODEL_NAME = "distilgpt2"
DATA_DIR = "data/processed_filtered"
OUTPUT_DIR = "outputs/sft_filtered"
LOG_DIR = "outputs/logs/sft_filtered"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

ACC_LOG_EVERY = 5
EVAL_EVERY = 100

ACC_EVAL_SAMPLES = 64
ACC_BATCH_SIZE = 8
MAX_SEQ_LEN = 256

tb = SummaryWriter(log_dir=LOG_DIR)


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

ds = ds.map(add_completion)

def formatting_func(ex):
    if isinstance(ex["prompt"], list):
        return [p + c for p, c in zip(ex["prompt"], ex["completion"])]
    return ex["prompt"] + ex["completion"]

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


@torch.no_grad()
def token_accuracy_cpu(model, tokenizer, texts, max_length=MAX_SEQ_LEN, batch_size=8):
    if not texts:
        return 0.0

    model_cpu = model.to("cpu")
    model_cpu.eval()

    total_correct = 0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        out = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

        pred = logits[:, :-1, :].argmax(dim=-1)
        gold = input_ids[:, 1:]
        mask = attention_mask[:, 1:]

        total_correct += ((pred == gold) * mask).sum().item()
        total_tokens += mask.sum().item()

    return float(total_correct) / float(total_tokens) if total_tokens > 0 else 0.0


class TokenAccuracyCallback(TrainerCallback):
    def __init__(self, eval_ds, tokenizer, tb_writer, every_steps=5, n_samples=64, batch_size=8):
        self.eval_ds = eval_ds
        self.tokenizer = tokenizer
        self.tb = tb_writer
        self.every_steps = every_steps
        self.n_samples = n_samples
        self.batch_size = batch_size

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step is None or state.global_step == 0:
            return
        if state.global_step % self.every_steps != 0:
            return

        model = kwargs.get("model", None)
        if model is None:
            return

        n = min(self.n_samples, len(self.eval_ds))
        idxs = random.sample(range(len(self.eval_ds)), k=n)

        texts = []
        for i in idxs:
            ex = self.eval_ds[i]
            p = ex.get("prompt", "")
            c = ex.get("completion", "")
            if isinstance(p, str) and isinstance(c, str):
                texts.append(p + c)

        acc = token_accuracy_cpu(
            model=model,
            tokenizer=self.tokenizer,
            texts=texts,
            max_length=MAX_SEQ_LEN,
            batch_size=self.batch_size,
        )

        if torch.cuda.is_available():
            model.to("cuda")
            torch.cuda.empty_cache()

        print(f"[eval/token_accuracy] step={state.global_step} acc={acc:.4f}")

        self.tb.add_scalar("eval/token_accuracy", acc, state.global_step)
        self.tb.flush()


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),

    logging_steps=50,
    save_steps=EVAL_EVERY,
    save_total_limit=2,

    eval_strategy="steps",
    eval_steps=EVAL_EVERY,

    report_to="tensorboard",
    logging_dir=LOG_DIR,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    formatting_func=formatting_func,
    max_seq_length=MAX_SEQ_LEN,
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
trainer.add_callback(
    TokenAccuracyCallback(
        eval_ds=ds["test"],
        tokenizer=tokenizer,
        tb_writer=tb,
        every_steps=ACC_LOG_EVERY,
        n_samples=ACC_EVAL_SAMPLES,
        batch_size=ACC_BATCH_SIZE,
    )
)

print(" Starting SFT (filtered) ...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

tb.close()
print(" SFT (filtered) saved to:", OUTPUT_DIR)
print(" TensorBoard logs:", LOG_DIR)
