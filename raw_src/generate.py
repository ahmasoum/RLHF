from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "outputs/ppo" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

prompt = "Write a short engaging tweet about science and technology:\n"
inputs = tokenizer(prompt, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_p=0.9, temperature=1.0)
print(tokenizer.decode(out[0], skip_special_tokens=True))
