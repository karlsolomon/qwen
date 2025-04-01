from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer, pipeline

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model in 4-bit on CUDA:1
model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    device="cuda:0",
    trust_remote_code=True,
    use_safetensors=True,
    inject_fused_attention=True,  # Optional performance boost
    revision="main",  # Or "gptq-4bit-128g" if specific
)

# Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=1)

# Inference
prompt = "def quicksort(arr):"
output = pipe(prompt, max_new_tokens=256, do_sample=False)
print(output[0]["generated_text"])
