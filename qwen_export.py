from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "Qwen/Qwen-VL-Chat"
adapter_path = "./saves/qwen-vl/qwen-vl_highavg" 
output_dir = "./models/qwen-vl/qwen-vl-merged-epoch3"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, device_map="auto", trust_remote_code=True
)

# Load adapter and merge
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()  # <-- merge LoRA into base model

# Save merged model in safetensors format
model.save_pretrained(output_dir, safe_serialization=True)

# Save tokenizer too
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)
