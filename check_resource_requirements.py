from transformers import AutoTokenizer, AutoModelForCausalLM
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

# get a rough estimate of GPU reequirements
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
print(estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1))