import os
os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"

from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor 
from magma.processing_magma import MagmaProcessor
from magma.modeling_magma import MagmaForCausalLM

#device = torch.device('cuda:0')
dtype = torch.float32
model = MagmaForCausalLM.from_pretrained("microsoft/Magma-8B", trust_remote_code=True, torch_dtype=dtype)
processor = MagmaProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
model.to('cuda:0')

# Inference
# image = Image.open("./assets/images/unreal_temple.jpg").convert("RGB")
image = Image.open("./assets/images/magma_logo.jpg").convert("RGB")

convs = [
    {"role": "system", "content": "You are agent that can see, talk and act."},            
    {"role": "user", "content": "<image_start><image><image_end>\nWhat is the letter on the robot's body?."},
]
prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
inputs = processor(images=[image], texts=prompt, return_tensors="pt")

inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
#inputs['pixel_values'] = inputs['pixel_values']
#inputs['image_sizes'] = inputs['image_sizes']

inputs = inputs.to("cuda:0").to(dtype)

generation_args = { 
    "max_new_tokens": 500, 
    "temperature": 0.0, 
    "do_sample": False, 
    "use_cache": True,
    "num_beams": 1,
} 

print("--- Input Details ---")
for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: shape ={value.shape}, data type ={value.dtype}, device = {value.device}")
    else:
        print(f"{key}: {value}")
print("------------------------")

with torch.inference_mode():
    generate_ids = model.generate(**inputs, **generation_args)

generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

print(response)
