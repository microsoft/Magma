import os
#os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"

from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor 

# Set device consistently
device = torch.device("cuda:1")
dtype = torch.bfloat16

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Magma-8B", trust_remote_code=True, torch_dtype=dtype)
processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)

# Move model to device
model = model.to(device)

# Inference
image = Image.open("./assets/images/magma_logo.jpg").convert("RGB")

convs = [
    {"role": "system", "content": "You are agent that can see, talk and act."},            
    {"role": "user", "content": "<image_start><image><image_end>\nWhat is the letter on the robot?"},
]
prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)

# Process inputs and move to device properly
inputs = processor(images=[image], texts=prompt, return_tensors="pt")

# Move all tensors to the same device
for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        inputs[key] = value.to(device)

# Handle nested tensors properly
if 'pixel_values' in inputs:
    if inputs['pixel_values'].dim() == 3:  # Add batch dimension if needed
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['pixel_values'] = inputs['pixel_values'].to(device).to(dtype)

# The image_sizes tensor needs to be properly formatted for Magma
# It should be a 3D tensor with shape [batch_size, num_images, 2]
if 'image_sizes' in inputs:
    # Get original dimensions
    if inputs['image_sizes'].dim() == 1:
        # If it's just [height, width]
        height, width = inputs['image_sizes'].tolist()
        # Reshape to [1, 1, 2] - batch_size=1, num_images=1, dimensions=2
        inputs['image_sizes'] = torch.tensor([[[height, width]]], device=device)

# Convert all tensor inputs to the specified dtype
for key in inputs:
    if isinstance(inputs[key], torch.Tensor) and inputs[key].dtype == torch.float32:
        inputs[key] = inputs[key].to(dtype)

generation_args = { 
    "max_new_tokens": 500, 
    "temperature": 0.0, 
    "do_sample": False, 
    "use_cache": True,
    "num_beams": 1,
} 

with torch.inference_mode():
    generate_ids = model.generate(**inputs, **generation_args)

generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

print(response)