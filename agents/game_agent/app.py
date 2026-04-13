# --------------------------------------------------------
# Magma - Multimodal AI Agent at Microsoft Research
# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import random
import re

import gradio as gr
import numpy as np
import pygame
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

pygame.mixer.quit()  # Disable sound

# Constants
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 80
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)  # Forest green - more like an apple
RED = (200, 50, 50)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
YELLOW = (218, 165, 32)  # Golden yellow color

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
STATIC = (0, 0)

ACTIONS = ["up", "down", "left", "right", "static"]

# Load AI Model
magma_model_id = "microsoft/Magma-8B"

dtype = torch.bfloat16
magma_model = AutoModelForCausalLM.from_pretrained(magma_model_id, trust_remote_code=True, torch_dtype=dtype)
magma_processor = AutoProcessor.from_pretrained(magma_model_id, trust_remote_code=True, torch_dtype=dtype)
magma_model.to("cuda")

# Load magma image
magma_img = pygame.image.load("./assets/images/magma_game.png")
magma_img = pygame.transform.scale(magma_img, (GRID_SIZE, GRID_SIZE))

class MagmaFindGPU:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.snake = [(5, 5)]
        self.direction = RIGHT
        self.score = 0
        self.game_over = False
        self.place_target()

    def place_target(self):
        while True:
            target_x = np.random.randint(1, WIDTH // GRID_SIZE - 1)
            target_y = np.random.randint(1, HEIGHT // GRID_SIZE - 1)
            if (target_x, target_y) not in self.snake:
                self.target = (target_x, target_y)
                break

    def step(self, action):
        if action == "up":
            self.direction = UP
        elif action == "down":
            self.direction = DOWN
        elif action == "left":
            self.direction = LEFT
        elif action == "right":
            self.direction = RIGHT
        elif action == "static":
            self.direction = STATIC
        
        if self.game_over:
            return self.render(), self.score
        
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        
        if new_head[0] < 0 or new_head[1] < 0 or new_head[0] >= WIDTH // GRID_SIZE or new_head[1] >= HEIGHT // GRID_SIZE:
            self.game_over = True
            return self.render(), self.score
        
        self.snake = [new_head]  # Keep only the head (single block snake)
        
        # Check if the target is covered by four surrounding squares
        head_x, head_y = self.snake[0]
        neighbors = set([(head_x, head_y - 1), (head_x, head_y + 1), (head_x - 1, head_y), (head_x + 1, head_y)])
        
        if neighbors.issuperset(set([self.target])):
            self.score += 1
            self.place_target()

        return self.render(), self.score
    
    def render(self):
        pygame.init()
        surface = pygame.Surface((WIDTH, HEIGHT))
        surface.fill(BLACK)
        
        head_x, head_y = self.snake[0]
        surface.blit(magma_img, (head_x * GRID_SIZE, head_y * GRID_SIZE))        
        
        # pygame.draw.rect(surface, RED, (self.snake[0][0] * GRID_SIZE, self.snake[0][1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(surface, GREEN, (self.target[0] * GRID_SIZE, self.target[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # Draw four surrounding squares with labels
        head_x, head_y = self.snake[0]
        neighbors = [(head_x, head_y - 1), (head_x, head_y + 1), (head_x - 1, head_y), (head_x + 1, head_y)]
        labels = ["1", "2", "3", "4"]
        font = pygame.font.Font(None, 48)
        
        # clone surface
        surface_nomark = surface.copy()
        for i, (nx, ny) in enumerate(neighbors):
            if 0 <= nx < WIDTH // GRID_SIZE and 0 <= ny < HEIGHT // GRID_SIZE:
                pygame.draw.rect(surface, RED, (nx * GRID_SIZE, ny * GRID_SIZE, GRID_SIZE, GRID_SIZE), GRID_SIZE)
                # pygame.draw.rect(surface_nomark, RED, (nx * GRID_SIZE, ny * GRID_SIZE, GRID_SIZE, GRID_SIZE), GRID_SIZE)

                text = font.render(labels[i], True, WHITE)
                text_rect = text.get_rect(center=(nx * GRID_SIZE + GRID_SIZE // 2, ny * GRID_SIZE + GRID_SIZE // 2))
                surface.blit(text, text_rect)
        
        return np.array(pygame.surfarray.array3d(surface_nomark)).swapaxes(0, 1), np.array(pygame.surfarray.array3d(surface)).swapaxes(0, 1)
    
    def get_state(self):
        return self.render()

game = MagmaFindGPU()

def play_game():
    state, state_som = game.get_state()
    pil_img = Image.fromarray(state_som)
    convs = [
        {"role": "system", "content": "You are an agent that can see, talk, and act."},            
        {"role": "user", "content": "<image_start><image><image_end>\nWhich mark is closer to green block? Answer with a single number."},
    ]
    prompt = magma_processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    inputs = magma_processor(images=[pil_img], texts=prompt, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)    
    inputs = inputs.to("cuda").to(dtype)
    generation_args = { 
        "max_new_tokens": 10, 
        "temperature": 0, 
        "do_sample": False, 
        "use_cache": True,
        "num_beams": 1,
    }

    with torch.inference_mode():
        generate_ids = magma_model.generate(**inputs, **generation_args)
    generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
    action = magma_processor.decode(generate_ids[0], skip_special_tokens=True).strip()
    # extract mark id fro action use re
    match = re.search(r'\d+', action)
    if match:
        action = match.group(0)
        if action.isdigit() and 1 <= int(action) <= 4:
            # epsilon sampling
            if random.random() < 0.1:
                action = random.choice(ACTIONS[:-1])
            else:
                action = ACTIONS[int(action) - 1]
        else:
            # random choose one from the pool
            action = random.choice(ACTIONS[:-1])
    else:
        action = random.choice(ACTIONS[:-1])

    img, score = game.step(action)
    img = img[0]
    return img, f"Score: {score}"

def reset_game():
    game.reset()
    return game.render()[0], "Score: 0"

MARKDOWN = """
<div align="center">
<h2>Magma: A Foundation Model for Multimodal AI Agents</h2>

Game: Magma finds the apple by moving up, down, left and right.

\[[arXiv Paper](https://www.arxiv.org/pdf/2502.13130)\] &nbsp; \[[Project Page](https://microsoft.github.io/Magma/)\] &nbsp; \[[Github Repo](https://github.com/microsoft/Magma)\] &nbsp; \[[Hugging Face Model](https://huggingface.co/microsoft/Magma-8B)\] &nbsp; 

This demo is powered by [Gradio](https://gradio.app/).
</div>
"""

with gr.Blocks() as interface:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        image_output = gr.Image(label="Game Screen")
        score_output = gr.Text(label="Score")
    with gr.Row():
        start_btn = gr.Button("Start/Reset Game")

    interface.load(fn=play_game, every=1, inputs=[], outputs=[image_output, score_output])
    start_btn.click(fn=reset_game, inputs=[], outputs=[image_output, score_output])

interface.launch()
