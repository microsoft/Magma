# --------------------------------------------------------
# Magma - Multimodal AI Agent at Microsoft Research
# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import ast
import random
from typing import TYPE_CHECKING

import gradio as gr
import torch
import torchvision
from transformers import AutoModelForCausalLM, AutoProcessor
from utils.visualizer import Visualizer

'''
build model
'''
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

spatial_quant_size = 256

# Load AI Model
dtype = torch.bfloat16
device = "cuda"
magma_model_id = "microsoft/Magma-8B"
model = AutoModelForCausalLM.from_pretrained(magma_model_id, trust_remote_code=True, torch_dtype=dtype)
processor = AutoProcessor.from_pretrained(magma_model_id, trust_remote_code=True)
model.to(device)

@torch.no_grad()
def inference(image, task, *args, **kwargs):
    # image = image['image']
    task_description = task
    num_marks = args[0]
    speed = args[1]
    steps = args[2]
    mark_ids = [i+1 for i in range(num_marks)]  

    image_resized = image.resize((256, 256))

    magma_template = (
        # "<image>\nThe image is labeled with numeric marks {}.\n"
        "<image>\nThe image is split into 256x256 grids and is labeled with numeric marks {}.\n"
        "The robot is doing: {}. To finish the task, how to move the numerical marks in the image with speed {} for the next {} steps?\n"
    )

    """
    Visual Trace Generation
    """
    if model.config.mm_use_image_start_end:
        magma_template = magma_template.replace("<image>", "<image_start><image><image_end>")    
    conv_user = magma_template.format(mark_ids, task_description, speed, steps)
    print(conv_user)

    convs = [
        {"role": "user", "content": conv_user},
    ]
    convs = [
        {
            "role": "system",
            "content": "You are agent that can see, talk and act.", 
        },            
    ] + convs     

    prompt = processor.tokenizer.apply_chat_template(
        convs,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(images=image_resized, texts=prompt, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)    
    inputs = inputs.to(dtype).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            temperature=0.3,
            do_sample=True,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
        )
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    if len(response)==0:
        return None
    # extract traces from response
    if "and their future positions are:" in response:
        selected_marks_str, traces_str = response.split("and their future positions are:\n")
    else:
        selected_marks_str, traces_str = None, response

    try:
        traces_dict = ast.literal_eval('{' + traces_str.strip().replace('\n\n',',') + '}')
        overlay_traces = []
        for mark_id, trace in traces_dict.items():
            # convert list of tuples to tensor
            trace = torch.tensor(ast.literal_eval(trace)).unsqueeze(1)
            overlay_traces.append(trace)
        # padded to the same length with the last element
        max_len = max([trace.shape[0] for trace in overlay_traces])
        for i in range(len(overlay_traces)):
            if overlay_traces[i].shape[0] < max_len:
                overlay_traces[i] = torch.cat([overlay_traces[i], overlay_traces[i][-1].unsqueeze(0).repeat(max_len - overlay_traces[i].shape[0], 1, 1)], dim=0)        
        overlay_traces = torch.cat(overlay_traces, dim=1).unsqueeze(0)
        # if selected_marks_str is not None:
        #     selected_marks = re.findall(r'\[(.*?)\]', selected_marks_str)
        #     selected_marks = [torch.tensor(ast.literal_eval(mark)).unsqueeze(0) for mark in selected_marks]
        #     selected_marks = torch.cat(selected_marks, dim=0).unsqueeze(0)        
        #     overlay_traces = torch.cat([selected_marks.unsqueeze(1), overlay_traces], dim=1)
        overlay_traces = overlay_traces.float() / 256
        overlay_traces[:,:,:,0] = overlay_traces[:,:,:,0] * image.size[0]
        overlay_traces[:,:,:,1] = overlay_traces[:,:,:,1] * image.size[1]
        images = [image] * overlay_traces.shape[1]
        overlay_visibility = overlay_traces.new(overlay_traces.shape[0], overlay_traces.shape[1], overlay_traces.shape[2]).fill_(True)
        video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
        vis = Visualizer(save_dir="./saved_videos", pad_value=0, linewidth=2, tracks_leave_trace=-1)
        vis.visualize(video, overlay_traces, overlay_visibility)
        # return video path
        return "./saved_videos/video.mp4"
    except Exception as e:
        print(e)
        return None

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)
    from typing import Literal, Sequence

    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer

class Video(gr.components.Video):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)
    from typing import Literal, Sequence

    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer


'''
launch app
'''
title = "Magma"

description = '''Magma: Multimodal Agent to Act'''
'''Usage
Instructions:
&#x1F388 Try our default examples first (Sketch is not automatically drawed on input and example image);
&#x1F388 For video demo, it takes about 30-60s to process, please refresh if you meet an error on uploading;
&#x1F388 Upload an image/video (If you want to use referred region of another image please check "Example" and upload another image in referring image panel);
&#x1F388 Select at least one type of prompt of your choice (If you want to use referred region of another image please check "Example");
&#x1F388 Remember to provide the actual prompt for each promt type you select, otherwise you will meet an error (e.g., rember to draw on the referring image);
&#x1F388 Our model by default support the vocabulary of COCO 133 categories, others will be classified to 'others' or misclassifed.
'''

article = "The Demo is Run on Magma-8B."
inputs = [
    gr.components.Image(label="Draw on Image",type="pil"), 
    gr.Textbox(label="Task"),
    gr.Slider(1, 50, value=10, label="Number of Marks", info="Choose between 1 and 50"),
    gr.Slider(2, 50, value=8, label="Speed", info="Choose between 2 and 50"),
    gr.Slider(2, 50, value=8, label="Steps", info="Choose between 2 and 50"),
]
gr.Interface(
    fn=inference,
    inputs=inputs,
    outputs=[
        gr.Video(
        label="Robot planning trajectory", format="mp4"
        ),
    ],
    examples=[
    ["agents/robot_traj/sample.png", "Pick up the chip bag.", 9, 8, 8],
    ],
    title=title,
    description=description,
    article=article,
    allow_flagging='never',
    cache_examples=False,
).launch(share=True)