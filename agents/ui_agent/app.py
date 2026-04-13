# --------------------------------------------------------
# Magma - Multimodal AI Agent at Microsoft Research
# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import base64
import io
from typing import Optional

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from util.process_utils import extract_bbox, extract_mark_id, pred_2_point
from util.som import MarkHelper, plot_boxes_with_marks, plot_circles_with_marks
from util.utils import check_ocr_box, get_caption_model_processor, get_som_labeled_img, get_yolo_model

# Define repository and local directory
repo_id = "microsoft/OmniParser-v2.0"  # HF repo
local_dir = "weights"  # Target local directory
dtype = torch.bfloat16
DEVICE = torch.device('cuda')  

som_generator = MarkHelper()
magma_som_prompt = "<image>\nIn this view I need to click a button to \"{}\"? Provide the coordinates and the mark index of the containing bounding box if applicable."
magma_qa_prompt = "<image>\n{} Answer the question briefly."
magma_model_id = "microsoft/Magma-8B"
magam_model = AutoModelForCausalLM.from_pretrained(magma_model_id, trust_remote_code=True, torch_dtype=dtype)
magma_processor = AutoProcessor.from_pretrained(magma_model_id, trust_remote_code=True)
magam_model.to(DEVICE)

# Download the entire repository
snapshot_download(repo_id=repo_id, local_dir=local_dir)

print(f"Repository downloaded to: {local_dir}")


yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption")
# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2")

MARKDOWN = """
<div align="center">
<h2>Magma: A Foundation Model for Multimodal AI Agents</h2>

\[[arXiv Paper](https://www.arxiv.org/pdf/2502.13130)\] &nbsp; \[[Project Page](https://microsoft.github.io/Magma/)\] &nbsp; \[[Github Repo](https://github.com/microsoft/Magma)\] &nbsp; \[[Hugging Face Model](https://huggingface.co/microsoft/Magma-8B)\] &nbsp; 

This demo is powered by [Gradio](https://gradio.app/) and uses [OmniParserv2](https://github.com/microsoft/OmniParser) to generate [Set-of-Mark prompts](https://github.com/microsoft/SoM).

The demo supports three modes:
1. Empty text inut: it downgrades to an OmniParser demo.
2. Text input starting with "Q:": it leads to a visual question answering demo.
3. Text input for UI navigation: it leads to a UI navigation demo.
</div>
"""

DEVICE = torch.device('cuda')  

@spaces.GPU
@torch.inference_mode()
def get_som_response(instruction, image_som):
    prompt = magma_som_prompt.format(instruction)
    if magam_model.config.mm_use_image_start_end:
        qs = prompt.replace('<image>', '<image_start><image><image_end>')
    else:
        qs = prompt        
    convs = [{"role": "user", "content": qs}]
    convs = [{"role": "system", "content": "You are agent that can see, talk and act."}] + convs     
    prompt = magma_processor.tokenizer.apply_chat_template(
        convs,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = magma_processor(images=[image_som], texts=prompt, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
    inputs = inputs.to(dtype).to(DEVICE)

    magam_model.generation_config.pad_token_id = magma_processor.tokenizer.pad_token_id
    with torch.inference_mode():
        output_ids = magam_model.generate(
            **inputs, 
            temperature=0.0, 
            do_sample=False, 
            num_beams=1, 
            max_new_tokens=128, 
            use_cache=True
        )
    
    prompt_decoded = magma_processor.batch_decode(inputs['input_ids'], skip_special_tokens=True)[0]
    response = magma_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    response = response.replace(prompt_decoded, '').strip()
    return response

@spaces.GPU
@torch.inference_mode()
def get_qa_response(instruction, image):
    prompt = magma_qa_prompt.format(instruction)
    if magam_model.config.mm_use_image_start_end:
        qs = prompt.replace('<image>', '<image_start><image><image_end>')
    else:
        qs = prompt        
    convs = [{"role": "user", "content": qs}]
    convs = [{"role": "system", "content": "You are agent that can see, talk and act."}] + convs     
    prompt = magma_processor.tokenizer.apply_chat_template(
        convs,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = magma_processor(images=[image], texts=prompt, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
    inputs = inputs.to(dtype).to(DEVICE)

    magam_model.generation_config.pad_token_id = magma_processor.tokenizer.pad_token_id
    with torch.inference_mode():
        output_ids = magam_model.generate(
            **inputs, 
            temperature=0.0, 
            do_sample=False, 
            num_beams=1, 
            max_new_tokens=128, 
            use_cache=True
        )
    
    prompt_decoded = magma_processor.batch_decode(inputs['input_ids'], skip_special_tokens=True)[0]
    response = magma_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    response = response.replace(prompt_decoded, '').strip()
    return response

@spaces.GPU
@torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz, 
    instruction,
) -> Optional[Image.Image]:

    # image_save_path = 'imgs/saved_image_demo.png'
    # image_input.save(image_save_path)
    # image = Image.open(image_save_path)
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_input, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_input, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=False, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)  
    parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    
    if len(instruction) == 0:
        print('finish processing')
        image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        return image, str(parsed_content_list)

    elif instruction.startswith('Q:'):
        response = get_qa_response(instruction, image_input)
        return image_input, response

    # parsed_content_list = str(parsed_content_list)
    # convert xywh to yxhw
    label_coordinates_yxhw = {}
    for key, val in label_coordinates.items():
        if val[2] < 0 or val[3] < 0:
            continue
        label_coordinates_yxhw[key] = [val[1], val[0], val[3], val[2]]
    image_som = plot_boxes_with_marks(image_input.copy(), [val for key, val in label_coordinates_yxhw.items()], som_generator, edgecolor=(255,0,0), fn_save=None, normalized_to_pixel=False)

    # convert xywh to xyxy
    for key, val in label_coordinates.items():
        label_coordinates[key] = [val[0], val[1], val[0] + val[2], val[1] + val[3]]

    # normalize label_coordinates
    for key, val in label_coordinates.items():
        label_coordinates[key] = [val[0] / image_input.size[0], val[1] / image_input.size[1], val[2] / image_input.size[0], val[3] / image_input.size[1]]
    
    magma_response = get_som_response(instruction, image_som)
    print("magma repsonse: ", magma_response)

    # map magma_response into the mark id
    mark_id = extract_mark_id(magma_response)
    if mark_id is not None:
        if str(mark_id) in label_coordinates:
            bbox_for_mark = label_coordinates[str(mark_id)]
        else:
            bbox_for_mark = None
    else:
        bbox_for_mark = None
    
    if bbox_for_mark:
        # draw bbox_for_mark on the image
        image_som = plot_boxes_with_marks(
            image_input, 
            [label_coordinates_yxhw[str(mark_id)]], 
            som_generator, 
            edgecolor=(255,127,111), 
            alpha=30, 
            fn_save=None, 
            normalized_to_pixel=False,
            add_mark=False
        )
    else:
        try:
            if 'box' in magma_response:
                pred_bbox = extract_bbox(magma_response)
                click_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2, (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
                click_point = [item / 1000 for item in click_point]
            else:
                click_point = pred_2_point(magma_response)
            # de-normalize click_point (width, height)
            click_point = [click_point[0] * image_input.size[0], click_point[1] * image_input.size[1]]

            image_som = plot_circles_with_marks(
                image_input, 
                [click_point],
                som_generator,
                edgecolor=(255,127,111), 
                linewidth=3,
                fn_save=None,
                normalized_to_pixel=False,
                add_mark=False
            )
        except:
            image_som = image_input
    
    return image_som, str(parsed_content_list)

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(
                type='pil', label='Upload image')
            # set the threshold for removing the bounding boxes with low confidence, default is 0.05
            with gr.Accordion("Parameters", open=False) as parameter_row:            
                box_threshold_component = gr.Slider(
                    label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05)
                # set the threshold for removing the bounding boxes with large overlap, default is 0.1
                iou_threshold_component = gr.Slider(
                    label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
                use_paddleocr_component = gr.Checkbox(
                    label='Use PaddleOCR', value=True)
                imgsz_component = gr.Slider(
                    label='Icon Detect Image Size', minimum=640, maximum=1920, step=32, value=640)
            # text box
            text_input_component = gr.Textbox(label='Text Input', placeholder='Text Input')
            submit_button_component = gr.Button(
                value='Submit', variant='primary')
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output')

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            use_paddleocr_component,
            imgsz_component, 
            text_input_component
        ],
        outputs=[image_output_component, text_output_component]
    )

# demo.launch(debug=False, show_error=True, share=True)
# demo.launch(share=True, server_port=7861, server_name='0.0.0.0')
demo.queue().launch(share=False)