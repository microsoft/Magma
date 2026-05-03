"""
Magma-8B model + OmniParser YOLO loading and inference.
Singleton pattern – models are loaded once and reused across requests.
All parameters match the reference notebook.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from ultralytics import YOLO

from src.domains.task.som import build_som_candidates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching notebook exactly)
# ---------------------------------------------------------------------------
BASE_MODEL = "microsoft/Magma-8B"
MODEL_IMAGE_SIZE = 768

USE_MOCK_MODEL = os.environ.get("USE_MOCK_MODEL", "0").lower() in ("1", "true", "yes", "on")
MOCK_MODEL_ACTION = os.environ.get("MOCK_MODEL_ACTION", "CLICK").upper()
MOCK_MODEL_VALUE = os.environ.get("MOCK_MODEL_VALUE")
MOCK_MODEL_X = os.environ.get("MOCK_MODEL_X")
MOCK_MODEL_Y = os.environ.get("MOCK_MODEL_Y")
MOCK_MODEL_TERMINATE_AFTER_CLICK = os.environ.get(
    "MOCK_MODEL_TERMINATE_AFTER_CLICK", "1"
).lower() in ("1", "true", "yes", "on")

# Checkpoint – will be downloaded from HuggingFace or Google Drive on first run.
# Override via env var MAGMA_CHECKPOINT_DIR if you have a local copy.

CHECKPOINT_LOCAL_DIR = Path(
    os.environ.get(
        "MAGMA_CHECKPOINT_DIR",
        str(
            Path(__file__).resolve().parents[5]
            / "checkpoints"
            / "finetune-3apps-r32-a64-maxlen2560-focal-marks-5actions"
            / "checkpoint-3300"
        ),
    )
)

# OmniParser YOLO weights
WEIGHTS_DIR = Path(
    os.environ.get(
        "OMNIPARSER_WEIGHTS_DIR",
        str(Path(__file__).resolve().parents[3] / "weights"),
    )
)
OMNIPARSER_MODEL_PATH = WEIGHTS_DIR / "icon_detect" / "model.pt"

# LoRA adapter repo on HuggingFace Hub. Set MAGMA_LORA_REPO to "<user>/<repo>".
# If unset, the checkpoint must already exist at MAGMA_CHECKPOINT_DIR.
LORA_REPO_ID = os.environ.get("MAGMA_LORA_REPO", "")

INSTRUCTION_TEMPLATE = (
    "Imagine that you are imitating humans doing GUI navigation step by step.\n\n"
    "You can perform actions such as CLICK, DOUBLE_CLICK, RIGHT_CLICK, MIDDLE_CLICK, "
    "MOVE, DRAG, SCROLL, HSCROLL, TYPE, PRESS, HOTKEY, TERMINATE.\n\n"
    "Output format must be:\n"
    '{{"ACTION": action_type, "MARK": numeric_id, "VALUE": text_or_null}}\n\n'
    "Task: {task_prompt}\n\n"
    "Previous actions:\n{previous_actions}\n\n"
    "For your convenience, UI elements are labeled with numeric marks.\n\n"
    "What is the next action?\n\n"
    "The following mark positions and labels are approximate references to help you "
    "identify elements. Always verify by checking the image.\n"
    "{marks_text}\n"
)

# ---------------------------------------------------------------------------
# torch.sum bool-tensor compatibility patch (from notebook)
# ---------------------------------------------------------------------------
if not hasattr(torch, "_original_sum_backup"):
    torch._original_sum_backup = torch.sum

    def _patched_sum(input, *args, **kwargs):
        if isinstance(input, bool):
            input = torch.tensor(input, dtype=torch.long)
        elif (
            isinstance(input, torch.Tensor)
            and input.dtype == torch.bool
            and (len(args) > 0 or "dim" in kwargs)
        ):
            input = input.long()
        return torch._original_sum_backup(input, *args, **kwargs)

    torch.sum = _patched_sum
    logger.info("Applied torch.sum bool-tensor compatibility patch")


# ---------------------------------------------------------------------------
# Safe weight init patch (from notebook)
# ---------------------------------------------------------------------------
def _safe_init_weights(self, module):
    std = (
        self.config.initializer_range
        if hasattr(self.config, "initializer_range")
        else self.config.text_config.initializer_range
    )
    if hasattr(module, "class_embedding"):
        if module.class_embedding.data.is_floating_point():
            module.class_embedding.data.normal_(mean=0.0, std=std)
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        weight = getattr(module, "weight", None)
        if weight is not None and weight.data.is_floating_point():
            weight.data.normal_(mean=0.0, std=std)
        bias = getattr(module, "bias", None)
        if bias is not None and bias.data.is_floating_point():
            bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        weight = getattr(module, "weight", None)
        if weight is not None and weight.data.is_floating_point():
            weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                weight.data[module.padding_idx].zero_()


# ---------------------------------------------------------------------------
# Singleton holders
# ---------------------------------------------------------------------------
_magma_model = None
_magma_processor = None
_yolo_model = None
_device: str = "cpu"


# ---------------------------------------------------------------------------
# OmniParser YOLO loading
# ---------------------------------------------------------------------------
def _download_omniparser_weights() -> None:
    if OMNIPARSER_MODEL_PATH.exists():
        return
    logger.info("Downloading OmniParser-v2.0 weights from HuggingFace …")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="microsoft/OmniParser-v2.0",
        local_dir=str(WEIGHTS_DIR),
        local_dir_use_symlinks=False,
    )


def load_yolo() -> YOLO:
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    _download_omniparser_weights()
    if not OMNIPARSER_MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO weights not found at: {OMNIPARSER_MODEL_PATH}")
    logger.info(f"Loading YOLO model from {OMNIPARSER_MODEL_PATH}")
    _yolo_model = YOLO(str(OMNIPARSER_MODEL_PATH))
    return _yolo_model


# ---------------------------------------------------------------------------
# Magma model loading
# ---------------------------------------------------------------------------
def _has_valid_checkpoint(path: Path) -> bool:
    has_config = (path / "adapter_config.json").exists()
    has_weights = (path / "adapter_model.safetensors").exists() or (
        path / "adapter_model.bin"
    ).exists()
    return has_config and has_weights


def _ensure_checkpoint() -> None:
    """Download the LoRA adapter from HuggingFace Hub if not already present."""
    if _has_valid_checkpoint(CHECKPOINT_LOCAL_DIR):
        logger.info(f"Using cached checkpoint in {CHECKPOINT_LOCAL_DIR}")
        return

    if not LORA_REPO_ID:
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_LOCAL_DIR} and MAGMA_LORA_REPO is not set. "
            "Set MAGMA_LORA_REPO=<user>/<repo> to pull the LoRA adapter from HuggingFace Hub, "
            "or point MAGMA_CHECKPOINT_DIR at a local adapter directory."
        )

    logger.info(
        f"Checkpoint not found at {CHECKPOINT_LOCAL_DIR}. "
        f"Downloading from HuggingFace Hub: {LORA_REPO_ID}"
    )
    CHECKPOINT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=LORA_REPO_ID, local_dir=str(CHECKPOINT_LOCAL_DIR))

    if not _has_valid_checkpoint(CHECKPOINT_LOCAL_DIR):
        raise RuntimeError(
            f"Downloaded checkpoint from {LORA_REPO_ID} is incomplete at {CHECKPOINT_LOCAL_DIR} "
            "(missing adapter_config.json or adapter_model.safetensors)."
        )
    logger.info(f"Checkpoint downloaded successfully from {LORA_REPO_ID}")


def load_magma():
    global _magma_model, _magma_processor, _device

    if _magma_model is not None:
        return _magma_model, _magma_processor

    # Download checkpoint from Drive if missing
    _ensure_checkpoint()

    logger.info(f"Loading Magma base model: {BASE_MODEL}")

    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if hasattr(config, "auto_map") and "AutoModelForCausalLM" in config.auto_map:
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModelForCausalLM"], BASE_MODEL
        )
    else:
        model_class = AutoModelForCausalLM._model_mapping[type(config)]

    # Patch init weights
    model_class._init_weights = _safe_init_weights

    if torch.cuda.is_available():
        _device = "cuda"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": {"": 0},
            "attn_implementation": "eager",
            "quantization_config": quantization_config,
        }
    else:
        _device = "cpu"
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32,
            "attn_implementation": "eager",
        }

    try:
        model = model_class.from_pretrained(BASE_MODEL, **load_kwargs)
    except Exception as exc:
        msg = str(exc)
        if "frozenset" in msg or "bitsandbytes" in msg.lower():
            logger.warning("4-bit load failed; retrying without quantization_config")
            fallback_kwargs = {
                "trust_remote_code": True,
                "attn_implementation": "eager",
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            }
            if torch.cuda.is_available():
                fallback_kwargs["device_map"] = {"": 0}
            model = model_class.from_pretrained(BASE_MODEL, **fallback_kwargs)
        else:
            raise

    if _device == "cpu":
        model.to("cpu")

    logger.info(f"Loading LoRA adapter from {CHECKPOINT_LOCAL_DIR}")
    model = PeftModel.from_pretrained(model, str(CHECKPOINT_LOCAL_DIR))

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if hasattr(processor, "image_processor") and hasattr(
        processor.image_processor, "base_img_size"
    ):
        processor.image_processor.base_img_size = MODEL_IMAGE_SIZE
        logger.info(f"Set processor.image_processor.base_img_size={MODEL_IMAGE_SIZE}")
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    _magma_model = model
    _magma_processor = processor
    logger.info("Magma model loaded successfully")
    return _magma_model, _magma_processor


# ---------------------------------------------------------------------------
# Warm-up: call at server startup
# ---------------------------------------------------------------------------
def warmup() -> None:
    if USE_MOCK_MODEL:
        logger.info("USE_MOCK_MODEL is enabled; skipping model warmup")
        return
    load_yolo()
    load_magma()
    logger.info("All models warmed up and ready")


# ---------------------------------------------------------------------------
# Action parsing (from notebook)
# ---------------------------------------------------------------------------
def parse_action(text: str) -> dict:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except Exception:
            pass

    return {"raw_response": text, "parse_error": True}


def _random_mock_coordinate(axis_size: int) -> int:
    if axis_size <= 1:
        return 0
    low = max(0, int(axis_size * 0.1))
    high = min(axis_size - 1, int(axis_size * 0.9))
    if low > high:
        low, high = 0, axis_size - 1
    return random.randint(low, high)


# ---------------------------------------------------------------------------
# Inference: screenshot + prompt → predicted action with coordinates
# ---------------------------------------------------------------------------
def infer(image: Image.Image, task_prompt: str, previous_actions: str = "None") -> dict:
    """
    Run full SoM + Magma inference pipeline.

    Returns dict with keys:
        action   – "CLICK" | "TYPE" | "DOUBLE_CLICK" | "RIGHT_CLICK"
        x, y     – pixel coordinates of the predicted target (or None)
        value    – text value for TYPE (or None)
        mark_id  – numeric mark index predicted by model (or None)
        raw_response – raw model text output
    """
    if USE_MOCK_MODEL:
        logger.info("USE_MOCK_MODEL is enabled; returning mocked inference response")
        action = MOCK_MODEL_ACTION
        if (
            action == "CLICK"
            and MOCK_MODEL_TERMINATE_AFTER_CLICK
            and previous_actions.strip().lower() != "none"
        ):
            action = "TERMINATE"

        if action == "CLICK":
            x = (
                int(MOCK_MODEL_X)
                if MOCK_MODEL_X is not None
                else _random_mock_coordinate(image.width)
            )
            y = (
                int(MOCK_MODEL_Y)
                if MOCK_MODEL_Y is not None
                else _random_mock_coordinate(image.height)
            )
        else:
            x = int(MOCK_MODEL_X) if MOCK_MODEL_X is not None else None
            y = int(MOCK_MODEL_Y) if MOCK_MODEL_Y is not None else None

        raw_response = json.dumps(
            {
                "ACTION": action,
                "MARK": None,
                "VALUE": MOCK_MODEL_VALUE,
                "mock": True,
                "task": task_prompt,
                "previous_actions": previous_actions,
            }
        )
        return {
            "action": action,
            "x": x,
            "y": y,
            "value": MOCK_MODEL_VALUE,
            "mark_id": None,
            "raw_response": raw_response,
            "som_image": image.copy(),
        }

    model, processor = load_magma()
    yolo = load_yolo()

    # 1. SoM detection + annotation (YOLO + OCR, matching training)
    som_annotated_image, candidate_bboxes, marks_text = build_som_candidates(image, yolo)

    # 2. Build prompt (matching training format with mark coordinates + OCR text)
    instruction = INSTRUCTION_TEMPLATE.format(
        task_prompt=task_prompt,
        previous_actions=previous_actions,
        marks_text=marks_text,
    )
    full_prompt = f"<image_start><image><image_end>\n{instruction}"

    convs = [
        {"role": "system", "content": "You are agent that can see, talk and act."},
        {"role": "user", "content": full_prompt},
    ]
    formatted_prompt = processor.tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )

    if hasattr(model, "config") and getattr(model.config, "mm_use_image_start_end", False):
        formatted_prompt = formatted_prompt.replace("<image>", "<image_start><image><image_end>")

    logger.info("=" * 60)
    logger.info("INFERENCE PROMPT:")
    logger.info(instruction)
    logger.info(f"Num marks: {len(candidate_bboxes)}")
    logger.info("=" * 60)

    # 3. Tokenize + run inference
    inputs = processor(images=[som_annotated_image], texts=formatted_prompt, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
    inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)

    if _device == "cuda":
        inputs = inputs.to(torch.bfloat16).to("cuda")
    else:
        inputs = inputs.to("cpu")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=2,
            max_new_tokens=256,
            use_cache=True,
        )

    generate_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
    response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

    logger.info(f"MODEL RAW RESPONSE: {response}")

    # 4. Parse action
    prediction = parse_action(response)
    predicted_action = str(prediction.get("ACTION", "")).upper()
    mark_id = prediction.get("MARK")
    value = prediction.get("VALUE")

    coordinate = None
    if mark_id is not None and mark_id in candidate_bboxes:
        x1, y1, x2, y2 = candidate_bboxes[mark_id]
        coordinate = {"x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)}

    return {
        "action": predicted_action,
        "x": coordinate["x"] if coordinate else None,
        "y": coordinate["y"] if coordinate else None,
        "value": value if value and str(value).lower() != "none" else None,
        "mark_id": mark_id,
        "raw_response": response,
        "som_image": som_annotated_image,
    }
