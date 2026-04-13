"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import random

import imageio
import numpy as np
import tensorflow as tf
import torch
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img

def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_obs(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    image = Image.fromarray(img)
    # resize image to 256x256
    image = resize_image(img, resize_size)
    return image

def get_max_steps(task_suite_name):
    if task_suite_name == "libero_spatial":
        max_steps = 220  
    elif task_suite_name == "libero_object":
        max_steps = 280 
    elif task_suite_name == "libero_goal":
        max_steps = 300  
    elif task_suite_name == "libero_10":
        max_steps = 520  
    else:
        max_steps = 400
    return max_steps

def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def save_rollout_video(replay_images, success, task_description):
    """Saves a video replay of a rollout in libero."""
    save_dir = "./libero_videos"
    os.makedirs(save_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    video_path = f"{save_dir}/quick_eval-success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(video_path, fps=30)
    for img in replay_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved libero video at path {video_path}")
    return video_path

def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)