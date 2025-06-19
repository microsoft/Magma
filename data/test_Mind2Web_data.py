import yaml
from Magma.data.dataset import build_joint_dataset
from Magma.magma.processing_magma import MagmaProcessor
from Magma.train import DataArguments

MODEL_PATH = "/home/oberon/projects/Magma/Magma-8B"
processor = MagmaProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)


data_args = DataArguments(
    data_path = "/home/oberon/projects/Magma/data_configs/mind2web_som_finetune.yaml"
    mm_use_som_tom = True,
    
)