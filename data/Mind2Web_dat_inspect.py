import pyarrow.dataset as ds
import pandas as pd


dataset_path = "/mnt/welles/scratch/datasets/Magma-Mind2Web-SoM"

try:
    arrow_dataset = ds.dataset(dataset_path, format = "arrow")
    print("Dataset schema:")
    print(arrow_dataset.schema)

    first_file_path = sorted(list(Path(dataset_path).glob("data-*.arrow")))[0]
    table = pd.read_table((first_file_path))
    print("\nFirst few rows of the first Arrow file:")
    print(table.to_pandas().head())

except Exception as e:
    print(f"Could not directly load with pyarrow.dataset: {e}")
    print("Trying with Hugging Face datasets library...")
    from datasets import load_dataset
    try:
        hf_dataset = load_dataset(dataset_path, trust_remote_code=True)
        split_name = list(hf_dataset.keys())[0]
        print(f"\nLoaded with Hugging Face datasets. Schema for split: '{split_name}':")
        print(hf_dataset[split_name].features)
        print("n\First example:")
        print(hf_dataset[split_name][0])
    except Exception as e_hf:
        print(f"Error loading with Hugging Face datasets library: {e_hf}")
        print("Please ensure directory contains a valid Hugging Face dataset structure!")