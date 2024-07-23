"""The goal of this script is to make sure every example in the data is at least longer than 1200 characters"""
from pathlib import Path
import jsonlines
from tqdm import tqdm

data_dir = Path('/scratch/sven/canine/pre_training_data/')

for jsonl_path in tqdm(sorted(data_dir.rglob('*.jsonl'), key=lambda x: x.stem)):
    # Create the corrected file
    if not jsonl_path.stem.endswith('_corrected'):
        jsonl_path.unlink()
    else:
        jsonl_path.rename(jsonl_path.parent / f'{jsonl_path.stem.replace("_corrected", "")}.jsonl')


#%%
