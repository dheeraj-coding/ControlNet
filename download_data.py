from datasets import load_dataset
from torch.utils.data import DataLoader
import pickle

dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", split="train", streaming=True)
dataset = dataset.map(lambda x: {'edit_prompt': x['edit_prompt']},
                      remove_columns=['edited_image', 'original_prompt', 'original_image', 'edited_prompt'])
loader = DataLoader(dataset=dataset, batch_size=16, num_workers=0)

with open('prompts.txt', 'ab+') as f:
    for batch in loader:
        pickle.dump(batch['edit_prompt'], f)
