import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


if __name__ == "__main__":
    class DataTransformer:
        def __init__(self):
            print("Hello Started!")
            self.transform = transforms.Compose([
                transforms.PILToTensor()
            ])

        def transformer(self, x):
            output = dict()
            output["jpg"] = self.transform(x['edited_image'])
            output["hint"] = self.transform(x['original_image'])
            output["prompt"] = x['edit_prompt']

            return output


    # dataset = MyDataset()
    dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", split="train", streaming=True)
    # dataset = dataset.shuffle(buffer_size=10000, seed=42)
    piltransformer = DataTransformer()
    dataset = dataset.map(lambda x: piltransformer.transformer(x))
    for batch in dataset:
        print(batch)
        exit(0)
