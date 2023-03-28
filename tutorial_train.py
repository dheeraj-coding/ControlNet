from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from datasets import load_dataset
from torchvision import transforms

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
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
dataset = dataset.shuffle(buffer_size=10000, seed=42)
piltransformer = DataTransformer()
dataset = dataset.map(lambda x: piltransformer.transformer(x))
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=2, precision=32, callbacks=[logger])

# Train!
trainer.fit(model, dataloader)
