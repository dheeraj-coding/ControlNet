from share import *
import torch
from einops import rearrange

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
model = create_model('./models/cldm_v15_copy.yaml').cpu()
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
        output["txt"] = x['edit_prompt']

        output['jpg'] = rearrange(output['jpg'], 'c h w -> h w c')
        output['hint'] = rearrange(output['hint'], 'c h w -> h w c')

        output['jpg'] = (output['jpg'].type(torch.float32) / 127.5) - 1.0
        output['hint'] = output['hint'].type(torch.float32) / 255.0

        return output


# dataset = MyDataset()
dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", split="train", streaming=True)
dataset = dataset.shuffle(buffer_size=10000, seed=42)
piltransformer = DataTransformer()
dataset = dataset.map(lambda x: piltransformer.transformer(x))
dataset = dataset.remove_columns(["edited_image", "original_prompt", "original_image", "edit_prompt", "edited_prompt"])
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size)
logger = ImageLogger(batch_frequency=logger_freq,
                     log_images_kwargs={"sample": True, "unconditional_guidance_scale": 1.0, "N": 1})
trainer = pl.Trainer(accelerator="gpu", devices=1, precision=32, callbacks=[logger])

# Train!
trainer.fit(model, dataloader)
