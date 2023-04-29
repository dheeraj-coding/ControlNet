from __future__ import annotations
from pathlib import Path
import requests
from io import BytesIO
import torch
import os
from diffusers import DiffusionPipeline, DDIMScheduler
import math
import random
import sys
from argparse import ArgumentParser
from cldm.cldm import ControlLDM
from share import *
import config
import cv2
import einops
import gradio as gr
from pytorch_lightning import seed_everything
from torchvision import transforms
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange
from PIL import Image, ImageOps
import json
import matplotlib.pyplot as plt
import seaborn
from fastai.basics import show_image, show_images
import clip
from datasets import load_dataset
from fastcore.parallel import Self
from predict import process

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1)


class ClipSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load("RN50", device=device, jit=False)
        self.model.eval().requires_grad_(False)

    def encode_text(self, text):
        text = clip.tokenize(text).to(device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image):
        image = self.preprocess(image).unsqueeze(0).to(device)
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features

    def forward(self, input_image, output_image, input_text, output_text):
        input_image_features = self.encode_image(input_image)
        output_image_features = self.encode_image(output_image)
        input_text_features = self.encode_text(input_text)
        output_text_features = self.encode_text(output_text)
        sim_0 = F.cosine_similarity(input_image_features, input_text_features)
        sim_1 = F.cosine_similarity(output_image_features, output_text_features)
        sim_direction = F.cosine_similarity(output_image_features - input_image_features,
                                            output_text_features - input_text_features)
        sim_image = F.cosine_similarity(input_image_features, output_image_features)
        return sim_0, sim_1, sim_direction, sim_image


def download_image(url):
    image = Image.open(requests.get(url, stream=True).raw)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


generator = torch.Generator("cuda").manual_seed(0)
seed = 0

dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", split="train",
                       streaming=True)  # will start loading the data when iterated over
dataset = dataset.take(10_000)

output_json_instruct = []
output_json_imagic = []

clip_similarity = ClipSimilarity().cuda()

output_json = []
num_sample = 1
sim_direction_avg_instruct = []
sim_image_avg_instruct = []
sim_direction_avg_imagic = []
sim_image_avg_imagic = []
stop_count = 25

data_transformer = transforms.Compose([
    transforms.PILToTensor()
])
to_transformer = transforms.ToPILImage()

for index, data in enumerate(dataset):
    if index > -1:
        out = []
        img = data['original_image']
        input_caption = data['original_prompt']
        output_caption = data['edited_prompt']
        prompt = data['edit_prompt']

        out.append(img)

        print("input caption :", input_caption, "output caption: ", output_caption)
        print(index, "- prompt: ", prompt)

        # ret_value = instruct_edit(img, prompt)
        img = data_transformer(img)
        img = rearrange(img, 'c h w -> h w c')
        img = img.type(torch.float32) / 255.0
        # ret_value = process(img, prompt, "", "", 1, 512, 20, False, 1.0, 9.0, seed, 0.0, 100, 200)
        ret_value = process(img, prompt, '', '', 1, 512, 20, False, 1, 9.0, seed, 0.0)
        ret_value = ret_value[0]
        ret_value = rearrange(ret_value, 'h w c -> c h w')
        ret_value = to_transformer(ret_value)
        img = rearrange(img, 'h w c -> c h w')
        img = to_transformer(img)

        if ret_value is not None:
            out.append(ret_value)
            image_0, image_1 = img, ret_value
            _, _, sim_direction_instruct, sim_image_instruct = clip_similarity(image_0, image_1, input_caption[:77],
                                                                               output_caption[:77])
            sim_direction_instruct = sim_direction_instruct.item()
            sim_image_instruct = sim_image_instruct.item()

            if not np.isnan(sim_direction_instruct) and not np.isnan(sim_image_instruct):
                sim_direction_avg_instruct.append(sim_direction_instruct)
                sim_image_avg_instruct.append(sim_image_instruct)

                if (index and index % num_sample == 0):
                    output_json_instruct.append({'sim_direction': np.array(sim_direction_avg_instruct).mean(),
                                                 'sim_image': np.array(sim_image_avg_instruct).mean()})

                    sim_direction_avg_instruct = []
                    sim_image_avg_instruct = []

    if index == stop_count:
        break

output_json_instruct = sorted(output_json_instruct, key=lambda k: k['sim_direction'])
x = [d["sim_direction"] for d in output_json_instruct]
y = [d["sim_image"] for d in output_json_instruct]

plt.rcParams.update({'font.size': 11.5})
seaborn.set_style("darkgrid")
plt.figure(figsize=(20.5 * 0.7, 10.8 * 0.7), dpi=200)

plt.plot(x, y, linewidth=2, markersize=4)

plt.xlabel("CLIP Text-Image Direction Similarity", labelpad=10)
plt.ylabel("CLIP Image Similarity", labelpad=10)

plt.savefig(Path("./") / Path("plot.pdf"), bbox_inches="tight")

# averaged results
x_avg = np.array(x)
n = len(x_avg)
n = (n // 3) * 3
x_avg = x_avg[:n]
x_avg = np.mean(np.array(x_avg).reshape(-1, 3), axis=1)

y_avg = np.array(y)
y_avg = y_avg[:n]
y_avg = np.mean(np.array(y_avg).reshape(-1, 3), axis=1)

# instructPix2Pix
plt.plot(x_avg, y_avg)

plt.xlabel("CLIP Text-Image Direction Similarity", labelpad=10)
plt.ylabel("CLIP Image Similarity", labelpad=10)

plt.savefig(Path("./") / Path("plot.pdf"), bbox_inches="tight")

np.savez('controlnet.npz', x=x, y=y, x_avg=x_avg, y_avg=y_avg)
