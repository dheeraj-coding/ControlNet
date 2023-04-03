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

model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('/home1/dheeraj/ControlNet/lightning_logs/version_14208795/checkpoints/epoch=1-step=99999.ckpt', location='cuda'))
model = model.cuda()
model = model.eval()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength,
            scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        # detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(input_image)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


for index, data in enumerate(dataset):
    if index > 9:
        out = []
        img = data['original_image']
        input_caption = data['original_prompt']
        output_caption = data['edited_prompt']
        prompt = data['edit_prompt']

        out.append(img)

        print("input caption :", input_caption, "output caption: ", output_caption)
        print(index, "- prompt: ", prompt)

        # ret_value = instruct_edit(img, prompt)
        print(type(img))
        ret_value = process(img, prompt, "", "", 1, 512, 20, False, 1.0, 9.0, seed, 0.0, 100, 200)

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
# plt.plot(x2, y2,  linewidth=2, markersize=4)

plt.xlabel("CLIP Text-Image Direction Similarity", labelpad=10)
plt.ylabel("CLIP Image Similarity", labelpad=10)

plt.savefig(Path("./") / Path("plot.pdf"), bbox_inches="tight")

# averaged results
x_avg = np.array(x)
x_avg = np.delete(x_avg, -1)
x_avg = np.mean(np.array(x_avg).reshape(-1, 3), axis=1)

y_avg = np.array(y)
y_avg = np.delete(y_avg, -1)
y_avg = np.mean(np.array(y_avg).reshape(-1, 3), axis=1)

# instructPix2Pix
plt.plot(x_avg, y_avg)

plt.xlabel("CLIP Text-Image Direction Similarity", labelpad=10)
plt.ylabel("CLIP Image Similarity", labelpad=10)

plt.savefig(Path("./") / Path("plot.pdf"), bbox_inches="tight")
