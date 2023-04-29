from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import sys
import random
from PIL import Image

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from einops import rearrange

apply_canny = CannyDetector()

model = create_model('./models/cldm_v15_copy.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model = model.cuda()
model.eval()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength,
            scale, seed, eta):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
            # seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uncond = 0.05
        random = torch.rand(control.size(0), device=control.device)
        input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")
        c_concat = [
            input_mask * model.encode_first_stage(control).mode().detach()]

        cond = {"c_concat": c_concat, "c_edited": control, "c_prompt": [prompt + ', ' + a_prompt] * num_samples,
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)],
                "gt_imgs": control, "c_control": [control]}
        un_cond = {"c_concat": None if guess_mode else c_concat, "c_edited": control,
                   "c_prompt": [prompt + ', ' + a_prompt] * num_samples,
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
                   "gt_imgs": control, "c_control": [control]}
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
    return results


if __name__ == "__main__":
    img_path = sys.argv[1]
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.open(img_path)
    prompt = sys.argv[2]
    a_prompt = ''
    n_prompt = ''
    num_samples = 1
    image_resolution = 512
    ddim_steps = 20
    guess_mode = False
    strength = 1
    scale = 9.0
    seed = random.randint(-1, 2147483647)
    eta = 0.0
    process(np.asarray(img), prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode,
            strength,
            scale, seed, eta)
