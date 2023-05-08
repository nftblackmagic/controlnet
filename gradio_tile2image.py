import cv2
import einops
import gradio as gr
import numpy as np
import torch

from cldm.hack import disable_verbosity
disable_verbosity()

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import apply_uniformer
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler

def process_tile(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, scale, seed, eta, denoise_strength, model, ddim_sampler):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = input_image.copy()

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0
        img = torch.stack([img for _ in range(num_samples)], dim=0)
        img = einops.rearrange(img, 'b h w c -> b c h w').clone()

        seed_everything(seed)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=False)
        t_enc = min(int(denoise_strength * ddim_steps), ddim_steps - 1)
        z = model.get_first_stage_encoding(model.encode_first_stage(img))
        z_enc = ddim_sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))

        samples = ddim_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results
