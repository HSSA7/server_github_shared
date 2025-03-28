# ----------------------------------------------------------------------------
# gradio_hough2image.py (revised to force einops to use PyTorch backend)
# ----------------------------------------------------------------------------

import os
# Force einops to use torch before it looks for TF/Keras.
os.environ["EINOPS_BACKEND"] = "torch"

import random
import cv2
import einops

# If you have einops >= 0.4, you can do:
try:
    einops.set_backend("torch")
except AttributeError:
    # set_backend() was introduced in einops 0.4.0
    pass

import numpy as np
import torch

from share import *
import config

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.mlsd import MLSDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# Initialize the MLSD detector
apply_mlsd = MLSDdetector()

# Create and load the model
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_mlsd.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def run_inference(
    input_image_path,
    output_file,
    prompt="A futuristic cityscape",
    a_prompt="best quality, extremely detailed",
    n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
    num_samples=1,
    use_original_resolution=True,
    detect_resolution=512,
    ddim_steps=20,
    guess_mode=False,
    strength=1.0,
    scale=9.0,
    seed=1234,
    eta=0.0,
    value_threshold=0.1,
    distance_threshold=0.1
):
    # Read input image
    input_image = cv2.imread(input_image_path)

    if input_image is None:
        raise ValueError(f"Could not load image from {input_image_path}")
    input_image = HWC3(input_image)

    # Use the original image dimensions (rounded down to multiples of 8)
    if use_original_resolution:
        orig_h, orig_w = input_image.shape[:2]
        new_h = (orig_h // 8) * 8
        new_w = (orig_w // 8) * 8
        if (new_h, new_w) != (orig_h, orig_w):
            print(f"Adjusting image size from {orig_h}x{orig_w} to {new_h}x{new_w} for model compatibility.")
        img = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # Fallback: force 512x512
        img = resize_image(input_image, 512)

    H, W, C = img.shape

    # MLSD (Hough line) detection
    '''
    detected_map = apply_mlsd(
        resize_image(input_image, detect_resolution),
        value_threshold=value_threshold,
        distance_threshold=distance_threshold
    )
    '''
    
    detected_map = apply_mlsd(
        resize_image(input_image, detect_resolution),
        value_threshold,
        distance_threshold
    )
    
    # If MLSDdetector doesn’t accept thresholds at all:
    # detected_map = apply_mlsd(resize_image(input_image, detect_resolution))


    detected_map = HWC3(detected_map)
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

    # Convert to torch tensor
    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = control.permute(0, 3, 1, 2).clone()

    # Set seed
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    # Prepare conditioning
    cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
    }

    shape = (4, H // 8, W // 8)

    # Control strength
    if guess_mode:
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)]
    else:
        model.control_scales = [strength] * 13

    # Diffusion sampling
    samples, intermediates = ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond
    )

    # Decode and save images
    x_samples = model.decode_first_stage(samples)
    x_samples = (x_samples.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy()
    x_samples = x_samples.clip(0, 255).astype(np.uint8)

    for i, sample in enumerate(x_samples):
        out_path = output_file
        if num_samples > 1:
            out_path = f"{output_file.rsplit('.', 1)[0]}_{i}.png"
        cv2.imwrite(out_path, sample)
        print(f"Saved output to {out_path}")

if __name__ == "__main__":
    # Example usage
    run_inference(
        input_image_path="test_imgs/room_modified.png",
        output_file="output.png",
        prompt="room",
        a_prompt="best quality, extremely detailed",
        n_prompt="lowres, cropped, worst quality, low quality",
        num_samples=1,
        use_original_resolution=True,
        detect_resolution=512,
        ddim_steps=20,
        guess_mode=False,
        strength=1.0,
        scale=9.0,
        seed=1234,
        eta=0.0,
        value_threshold=0.1,
        distance_threshold=0.1
    )

