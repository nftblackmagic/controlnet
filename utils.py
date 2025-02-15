import os
from subprocess import call

model_dl_urls = {
    "canny": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
    "depth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
    "softedge": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth",
    "normal": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth",
    "mlsd": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd.pth",
    "openpose": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
    "scribble": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth",
    "seg": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth",
    "tile": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth",
}

annotator_dl_urls = {
    "body_pose_model.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth",
    "dpt_hybrid-midas-501f0c75.pt": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt",
    "hand_pose_model.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth",
    "mlsd_large_512_fp32.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth",
    "mlsd_tiny_512_fp32.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_tiny_512_fp32.pth",
    "network-bsds500.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth",
    "upernet_global_small.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth",
}

def download_base():
    """
    Download base model from huggingface with wget and save to models directory
    """
    base_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt"
    relative_path_to_base = base_url.replace("https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/", "./models/")

    if not os.path.exists(relative_path_to_base):
        print(f"Downloading base model...")
        call(["wget", "-O", relative_path_to_base, base_url])

def download_annotator(model_name, urls_map):
    """
    Download model from huggingface with wget and save to models directory
    """
    model_url = urls_map[model_name]
    relative_path_to_model = model_url.replace("https://huggingface.co/lllyasviel/ControlNet/resolve/main/", "")
    if not os.path.exists(relative_path_to_model):
        print(f"Downloading {model_name}...")
        call(["wget", "-O", relative_path_to_model, model_url])

def download_model(model_name, urls_map):
    """
    Download model from huggingface with wget and save to models directory
    """
    model_url = urls_map[model_name]
    relative_path_to_model = model_url.replace("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/", "./models/")
    if not os.path.exists(relative_path_to_model):
        print(f"Downloading {model_name}...")
        call(["wget", "-O", relative_path_to_model, model_url])

def get_state_dict_path(model_name):
    """
    Get path to model state dict
    """
    return f"./models/control_v11p_sd15_{model_name}.pth"
