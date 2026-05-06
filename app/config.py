"""AI-diffusion Studio Configuration"""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR, "app", "settings.json")

DEFAULT_CONFIG = {
    "comfyui_url": "http://127.0.0.1:8188",
    "backend": "local",
    "runpod_api_key": "",
    "runpod_pod_id": "",
    "runpod_comfyui_url": "",
    "civitai_api_key": "",
    "anthropic_api_key": "",
    "openai_api_key": "",
    "xai_api_key": "",
    "replicate_api_key": "",
    "fal_api_key": "",
    "together_api_key": "",
    "dezgo_api_key": "",
    "novita_api_key": "",
    "models_dir": os.path.join(BASE_DIR, "models"),
    "output_dir_normal": os.path.join(BASE_DIR, "outputs", "normal"),
    "output_dir_adult": os.path.join(BASE_DIR, "outputs", "adult"),
    "google_drive_models_dir": "",
    "icloud_models_dir": "",
    "default_model_normal": "",
    "default_model_adult": "",
    "default_negative_prompt": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, bad hands, extra fingers, missing fingers",
    "default_steps": 25,
    "default_cfg": 7.0,
    "default_width": 512,
    "default_height": 768,
    "default_sampler": "euler_ancestral",
    "default_scheduler": "normal",
}

SAMPLERS = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp",
    "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde",
    "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
    "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
    "ddpm", "lcm", "ipndm", "ipndm_v", "deis",
    "res_multistep", "res_multistep_cfg_pp",
    "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
    "gradient_estimation", "gradient_estimation_cfg_pp",
    "er_sde", "seeds_2", "seeds_3",
    "sa_solver", "sa_solver_pece",
    "ddim", "uni_pc", "uni_pc_bh2",
]

SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"]


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            saved = json.load(f)
        config = {**DEFAULT_CONFIG, **saved}
    else:
        config = DEFAULT_CONFIG.copy()
    os.makedirs(config["output_dir_normal"], exist_ok=True)
    os.makedirs(config["output_dir_adult"], exist_ok=True)
    return config


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _scan_dirs(subdirs, exts):
    """Scan multiple directories for files with given extensions."""
    found = []
    for d in subdirs:
        if d and os.path.isdir(d):
            found.extend([f for f in os.listdir(d) if f.lower().endswith(exts)])
    return sorted(set(found))


def _gdrive_dir():
    """Get Google Drive models dir from saved config."""
    cfg = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
    return cfg.get("google_drive_models_dir", "")


def _icloud_dir():
    """Get iCloud models dir from saved config."""
    cfg = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
    return cfg.get("icloud_models_dir", "")


def get_available_models(models_dir):
    """Scan checkpoints directory for available models (local + Google Drive + iCloud)."""
    gdrive = _gdrive_dir()
    icloud = _icloud_dir()
    dirs = [
        os.path.join(models_dir, "checkpoints"),
        os.path.join(gdrive, "checkpoints") if gdrive else "",
        # iCloud: scan both checkpoints/ subfolder and root (flat structure)
        os.path.join(icloud, "checkpoints") if icloud else "",
        icloud if icloud else "",
    ]
    return _scan_dirs(dirs, (".safetensors", ".ckpt", ".pt"))


def get_available_loras(models_dir):
    """Scan loras directory (local + Google Drive + iCloud)."""
    gdrive = _gdrive_dir()
    icloud = _icloud_dir()
    dirs = [
        os.path.join(models_dir, "loras"),
        os.path.join(gdrive, "loras") if gdrive else "",
        os.path.join(icloud, "loras") if icloud else "",
    ]
    return _scan_dirs(dirs, (".safetensors", ".ckpt", ".pt"))


def get_available_vaes(models_dir):
    """Scan VAE directory (local + Google Drive + iCloud)."""
    gdrive = _gdrive_dir()
    icloud = _icloud_dir()
    dirs = [
        os.path.join(models_dir, "vae"),
        os.path.join(gdrive, "vae") if gdrive else "",
        os.path.join(icloud, "vae") if icloud else "",
    ]
    return _scan_dirs(dirs, (".safetensors", ".ckpt", ".pt"))


def get_available_unet_models(models_dir):
    """Scan diffusion_models/unet directories for Flux and other UNET models."""
    dirs = [
        os.path.join(models_dir, "diffusion_models"),
        os.path.join(models_dir, "unet"),
        os.path.join(BASE_DIR, "comfyui", "models", "diffusion_models"),
        os.path.join(BASE_DIR, "comfyui", "models", "unet"),
    ]
    exts = (".safetensors", ".ckpt", ".pt", ".gguf")
    found = []
    for d in dirs:
        if os.path.isdir(d):
            found.extend([f for f in os.listdir(d) if f.lower().endswith(exts)])
    return sorted(set(found))


def get_available_clip_models(models_dir):
    """Scan clip/text_encoders directories."""
    dirs = [
        os.path.join(models_dir, "clip"),
        os.path.join(models_dir, "text_encoders"),
        os.path.join(BASE_DIR, "comfyui", "models", "clip"),
        os.path.join(BASE_DIR, "comfyui", "models", "text_encoders"),
    ]
    exts = (".safetensors", ".ckpt", ".pt", ".gguf", ".bin")
    found = []
    for d in dirs:
        if os.path.isdir(d):
            found.extend([f for f in os.listdir(d) if f.lower().endswith(exts)])
    return sorted(set(found))


def get_available_upscale_models(models_dir):
    """Scan upscale_models directory."""
    dirs = [
        os.path.join(models_dir, "upscale_models"),
        os.path.join(BASE_DIR, "comfyui", "models", "upscale_models"),
    ]
    exts = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")
    found = []
    for d in dirs:
        if os.path.isdir(d):
            found.extend([f for f in os.listdir(d) if f.lower().endswith(exts)])
    return sorted(set(found))


def get_icloud_only_models():
    """Get checkpoint models ONLY from iCloud directory (not local or Google Drive)."""
    icloud = _icloud_dir()
    if not icloud:
        return []
    dirs = [
        os.path.join(icloud, "checkpoints"),
        icloud,  # flat structure (files directly in iCloud root)
    ]
    return _scan_dirs(dirs, (".safetensors", ".ckpt", ".pt"))


def get_available_motion_models():
    """Scan AnimateDiff motion models from the custom node's models directory."""
    paths = [
        os.path.join(BASE_DIR, "comfyui", "custom_nodes", "ComfyUI-AnimateDiff-Evolved", "models"),
        os.path.join(BASE_DIR, "models", "animatediff_models"),
    ]
    exts = (".ckpt", ".safetensors", ".pt", ".pth")
    models = []
    for p in paths:
        if os.path.isdir(p):
            models.extend([f for f in os.listdir(p) if f.lower().endswith(exts)])
    return sorted(set(models))
