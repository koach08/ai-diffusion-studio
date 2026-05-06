"""Dezgo API Client - Uncensored image & video generation."""
import json
import io
import urllib.request
import urllib.error

DEZGO_API_URL = "https://api.dezgo.com"

DEZGO_IMAGE_MODELS = {
    "Flux Dev (高品質)": {
        "endpoint": "/text2image_flux",
        "model": "flux1_dev",
        "defaults": {"steps": 28, "guidance": 3.5, "width": 1024, "height": 1024},
        "cost": "~$0.02/枚",
    },
    "Flux Schnell (高速)": {
        "endpoint": "/text2image_flux",
        "model": "flux1_schnell",
        "defaults": {"steps": 4, "width": 1024, "height": 1024},
        "cost": "~$0.003/枚",
    },
    "SDXL Lightning (高速・高品質)": {
        "endpoint": "/text2image_sdxl_lightning",
        "model": None,
        "defaults": {"width": 1024, "height": 1024},
        "cost": "~$0.003/枚",
    },
    "Juggernaut XL (リアル系)": {
        "endpoint": "/text2image_sdxl",
        "model": "juggernaut_xl_v9",
        "defaults": {"steps": 25, "guidance": 7, "width": 1024, "height": 1024},
        "cost": "~$0.008/枚",
    },
    "DreamShaper XL (万能)": {
        "endpoint": "/text2image_sdxl",
        "model": "dreamshaperxl_v21",
        "defaults": {"steps": 25, "guidance": 7, "width": 1024, "height": 1024},
        "cost": "~$0.008/枚",
    },
    "Realistic Vision (フォトリアル)": {
        "endpoint": "/text2image",
        "model": "realistic_vision_5",
        "defaults": {"steps": 30, "guidance": 7, "width": 512, "height": 768},
        "cost": "~$0.003/枚",
    },
}

DEZGO_VIDEO_MODELS = {
    "Wan 2.6 (最新・高品質動画)": {
        "endpoint": "/text2video_wan_2_6",
        "defaults": {"steps": 30, "width": 832, "height": 480},
        "cost": "~$0.10/本",
    },
    "Wan 2.1 (安定版動画)": {
        "endpoint": "/text2video_wan_2_1",
        "defaults": {"steps": 30, "width": 832, "height": 480},
        "cost": "~$0.10/本",
    },
}


class DezgoClient:
    def __init__(self, api_key=""):
        self.api_key = api_key

    def set_key(self, api_key):
        self.api_key = api_key

    def generate_image(self, model_key, prompt, negative_prompt="",
                       width=1024, height=1024, steps=None, guidance=None,
                       seed=-1):
        """Generate image. Returns raw PNG bytes."""
        if not self.api_key:
            raise ValueError("Dezgo API Key が設定されていません")

        model_info = DEZGO_IMAGE_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明なモデル: {model_key}")

        endpoint = model_info["endpoint"]
        defaults = model_info["defaults"]

        data = {"prompt": prompt}
        if model_info.get("model"):
            data["model"] = model_info["model"]
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        data["width"] = width or defaults.get("width", 1024)
        data["height"] = height or defaults.get("height", 1024)
        if steps is not None:
            data["steps"] = steps
        elif "steps" in defaults:
            data["steps"] = defaults["steps"]
        if guidance is not None:
            data["guidance"] = guidance
        elif "guidance" in defaults:
            data["guidance"] = defaults["guidance"]
        if seed >= 0:
            data["seed"] = seed

        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{DEZGO_API_URL}{endpoint}",
            data=body,
            headers={
                "X-Dezgo-Key": self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            return resp.read()  # raw PNG bytes
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Dezgo API Error ({e.code}): {error_body}")

    def img2img(self, prompt, image_path, negative_prompt="",
                strength=0.7, model="realistic_vision_5", steps=30, guidance=7,
                width=512, height=768, seed=-1):
        """img2img via Dezgo. Fully uncensored. Returns raw PNG bytes."""
        if not self.api_key:
            raise ValueError("Dezgo API Key が設定されていません")

        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        data = {
            "prompt": prompt,
            "init_image": f"data:image/png;base64,{img_b64}",
            "model": model,
            "strength": strength,
            "steps": steps,
            "guidance": guidance,
        }
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        if seed >= 0:
            data["seed"] = seed

        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{DEZGO_API_URL}/image2image",
            data=body,
            headers={
                "X-Dezgo-Key": self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            return resp.read()  # raw PNG bytes
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Dezgo img2img Error ({e.code}): {error_body}")

    def generate_video(self, model_key, prompt, negative_prompt="",
                       width=832, height=480, steps=None, seed=-1):
        """Generate video. Returns raw MP4 bytes."""
        if not self.api_key:
            raise ValueError("Dezgo API Key が設定されていません")

        model_info = DEZGO_VIDEO_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明な動画モデル: {model_key}")

        data = {"prompt": prompt}
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        data["width"] = width
        data["height"] = height
        if steps is not None:
            data["steps"] = steps
        if seed >= 0:
            data["seed"] = seed

        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{DEZGO_API_URL}{model_info['endpoint']}",
            data=body,
            headers={
                "X-Dezgo-Key": self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            return resp.read()  # raw MP4 bytes
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Dezgo Video API Error ({e.code}): {error_body}")

    def check_api_key(self):
        try:
            # Try a minimal generation
            self.generate_image(
                "Flux Schnell (高速)", "test", width=256, height=256, steps=1
            )
            return True
        except Exception:
            return False


def decode_dezgo_image(png_bytes):
    """Convert raw PNG bytes to PIL Image."""
    from PIL import Image
    return Image.open(io.BytesIO(png_bytes))
