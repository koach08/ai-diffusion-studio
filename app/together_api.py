"""Together.ai API Client - Flux image generation with LoRA, NSFW configurable."""
import json
import os
import io
import urllib.request
import urllib.error

TOGETHER_API_URL = "https://api.together.ai/v1"

TOGETHER_MODELS = {
    "Flux.1 Schnell (高速・無料枠あり)": {
        "model": "black-forest-labs/FLUX.1-schnell-Free",
        "defaults": {"steps": 4, "width": 1024, "height": 1024},
        "cost": "無料",
        "nsfw_ok": False,  # Free tier has safety checker forced
    },
    "Flux.1 Schnell (高速)": {
        "model": "black-forest-labs/FLUX.1-schnell",
        "defaults": {"steps": 4, "width": 1024, "height": 1024},
        "cost": "~$0.003/枚",
        "nsfw_ok": True,
    },
    "Flux.1 Dev (高品質)": {
        "model": "black-forest-labs/FLUX.1-dev",
        "defaults": {"steps": 28, "width": 1024, "height": 1024},
        "cost": "~$0.025/枚",
        "nsfw_ok": True,
    },
    "Flux.1 Pro (最高品質)": {
        "model": "black-forest-labs/FLUX.1-pro",
        "defaults": {"steps": 28, "width": 1024, "height": 1024},
        "cost": "~$0.05/枚",
        "nsfw_ok": True,
    },
}


class TogetherClient:
    def __init__(self, api_key=""):
        self.api_key = api_key

    def set_key(self, api_key):
        self.api_key = api_key

    def _request(self, method, endpoint, data=None):
        if not self.api_key:
            raise ValueError("Together.ai API Key が設定されていません")
        url = f"{TOGETHER_API_URL}{endpoint}"
        body = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(
            url, data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Together API Error ({e.code}): {error_body}")

    def generate_image(self, model_key, prompt, negative_prompt="",
                       width=1024, height=1024, steps=None, seed=-1,
                       num_images=1, disable_safety=True):
        """Generate image via Together.ai. Returns list of base64 image data."""
        model_info = TOGETHER_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明なモデル: {model_key}")

        data = {
            "model": model_info["model"],
            "prompt": prompt,
            "width": width,
            "height": height,
            "n": num_images,
            "response_format": "b64_json",
        }
        if steps is not None:
            data["steps"] = steps
        elif "steps" in model_info["defaults"]:
            data["steps"] = model_info["defaults"]["steps"]
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        if seed >= 0:
            data["seed"] = seed
        if disable_safety and model_info.get("nsfw_ok", True):
            data["disable_safety_checker"] = True

        result = self._request("POST", "/images/generations", data)

        images = []
        for item in result.get("data", []):
            b64 = item.get("b64_json", "")
            if b64:
                images.append(b64)
        return images

    def check_api_key(self):
        try:
            self._request("GET", "/models")
            return True
        except Exception:
            return False


def decode_together_image(b64_data):
    """Decode base64 image data to PIL Image."""
    import base64
    from PIL import Image
    img_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(img_bytes))
