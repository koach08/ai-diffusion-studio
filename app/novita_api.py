"""Novita.ai API Client - Uncensored image generation with many models."""
import json
import io
import time
import urllib.request
import urllib.error

NOVITA_API_URL = "https://api.novita.ai/v3"

NOVITA_MODELS = {
    "Flux Dev (高品質)": {
        "model": "flux-dev",
        "defaults": {"steps": 28, "guidance_scale": 3.5, "width": 1024, "height": 1024},
        "cost": "~$0.02/枚",
    },
    "Flux Schnell (高速)": {
        "model": "flux-schnell",
        "defaults": {"steps": 4, "width": 1024, "height": 1024},
        "cost": "~$0.003/枚",
    },
    "DreamShaper XL (万能)": {
        "model": "dreamshaperXL_v21Turbo_631290.safetensors",
        "defaults": {"steps": 8, "guidance_scale": 2, "width": 1024, "height": 1024},
        "cost": "~$0.002/枚",
    },
    "Realistic Vision (フォトリアル)": {
        "model": "realisticVisionV60B1_v51VAE_127635.safetensors",
        "defaults": {"steps": 25, "guidance_scale": 7, "width": 512, "height": 768},
        "cost": "~$0.0015/枚",
    },
    "MeinaMix (アニメ)": {
        "model": "meinamix_meinaV11_97584.safetensors",
        "defaults": {"steps": 25, "guidance_scale": 7, "width": 512, "height": 768},
        "cost": "~$0.0015/枚",
    },
}


class NovitaClient:
    def __init__(self, api_key=""):
        self.api_key = api_key

    def set_key(self, api_key):
        self.api_key = api_key

    def _request(self, method, endpoint, data=None):
        if not self.api_key:
            raise ValueError("Novita.ai API Key が設定されていません")
        url = f"{NOVITA_API_URL}{endpoint}"
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
            raise RuntimeError(f"Novita API Error ({e.code}): {error_body}")

    def generate_image(self, model_key, prompt, negative_prompt="",
                       width=1024, height=1024, steps=None, guidance=None,
                       seed=-1, num_images=1):
        """Generate image via Novita.ai. Returns list of base64 image data."""
        model_info = NOVITA_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明なモデル: {model_key}")

        model_name = model_info["model"]
        defaults = model_info["defaults"]

        # Flux models use a different endpoint
        is_flux = "flux" in model_name.lower()

        if is_flux:
            data = {
                "model_name": model_name,
                "prompt": prompt,
                "width": width or defaults.get("width", 1024),
                "height": height or defaults.get("height", 1024),
                "image_num": num_images,
            }
            if steps:
                data["steps"] = steps
            elif "steps" in defaults:
                data["steps"] = defaults["steps"]
            if guidance:
                data["guidance_scale"] = guidance
            elif "guidance_scale" in defaults:
                data["guidance_scale"] = defaults["guidance_scale"]
            if seed >= 0:
                data["seed"] = seed
            if negative_prompt:
                data["negative_prompt"] = negative_prompt

            result = self._request("POST", "/async/flux", data)
        else:
            data = {
                "model_name": model_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt or "(worst quality, low quality:1.3)",
                "width": width or defaults.get("width", 512),
                "height": height or defaults.get("height", 768),
                "image_num": num_images,
                "steps": steps or defaults.get("steps", 25),
                "guidance_scale": guidance or defaults.get("guidance_scale", 7),
                "enable_nsfw_detection": False,
            }
            if seed >= 0:
                data["seed"] = seed

            result = self._request("POST", "/async/txt2img", data)

        # Poll for result
        task_id = result.get("task_id", "")
        if not task_id:
            raise RuntimeError("タスクIDが取得できませんでした")
        return self._poll_task(task_id)

    def _poll_task(self, task_id, timeout=300):
        """Poll for task completion."""
        start = time.time()
        while time.time() - start < timeout:
            result = self._request("GET", f"/async/task-result?task_id={task_id}")
            status = result.get("task", {}).get("status", "")
            if status == "TASK_STATUS_SUCCEED":
                images = result.get("images", [])
                urls = []
                for img in images:
                    url = img.get("image_url", "")
                    if url:
                        urls.append(url)
                return urls
            elif status in ("TASK_STATUS_FAILED", "TASK_STATUS_CANCELED"):
                reason = result.get("task", {}).get("reason", "Unknown")
                raise RuntimeError(f"生成失敗: {reason}")
            time.sleep(2)
        raise TimeoutError(f"生成が{timeout}秒以内に完了しませんでした")

    def img2img(self, model_key, prompt, image_path, negative_prompt="",
                strength=0.7, width=None, height=None, steps=None, guidance=None,
                seed=-1):
        """img2img via Novita.ai. Fully uncensored. Returns list of image URLs."""
        import base64
        model_info = NOVITA_MODELS.get(model_key)
        if not model_info:
            # Default to Realistic Vision for img2img
            model_info = NOVITA_MODELS.get("Realistic Vision (フォトリアル)", list(NOVITA_MODELS.values())[0])

        model_name = model_info["model"]
        defaults = model_info["defaults"]
        is_flux = "flux" in model_name.lower()

        # Read and encode image
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        if is_flux:
            # Flux img2img
            data = {
                "model_name": model_name,
                "prompt": prompt,
                "image_url": f"data:image/png;base64,{img_b64}",
                "strength": strength,
                "width": width or defaults.get("width", 1024),
                "height": height or defaults.get("height", 1024),
                "image_num": 1,
            }
            if steps:
                data["steps"] = steps
            elif "steps" in defaults:
                data["steps"] = defaults["steps"]
            if guidance:
                data["guidance_scale"] = guidance
            elif "guidance_scale" in defaults:
                data["guidance_scale"] = defaults["guidance_scale"]
            if seed >= 0:
                data["seed"] = seed
            if negative_prompt:
                data["negative_prompt"] = negative_prompt

            result = self._request("POST", "/async/flux-img2img", data)
        else:
            # SD img2img — fully uncensored
            data = {
                "model_name": model_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt or "(worst quality, low quality:1.3)",
                "init_imgs": [img_b64],
                "denoising_strength": strength,
                "width": width or defaults.get("width", 512),
                "height": height or defaults.get("height", 768),
                "image_num": 1,
                "steps": steps or defaults.get("steps", 25),
                "guidance_scale": guidance or defaults.get("guidance_scale", 7),
                "enable_nsfw_detection": False,
            }
            if seed >= 0:
                data["seed"] = seed

            result = self._request("POST", "/async/img2img", data)

        task_id = result.get("task_id", "")
        if not task_id:
            raise RuntimeError("タスクIDが取得できませんでした")
        return self._poll_task(task_id)

    def check_api_key(self):
        try:
            self._request("GET", "/async/task-result?task_id=test")
            return True
        except RuntimeError as e:
            # 404 or similar means the key works but task doesn't exist
            if "401" not in str(e):
                return True
            return False
        except Exception:
            return False


def download_novita_image(url):
    """Download image from Novita URL and return PIL Image."""
    from PIL import Image
    req = urllib.request.Request(url, headers={"User-Agent": "AI-diffusion/1.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    return Image.open(io.BytesIO(resp.read()))
