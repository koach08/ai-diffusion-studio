"""fal.ai API Client - Flux image & video generation (NSFW OK)."""
import os
import fal_client
import urllib.request
import io

FAL_MODELS = {
    "Flux Pro 1.1 (最高品質)": {
        "model_id": "fal-ai/flux-pro/v1.1",
        "defaults": {"safety_tolerance": 6},
        "cost": "~$0.05/枚",
    },
    "Flux Dev (高品質・NSFW OK)": {
        "model_id": "fal-ai/flux/dev",
        "defaults": {"num_inference_steps": 28, "guidance_scale": 3.5, "safety_tolerance": 6},
        "cost": "~$0.025/枚",
    },
    "Flux Schnell (高速・安い)": {
        "model_id": "fal-ai/flux/schnell",
        "defaults": {"num_inference_steps": 4, "safety_tolerance": 6},
        "cost": "~$0.003/枚",
    },
    "Flux Realism (フォトリアル特化)": {
        "model_id": "fal-ai/flux-realism",
        "defaults": {"num_inference_steps": 28, "guidance_scale": 3.5, "safety_tolerance": 6},
        "cost": "~$0.025/枚",
    },
}

FAL_VIDEO_MODELS = {
    "Kling 2.5 Turbo Pro (最高品質)": {
        "model_id": "fal-ai/kling-video/v2.5-turbo/pro/text-to-video",
        "cost": "~$0.10/本",
    },
    "LTX 2.3 (高速・安い)": {
        "model_id": "fal-ai/ltx-2.3/text-to-video/fast",
        "cost": "~$0.02/本",
    },
    "Veo 3.1 Fast (Google最新)": {
        "model_id": "fal-ai/veo3.1/fast",
        "cost": "~$0.15/本",
    },
    "Sora 2 (OpenAI)": {
        "model_id": "fal-ai/sora-2/text-to-video",
        "cost": "~$0.20/本",
    },
}


class FalClient:
    def __init__(self, api_key=""):
        self.api_key = api_key
        if api_key:
            os.environ["FAL_KEY"] = api_key

    def set_key(self, api_key):
        self.api_key = api_key
        os.environ["FAL_KEY"] = api_key

    def generate_image(self, model_key, prompt, negative_prompt="",
                       width=1024, height=1024, steps=None, guidance=None,
                       seed=-1, num_images=1, safety_checker=False):
        """Generate image via fal.ai. Returns list of image URLs."""
        if not self.api_key:
            raise ValueError("fal.ai API Key が設定されていません")
        os.environ["FAL_KEY"] = self.api_key

        model_info = FAL_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明なモデル: {model_key}")

        model_id = model_info["model_id"]
        args = {
            "prompt": prompt,
            "image_size": {"width": width, "height": height},
            "num_images": num_images,
            "enable_safety_checker": False,
        }

        defaults = model_info.get("defaults", {})
        if "safety_tolerance" in defaults:
            args["safety_tolerance"] = defaults["safety_tolerance"]

        if negative_prompt:
            args["negative_prompt"] = negative_prompt

        if "num_inference_steps" in defaults:
            args["num_inference_steps"] = defaults["num_inference_steps"]
        if "guidance_scale" in defaults:
            args["guidance_scale"] = defaults["guidance_scale"]

        if steps is not None:
            args["num_inference_steps"] = steps
        if guidance is not None:
            args["guidance_scale"] = guidance
        if seed >= 0:
            args["seed"] = seed

        result = fal_client.subscribe(model_id, arguments=args)
        urls = []
        for img in result.get("images", []):
            url = img.get("url", "")
            if url:
                urls.append(url)
        return urls

    def generate_video(self, model_key, prompt):
        """Generate video via fal.ai. Returns video URL."""
        if not self.api_key:
            raise ValueError("fal.ai API Key が設定されていません")
        os.environ["FAL_KEY"] = self.api_key

        model_info = FAL_VIDEO_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明な動画モデル: {model_key}")

        model_id = model_info["model_id"]
        args = {"prompt": prompt}

        result = fal_client.subscribe(model_id, arguments=args)

        if isinstance(result, dict):
            vid = result.get("video", {})
            if isinstance(vid, dict):
                return vid.get("url", "")
            elif isinstance(vid, str):
                return vid
            return result.get("url", "")
        return ""

    def check_api_key(self):
        try:
            os.environ["FAL_KEY"] = self.api_key
            fal_client.subscribe(
                "fal-ai/flux/schnell",
                arguments={"prompt": "test", "image_size": {"width": 256, "height": 256}, "num_inference_steps": 1},
            )
            return True
        except Exception:
            return False


def download_fal_image(url):
    """Download image from fal.ai URL and return PIL Image."""
    from PIL import Image
    req = urllib.request.Request(url, headers={"User-Agent": "AI-diffusion/1.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    return Image.open(io.BytesIO(resp.read()))
