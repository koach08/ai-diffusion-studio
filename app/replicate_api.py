"""Replicate API Client - Serverless AI model inference."""
import json
import time
import urllib.request
import urllib.error

REPLICATE_API_URL = "https://api.replicate.com/v1"

# Available models with their Replicate identifiers and default params
MODELS = {
    "Flux 1.1 Pro (最高品質)": {
        "model": "black-forest-labs/flux-1.1-pro",
        "defaults": {"width": 1024, "height": 1024, "prompt_upsampling": True},
        "cost_per_image": "~$0.04",
    },
    "Flux Dev (高品質・バランス)": {
        "model": "black-forest-labs/flux-dev",
        "defaults": {"width": 1024, "height": 1024, "num_inference_steps": 28, "guidance": 3.5},
        "cost_per_image": "~$0.03",
    },
    "Flux Schnell (高速・安い)": {
        "model": "black-forest-labs/flux-schnell",
        "defaults": {"width": 1024, "height": 1024, "num_inference_steps": 4},
        "cost_per_image": "~$0.003",
    },
    "SDXL (Stable Diffusion XL)": {
        "model": "stability-ai/sdxl",
        "defaults": {"width": 1024, "height": 1024, "num_inference_steps": 25, "guidance_scale": 7.5},
        "cost_per_image": "~$0.01",
    },
    "Playground v2.5 (高品質・無料枠あり)": {
        "model": "playgroundai/playground-v2.5-1024px-aesthetic",
        "defaults": {"width": 1024, "height": 1024, "num_inference_steps": 25, "guidance_scale": 3.0},
        "cost_per_image": "~$0.01",
    },
}

# Video models
VIDEO_MODELS = {
    "Wan 2.1 (高品質・推奨)": {
        "model": "wan-ai/wan-2.1-t2v-480p",
        "defaults": {},
        "cost_per_video": "~$0.35",
    },
    "Minimax Video-01 (高品質動画)": {
        "model": "minimax/video-01",
        "defaults": {},
        "cost_per_video": "~$0.13",
    },
    "Kling v1.6 (高品質動画)": {
        "model": "kwaivgi/kling-v1.6-standard",
        "defaults": {},
        "cost_per_video": "~$0.10",
    },
    "Hunyuan Video (テンセント)": {
        "model": "tencent/hunyuan-video",
        "defaults": {"width": 864, "height": 480, "num_frames": 129},
        "cost_per_video": "~$0.32",
    },
    "LTX Video (高速)": {
        "model": "lightricks/ltx-video-v0.9.1",
        "defaults": {},
        "cost_per_video": "~$0.02",
    },
}


class ReplicateClient:
    def __init__(self, api_key=""):
        self.api_key = api_key

    def _request(self, method, endpoint, data=None):
        if not self.api_key:
            raise ValueError("Replicate API Key が設定されていません")
        url = f"{REPLICATE_API_URL}{endpoint}"
        body = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Prefer": "wait",
            },
            method=method,
        )
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Replicate API Error ({e.code}): {error_body}")

    def create_prediction(self, model, input_params):
        """Create a prediction (start generation)."""
        # Use /models/{owner}/{name}/predictions for official models
        data = {"input": input_params}
        return self._request("POST", f"/models/{model}/predictions", data)

    def get_prediction(self, prediction_id):
        """Check prediction status."""
        return self._request("GET", f"/predictions/{prediction_id}")

    def run(self, model, input_params, timeout=300):
        """Run a model and wait for results. Returns list of output URLs."""
        prediction = self.create_prediction(model, input_params)
        pred_id = prediction["id"]
        status = prediction.get("status", "")

        # If the "Prefer: wait" header worked, we might already have output
        if status == "succeeded" and prediction.get("output"):
            output = prediction["output"]
            return output if isinstance(output, list) else [output]

        # Otherwise poll
        start = time.time()
        while time.time() - start < timeout:
            pred = self.get_prediction(pred_id)
            status = pred.get("status", "")
            if status == "succeeded":
                output = pred.get("output")
                if output is None:
                    return []
                return output if isinstance(output, list) else [output]
            elif status == "failed":
                error = pred.get("error", "Unknown error")
                raise RuntimeError(f"生成失敗: {error}")
            elif status == "canceled":
                raise RuntimeError("生成がキャンセルされました")
            time.sleep(2)

        raise TimeoutError(f"生成が{timeout}秒以内に完了しませんでした")

    def generate_image(self, model_key, prompt, negative_prompt="",
                       width=1024, height=1024, num_outputs=1, seed=-1,
                       steps=None, guidance=None):
        """High-level image generation. Returns list of image URLs."""
        model_info = MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明なモデル: {model_key}")

        model_id = model_info["model"]
        params = {**model_info["defaults"]}
        params["prompt"] = prompt
        params["width"] = width
        params["height"] = height
        params["num_outputs"] = num_outputs

        if negative_prompt and "sdxl" in model_id.lower():
            params["negative_prompt"] = negative_prompt
        if negative_prompt and "playground" in model_id.lower():
            params["negative_prompt"] = negative_prompt

        if seed >= 0:
            params["seed"] = seed
        if steps is not None:
            # Different models use different param names
            if "num_inference_steps" in model_info["defaults"]:
                params["num_inference_steps"] = steps
        if guidance is not None:
            if "guidance" in model_info["defaults"]:
                params["guidance"] = guidance
            elif "guidance_scale" in model_info["defaults"]:
                params["guidance_scale"] = guidance

        return self.run(model_id, params, timeout=300)

    def generate_video(self, model_key, prompt, image_url=None, **kwargs):
        """High-level video generation. Returns list of video URLs."""
        model_info = VIDEO_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明な動画モデル: {model_key}")

        model_id = model_info["model"]
        params = {**model_info["defaults"]}
        params["prompt"] = prompt
        if image_url:
            params["first_frame_image"] = image_url
        params.update(kwargs)

        return self.run(model_id, params, timeout=600)

    def check_api_key(self):
        """Verify the API key works."""
        try:
            self._request("GET", "/account")
            return True
        except Exception:
            return False


def download_url_to_pil(url):
    """Download an image URL and return PIL Image."""
    from PIL import Image
    import io
    req = urllib.request.Request(url, headers={"User-Agent": "AI-diffusion/1.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    return Image.open(io.BytesIO(resp.read()))
