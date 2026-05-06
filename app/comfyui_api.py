"""ComfyUI API Client - Communicates with ComfyUI server to queue and retrieve images/videos."""
import io
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
import uuid


class ComfyUIClient:
    _HEADERS = {"User-Agent": "AI-diffusion/1.0"}

    def __init__(self, server_url="http://127.0.0.1:8188"):
        self.server_url = server_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _request(self, path, data=None, extra_headers=None, timeout=10):
        """Make a request with proper User-Agent (required for Cloudflare)."""
        url = f"{self.server_url}{path}"
        headers = dict(self._HEADERS)
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(url, data=data, headers=headers)
        return urllib.request.urlopen(req, timeout=timeout)

    def is_server_running(self):
        # Retry once for HTTPS (Cloudflare tunnel SSL handshake can flake)
        is_https = self.server_url.startswith("https")
        timeout = 15 if is_https else 3
        attempts = 2 if is_https else 1
        for i in range(attempts):
            try:
                self._request("/system_stats", timeout=timeout)
                return True
            except Exception:
                if i < attempts - 1:
                    import time
                    time.sleep(1)
                continue
        return False

    def queue_prompt(self, workflow):
        """Queue a workflow prompt and return the prompt_id."""
        data = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode("utf-8")
        resp = self._request("/prompt", data=data, extra_headers={"Content-Type": "application/json"})
        return json.loads(resp.read())["prompt_id"]

    def get_history(self, prompt_id):
        resp = self._request(f"/history/{prompt_id}")
        return json.loads(resp.read())

    def get_image(self, filename, subfolder="", folder_type="output"):
        params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        resp = self._request(f"/view?{params}")
        return resp.read()

    def get_queue_status(self):
        """Check ComfyUI queue status."""
        try:
            resp = self._request("/queue", timeout=5)
            return json.loads(resp.read())
        except Exception:
            return {}

    def wait_for_result(self, prompt_id, timeout=300):
        """Poll until the prompt completes, return list of image bytes."""
        import logging
        logger = logging.getLogger("ai-diffusion")
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                entry = history[prompt_id]
                # Check for execution errors
                status_info = entry.get("status", {})
                if status_info.get("status_str") == "error":
                    msgs = status_info.get("messages", [])
                    logger.error(f"[ComfyUI] Workflow error: {msgs}")
                outputs = entry.get("outputs", {})
                if not outputs:
                    logger.warning(f"[ComfyUI] No outputs in result. Status: {status_info}")
                images = []
                gifs = []
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for img_info in node_output["images"]:
                            img_data = self.get_image(
                                img_info["filename"],
                                img_info.get("subfolder", ""),
                                img_info.get("type", "output"),
                            )
                            images.append(img_data)
                    if "gifs" in node_output:
                        for gif_info in node_output["gifs"]:
                            gif_data = self.get_image(
                                gif_info["filename"],
                                gif_info.get("subfolder", ""),
                                gif_info.get("type", "output"),
                            )
                            gifs.append((gif_data, gif_info["filename"]))
                logger.info(f"[ComfyUI] Result: {len(images)} images, {len(gifs)} gifs, nodes: {list(outputs.keys())}")
                return {"images": images, "gifs": gifs}
            time.sleep(1)
        raise TimeoutError(f"ComfyUI did not complete within {timeout}s")

    def generate(self, workflow, timeout=300):
        """Queue a workflow and wait for results. Returns list of PIL Images."""
        from PIL import Image

        prompt_id = self.queue_prompt(workflow)
        result = self.wait_for_result(prompt_id, timeout)
        images = result if isinstance(result, list) else result.get("images", [])
        return [Image.open(io.BytesIO(data)) for data in images]

    def generate_video(self, workflow, output_dir, timeout=600):
        """Queue a video workflow. Returns (frame_images, video_path)."""
        from PIL import Image

        prompt_id = self.queue_prompt(workflow)
        result = self.wait_for_result(prompt_id, timeout)

        frames = []
        video_path = None

        # Handle GIF/video outputs from VHS nodes
        for gif_data, filename in result.get("gifs", []):
            os.makedirs(output_dir, exist_ok=True)
            ext = os.path.splitext(filename)[1].lower()
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"video_{ts}{ext}"
            video_path = os.path.join(output_dir, save_name)
            with open(video_path, "wb") as f:
                f.write(gif_data)
            break  # Take first video output

        # Handle frame images
        for img_data in result.get("images", []):
            try:
                frames.append(Image.open(io.BytesIO(img_data)))
            except Exception:
                pass

        # If no VHS output but we have frames, assemble into GIF
        if not video_path and frames:
            os.makedirs(output_dir, exist_ok=True)
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(output_dir, f"video_{ts}.gif")
            frames[0].save(
                video_path, save_all=True, append_images=frames[1:],
                duration=100, loop=0,
            )

        return frames, video_path

    def upload_image(self, filepath, filename=None):
        """Upload an image to ComfyUI's input folder."""
        if filename is None:
            filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            img_data = f.read()

        # Multipart form upload
        import mimetypes
        boundary = "----ComfyUIBoundary"
        content_type = mimetypes.guess_type(filepath)[0] or "image/png"

        body = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8") + img_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            f"{self.server_url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        return result  # {"name": "filename.png", "subfolder": "", "type": "input"}

    def get_models(self):
        """Get list of available checkpoint models from ComfyUI."""
        try:
            timeout = 10 if self.server_url.startswith("https") else 5
            resp = self._request("/object_info/CheckpointLoaderSimple", timeout=timeout)
            data = json.loads(resp.read())
            models = data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
            return [m for m in models if m and not m.startswith("put_")]
        except Exception:
            return []


def build_txt2img_workflow(
    prompt,
    negative_prompt="",
    model="",
    width=512,
    height=768,
    steps=25,
    cfg=7.0,
    sampler="euler_ancestral",
    scheduler="normal",
    seed=-1,
    batch_size=1,
    lora_name="",
    lora_strength=0.8,
    vae_name="",
    hires_fix=False,
    hires_scale=1.5,
    hires_denoise=0.5,
    hires_steps=15,
    upscale_model="",
    face_detailer=False,
    face_denoise=0.4,
    face_guide_size=512,
    face_bbox_model="face_yolov8m.pt",
    loras=None,
):
    """Build a ComfyUI txt2img workflow JSON. With optional Hires Fix.

    loras: list of (lora_name, strength) tuples for multi-LoRA support.
           If provided, overrides lora_name/lora_strength.
    """
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    # Build LoRA list: prefer new 'loras' param, fall back to legacy single LoRA
    lora_list = []
    if loras:
        lora_list = [(n, s) for n, s in loras if n and n not in ("None", "")]
    elif lora_name and lora_name not in ("None", ""):
        lora_list = [(lora_name, lora_strength)]

    # Determine model/clip output refs based on LoRA chain
    if lora_list:
        last_lora_id = str(10 + (len(lora_list) - 1) * 2)
        model_out = [last_lora_id, 0]
        clip_out = [last_lora_id, 1]
    else:
        model_out = ["4", 0]
        clip_out = ["4", 1]
    vae_out = ["4", 2] if not vae_name else ["11", 0]

    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": 1.0,
                "latent_image": ["5", 0],
                "model": model_out,
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": sampler,
                "scheduler": scheduler,
                "seed": seed,
                "steps": steps,
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": batch_size, "height": height, "width": width},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_out, "text": prompt},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_out, "text": negative_prompt},
        },
    }

    if hires_fix:
        # Hires Fix: 1st pass → upscale → 2nd pass with lower denoise
        if upscale_model:
            # Pixel-space upscale with model (higher quality)
            workflow["50"] = {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": vae_out},
            }
            workflow["51"] = {
                "class_type": "UpscaleModelLoader",
                "inputs": {"model_name": upscale_model},
            }
            workflow["52"] = {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": ["51", 0], "image": ["50", 0]},
            }
            workflow["53"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["52", 0],
                    "width": int(width * hires_scale),
                    "height": int(height * hires_scale),
                    "upscale_method": "lanczos",
                    "crop": "disabled",
                },
            }
            workflow["54"] = {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["53", 0], "vae": vae_out},
            }
            hires_latent = ["54", 0]
        else:
            # Latent upscale (no model needed)
            workflow["50"] = {
                "class_type": "LatentUpscale",
                "inputs": {
                    "samples": ["3", 0],
                    "upscale_method": "bislerp",
                    "width": int(width * hires_scale),
                    "height": int(height * hires_scale),
                    "crop": "disabled",
                },
            }
            hires_latent = ["50", 0]

        workflow["55"] = {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": hires_denoise,
                "latent_image": hires_latent,
                "model": model_out,
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": sampler,
                "scheduler": scheduler,
                "seed": seed,
                "steps": hires_steps,
            },
        }
        final_samples = ["55", 0]
    else:
        final_samples = ["3", 0]

    workflow["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": final_samples, "vae": vae_out},
    }

    if face_detailer:
        # FaceDetailer: detect faces → re-generate face region at higher detail
        workflow["60"] = {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": f"bbox/{face_bbox_model}"},
        }
        workflow["61"] = {
            "class_type": "SAMLoader",
            "inputs": {"model_name": "sam_vit_b_01ec64.pth", "device_mode": "AUTO"},
        }
        workflow["62"] = {
            "class_type": "FaceDetailer",
            "inputs": {
                "image": ["8", 0],
                "model": model_out,
                "clip": clip_out,
                "vae": vae_out,
                "positive": ["6", 0],
                "negative": ["7", 0],
                "bbox_detector": ["60", 0],
                "sam_model_opt": ["61", 0],
                "guide_size": face_guide_size,
                "guide_size_for": True,
                "max_size": 1024,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": face_denoise,
                "feather": 5,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": 0.5,
                "bbox_dilation": 10,
                "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1",
                "sam_dilation": 0,
                "sam_threshold": 0.93,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "wildcard": "",
                "cycle": 1,
            },
        }
        save_image_input = ["62", 0]
    else:
        save_image_input = ["8", 0]

    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "AI-diffusion", "images": save_image_input},
    }

    # Build LoRA chain: node 10, 12, 14, ...
    prev_model = ["4", 0]
    prev_clip = ["4", 1]
    for i, (l_name, l_str) in enumerate(lora_list):
        node_id = str(10 + i * 2)
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": l_name,
                "model": prev_model,
                "clip": prev_clip,
                "strength_model": l_str,
                "strength_clip": l_str,
            },
        }
        prev_model = [node_id, 0]
        prev_clip = [node_id, 1]

    if vae_name:
        workflow["11"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        }

    return workflow


def build_img2img_workflow(
    prompt,
    negative_prompt="",
    model="",
    image_path="",
    width=832,
    height=1216,
    steps=30,
    cfg=7.0,
    sampler="euler_ancestral",
    scheduler="normal",
    seed=-1,
    denoise=0.7,
    vae_name="",
    loras=None,
):
    """Build a ComfyUI img2img workflow. Loads image, encodes to latent, denoises with KSampler."""
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    lora_list = [(n, s) for n, s in (loras or []) if n and n not in ("None", "")]
    if lora_list:
        last_lora_id = str(10 + (len(lora_list) - 1) * 2)
        model_out = [last_lora_id, 0]
        clip_out = [last_lora_id, 1]
    else:
        model_out = ["4", 0]
        clip_out = ["4", 1]
    vae_out = ["4", 2] if not vae_name else ["11", 0]

    workflow = {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_out, "text": prompt},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_out, "text": negative_prompt},
        },
        "20": {
            "class_type": "LoadImage",
            "inputs": {"image": image_path},
        },
        "21": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["20", 0], "vae": vae_out},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": denoise,
                "latent_image": ["21", 0],
                "model": model_out,
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": sampler,
                "scheduler": scheduler,
                "seed": seed,
                "steps": steps,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": vae_out},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "img2img", "images": ["8", 0]},
        },
    }

    # Build LoRA chain
    prev_model = ["4", 0]
    prev_clip = ["4", 1]
    for i, (l_name, l_str) in enumerate(lora_list):
        node_id = str(10 + i * 2)
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": l_name,
                "model": prev_model,
                "clip": prev_clip,
                "strength_model": l_str,
                "strength_clip": l_str,
            },
        }
        prev_model = [node_id, 0]
        prev_clip = [node_id, 1]

    if vae_name:
        workflow["11"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        }

    return workflow


def build_refine_workflow(
    prompt,
    negative_prompt="",
    model="",
    image_path="",
    width=832,
    height=1216,
    steps=25,
    cfg=7.0,
    sampler="dpmpp_2m_sde",
    scheduler="karras",
    seed=-1,
    denoise=0.35,
    vae_name="",
    loras=None,
    upscale_model="4x-UltraSharp.pth",
    upscale_scale=1.5,
    face_detailer=True,
    face_denoise=0.35,
    face_guide_size=768,
    face_bbox_model="face_yolov8m.pt",
):
    """Build a ComfyUI refine workflow: img2img + pixel upscale + face detailer.

    Takes existing image, applies light denoise + upscale + face fix in one pass.
    Designed for post-generation quality enhancement.
    """
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    lora_list = [(n, s) for n, s in (loras or []) if n and n not in ("None", "")]
    if lora_list:
        last_lora_id = str(10 + (len(lora_list) - 1) * 2)
        model_out = [last_lora_id, 0]
        clip_out = [last_lora_id, 1]
    else:
        model_out = ["4", 0]
        clip_out = ["4", 1]
    vae_out = ["4", 2] if not vae_name else ["11", 0]

    workflow = {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_out, "text": prompt},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_out, "text": negative_prompt},
        },
        # Load source image
        "20": {
            "class_type": "LoadImage",
            "inputs": {"image": image_path},
        },
    }

    # Upscale path: pixel upscale with model → resize to target
    if upscale_model and upscale_model != "None":
        workflow["30"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": upscale_model},
        }
        workflow["31"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["30", 0], "image": ["20", 0]},
        }
        workflow["32"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["31", 0],
                "width": int(width * upscale_scale),
                "height": int(height * upscale_scale),
                "upscale_method": "lanczos",
                "crop": "disabled",
            },
        }
        img_to_encode = ["32", 0]
    else:
        img_to_encode = ["20", 0]

    # Encode to latent → light denoise pass
    workflow["21"] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": img_to_encode, "vae": vae_out},
    }
    workflow["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "cfg": cfg,
            "denoise": denoise,
            "latent_image": ["21", 0],
            "model": model_out,
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": sampler,
            "scheduler": scheduler,
            "seed": seed,
            "steps": steps,
        },
    }
    workflow["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": vae_out},
    }

    # Face Detailer
    if face_detailer:
        workflow["60"] = {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": f"bbox/{face_bbox_model}"},
        }
        workflow["61"] = {
            "class_type": "SAMLoader",
            "inputs": {"model_name": "sam_vit_b_01ec64.pth", "device_mode": "AUTO"},
        }
        workflow["62"] = {
            "class_type": "FaceDetailer",
            "inputs": {
                "image": ["8", 0],
                "model": model_out,
                "clip": clip_out,
                "vae": vae_out,
                "positive": ["6", 0],
                "negative": ["7", 0],
                "bbox_detector": ["60", 0],
                "sam_model_opt": ["61", 0],
                "guide_size": face_guide_size,
                "guide_size_for": True,
                "max_size": 1536,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": face_denoise,
                "feather": 5,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": 0.5,
                "bbox_dilation": 10,
                "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1",
                "sam_dilation": 0,
                "sam_threshold": 0.93,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "wildcard": "",
                "cycle": 1,
            },
        }
        save_input = ["62", 0]
    else:
        save_input = ["8", 0]

    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "refine", "images": save_input},
    }

    # LoRA chain
    prev_model = ["4", 0]
    prev_clip = ["4", 1]
    for i, (l_name, l_str) in enumerate(lora_list):
        node_id = str(10 + i * 2)
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": l_name,
                "model": prev_model,
                "clip": prev_clip,
                "strength_model": l_str,
                "strength_clip": l_str,
            },
        }
        prev_model = [node_id, 0]
        prev_clip = [node_id, 1]

    if vae_name:
        workflow["11"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        }

    return workflow


def build_inpaint_workflow(
    prompt,
    negative_prompt="",
    model="",
    image_path="",
    width=832,
    height=1216,
    steps=30,
    cfg=5.0,
    sampler="euler_ancestral",
    scheduler="normal",
    seed=-1,
    denoise=0.85,
    vae_name="",
    loras=None,
):
    """Build a ComfyUI inpainting workflow with edge blending.

    The image_path should be a PNG with alpha channel as mask.
    LoadImage output 0 = IMAGE, output 1 = MASK (from alpha).
    Transparent alpha (0) = area to regenerate.

    Pipeline:
      LoadImage → GrowMask(+12px) → FeatherMask(edges) → VAEEncodeForInpaint
      → KSampler → VAEDecode → ImageCompositeMasked (blend with original) → Save
    """
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    lora_list = [(n, s) for n, s in (loras or []) if n and n not in ("None", "")]
    if lora_list:
        last_lora_id = str(10 + (len(lora_list) - 1) * 2)
        model_out = [last_lora_id, 0]
        clip_out = [last_lora_id, 1]
    else:
        model_out = ["4", 0]
        clip_out = ["4", 1]
    vae_out = ["4", 2] if not vae_name else ["11", 0]

    workflow = {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_out, "text": prompt},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": clip_out, "text": negative_prompt},
        },
        # Load image with alpha mask
        "20": {
            "class_type": "LoadImage",
            "inputs": {"image": image_path},
        },
        # Grow mask outward for better coverage
        "30": {
            "class_type": "GrowMask",
            "inputs": {
                "mask": ["20", 1],
                "expand": 12,
                "tapered_corners": True,
            },
        },
        # Feather mask edges for smooth blending
        "31": {
            "class_type": "FeatherMask",
            "inputs": {
                "mask": ["30", 0],
                "left": 16,
                "top": 16,
                "right": 16,
                "bottom": 16,
            },
        },
        # Encode for inpainting with processed mask
        "21": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {
                "pixels": ["20", 0],
                "vae": vae_out,
                "mask": ["31", 0],
                "grow_mask_by": 6,
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": denoise,
                "latent_image": ["21", 0],
                "model": model_out,
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": sampler,
                "scheduler": scheduler,
                "seed": seed,
                "steps": steps,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": vae_out},
        },
        # Composite inpainted result onto original using feathered mask
        "32": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["20", 0],
                "source": ["8", 0],
                "mask": ["31", 0],
                "x": 0,
                "y": 0,
                "resize_source": False,
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "inpaint", "images": ["32", 0]},
        },
    }

    # Build LoRA chain
    prev_model = ["4", 0]
    prev_clip = ["4", 1]
    for i, (l_name, l_str) in enumerate(lora_list):
        node_id = str(10 + i * 2)
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": l_name,
                "model": prev_model,
                "clip": prev_clip,
                "strength_model": l_str,
                "strength_clip": l_str,
            },
        }
        prev_model = [node_id, 0]
        prev_clip = [node_id, 1]

    if vae_name:
        workflow["11"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        }

    return workflow


def build_animatediff_workflow(
    prompt,
    negative_prompt="",
    model="",
    motion_model="mm_sd_v15_v2.ckpt",
    width=512,
    height=512,
    steps=20,
    cfg=7.5,
    sampler="euler_ancestral",
    scheduler="normal",
    seed=-1,
    frame_count=16,
    fps=8,
    lora_name="",
    lora_strength=0.8,
    vae_name="",
    output_format="gif",
    loras=None,
):
    """Build AnimateDiff txt2vid workflow for ComfyUI."""
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    # Build LoRA list: prefer new 'loras' param, fall back to legacy single LoRA
    lora_list = []
    if loras:
        lora_list = [(n, s) for n, s in loras if n and n not in ("None", "")]
    elif lora_name and lora_name not in ("None", ""):
        lora_list = [(lora_name, lora_strength)]

    if lora_list:
        last_lora_id = str(10 + (len(lora_list) - 1) * 2)
        model_out_for_ad = [last_lora_id, 0]
        clip_out = [last_lora_id, 1]
    else:
        model_out_for_ad = ["4", 0]
        clip_out = ["4", 1]

    workflow = {
        # Checkpoint loader
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model},
        },
        # AnimateDiff loader
        "20": {
            "class_type": "ADE_AnimateDiffLoaderWithContext",
            "inputs": {
                "model": model_out_for_ad,
                "model_name": motion_model,
                "beta_schedule": "sqrt_linear (AnimateDiff)",
                "context_options": ["21", 0],
            },
        },
        # Context options
        "21": {
            "class_type": "ADE_StandardStaticContextOptions",
            "inputs": {
                "context_length": 16,
                "context_overlap": 4,
            },
        },
        # Empty latent (with frame count as batch)
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": frame_count, "height": height, "width": width},
        },
        # Positive prompt
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": clip_out,
                "text": prompt,
            },
        },
        # Negative prompt
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": clip_out,
                "text": negative_prompt,
            },
        },
        # KSampler
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": 1.0,
                "latent_image": ["5", 0],
                "model": ["20", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": sampler,
                "scheduler": scheduler,
                "seed": seed,
                "steps": steps,
            },
        },
        # VAE Decode
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2] if not vae_name else ["11", 0],
            },
        },
    }

    # Output node - use VHS if available, otherwise SaveImage
    if output_format in ("gif", "webp"):
        workflow["30"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "AnimateDiff",
                "format": "image/gif" if output_format == "gif" else "image/webp",
                "pingpong": False,
                "save_output": True,
            },
        }
    elif output_format == "mp4":
        workflow["30"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "AnimateDiff",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        }
    else:
        # Fallback: save as individual frames
        workflow["9"] = {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "AnimateDiff_frame", "images": ["8", 0]},
        }

    # Build LoRA chain
    prev_model = ["4", 0]
    prev_clip = ["4", 1]
    for i, (l_name, l_str) in enumerate(lora_list):
        node_id = str(10 + i * 2)
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": l_name,
                "model": prev_model,
                "clip": prev_clip,
                "strength_model": l_str,
                "strength_clip": l_str,
            },
        }
        prev_model = [node_id, 0]
        prev_clip = [node_id, 1]

    if vae_name:
        workflow["11"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        }

    return workflow


def build_img2vid_workflow(
    image_path,
    model="",
    motion_model="mm_sd_v15_v2.ckpt",
    width=512,
    height=512,
    steps=20,
    cfg=7.5,
    sampler="euler_ancestral",
    scheduler="normal",
    seed=-1,
    frame_count=16,
    fps=8,
    denoise=0.7,
    prompt="",
    negative_prompt="",
    output_format="gif",
):
    """Build img2vid workflow: takes an image and animates it using AnimateDiff."""
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    workflow = {
        # Checkpoint loader
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model},
        },
        # Load input image
        "40": {
            "class_type": "LoadImage",
            "inputs": {"image": os.path.basename(image_path)},
        },
        # Resize image to match target
        "41": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["40", 0],
                "width": width,
                "height": height,
                "upscale_method": "lanczos",
                "crop": "center",
            },
        },
        # VAE Encode the image
        "42": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["41", 0],
                "vae": ["4", 2],
            },
        },
        # Repeat latent for frame count
        "43": {
            "class_type": "RepeatLatentBatch",
            "inputs": {
                "samples": ["42", 0],
                "amount": frame_count,
            },
        },
        # AnimateDiff loader
        "20": {
            "class_type": "ADE_AnimateDiffLoaderWithContext",
            "inputs": {
                "model": ["4", 0],
                "model_name": motion_model,
                "beta_schedule": "sqrt_linear (AnimateDiff)",
                "context_options": ["21", 0],
            },
        },
        "21": {
            "class_type": "ADE_StandardStaticContextOptions",
            "inputs": {
                "context_length": 16,
                "context_overlap": 4,
            },
        },
        # Prompts
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": prompt or "high quality, smooth motion"},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": negative_prompt or "worst quality, static, blurry"},
        },
        # KSampler with denoise < 1 for img2vid
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": denoise,
                "latent_image": ["43", 0],
                "model": ["20", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": sampler,
                "scheduler": scheduler,
                "seed": seed,
                "steps": steps,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
    }

    # Output
    if output_format in ("gif", "webp", "mp4"):
        fmt = {"gif": "image/gif", "webp": "image/webp", "mp4": "video/h264-mp4"}
        workflow["30"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "Img2Vid",
                "format": fmt[output_format],
                "pingpong": False,
                "save_output": True,
            },
        }
    else:
        workflow["9"] = {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "Img2Vid_frame", "images": ["8", 0]},
        }

    return workflow


def build_vid2vid_workflow(
    video_path,
    prompt="",
    negative_prompt="",
    model="",
    motion_model="mm_sd_v15_v2.ckpt",
    width=512,
    height=512,
    steps=20,
    cfg=7.5,
    sampler="euler_ancestral",
    scheduler="normal",
    seed=-1,
    denoise=0.55,
    fps=8,
    frame_limit=32,
    output_format="gif",
):
    """Build vid2vid workflow: load existing video → AnimateDiff + KSampler → stylized output."""
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    workflow = {
        # Load video frames
        "40": {
            "class_type": "VHS_LoadVideoPath",
            "inputs": {
                "video": video_path,
                "force_rate": fps,
                "force_size": "Custom Width",
                "custom_width": width,
                "custom_height": height,
                "frame_load_cap": frame_limit,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        # Checkpoint
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model},
        },
        # VAE Encode video frames
        "42": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["40", 0],
                "vae": ["4", 2],
            },
        },
        # AnimateDiff
        "20": {
            "class_type": "ADE_AnimateDiffLoaderWithContext",
            "inputs": {
                "model": ["4", 0],
                "model_name": motion_model,
                "beta_schedule": "sqrt_linear (AnimateDiff)",
                "context_options": ["21", 0],
            },
        },
        "21": {
            "class_type": "ADE_StandardStaticContextOptions",
            "inputs": {
                "context_length": 16,
                "context_overlap": 4,
            },
        },
        # Prompts
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": prompt or "high quality, smooth motion"},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": negative_prompt or "worst quality, blurry, distorted"},
        },
        # KSampler with low denoise to preserve original video structure
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": denoise,
                "latent_image": ["42", 0],
                "model": ["20", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": sampler,
                "scheduler": scheduler,
                "seed": seed,
                "steps": steps,
            },
        },
        # Decode
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
    }

    # Output
    if output_format in ("gif", "webp", "mp4"):
        fmt = {"gif": "image/gif", "webp": "image/webp", "mp4": "video/h264-mp4"}
        workflow["30"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "Vid2Vid",
                "format": fmt[output_format],
                "pingpong": False,
                "save_output": True,
            },
        }
    else:
        workflow["9"] = {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "Vid2Vid_frame", "images": ["8", 0]},
        }

    return workflow


def build_flux_workflow(
    prompt,
    unet_model="flux1-dev-fp8.safetensors",
    clip_l="clip_l.safetensors",
    t5xxl="t5xxl_fp8_e4m3fn.safetensors",
    vae_name="ae.safetensors",
    width=1024,
    height=1024,
    steps=20,
    guidance=3.5,
    sampler="euler",
    scheduler="simple",
    seed=-1,
    batch_size=1,
    weight_dtype="fp8_e4m3fn",
    lora_name="",
    lora_strength=0.8,
):
    """Build Flux.1 workflow for ComfyUI. Uses SamplerCustomAdvanced pipeline."""
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    model_out = ["1", 0] if not lora_name else ["10", 0]

    workflow = {
        # UNET Loader (Flux model)
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_model,
                "weight_dtype": weight_dtype,
            },
        },
        # DualCLIP Loader (clip_l + t5xxl)
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": clip_l,
                "clip_name2": t5xxl,
                "type": "flux",
            },
        },
        # VAE Loader
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        # CLIP Text Encode (Flux-specific with guidance)
        "4": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": ["2", 0] if not lora_name else ["10", 1],
                "clip_l": prompt,
                "t5xxl": prompt,
                "guidance": guidance,
            },
        },
        # Empty Latent Image
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": batch_size,
                "height": height,
                "width": width,
            },
        },
        # Random Noise
        "6": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        # Basic Guider
        "7": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": model_out,
                "conditioning": ["4", 0],
            },
        },
        # Sampler Select
        "8": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": sampler},
        },
        # Scheduler
        "9": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": model_out,
                "scheduler": scheduler,
                "steps": steps,
                "denoise": 1.0,
            },
        },
        # SamplerCustomAdvanced
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["7", 0],
                "sampler": ["8", 0],
                "sigmas": ["9", 0],
                "latent_image": ["5", 0],
            },
        },
        # VAE Decode
        "12": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["11", 0],
                "vae": ["3", 0],
            },
        },
        # Save Image
        "13": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "Flux",
                "images": ["12", 0],
            },
        },
    }

    if lora_name:
        workflow["10"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora_name,
                "model": ["1", 0],
                "clip": ["2", 0],
                "strength_clip": lora_strength,
                "strength_model": lora_strength,
            },
        }

    return workflow


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ControlNet Workflow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_controlnet_workflow(
    prompt, negative_prompt="", model="",
    control_image_path="", control_type="openpose", control_strength=1.0,
    width=512, height=768, steps=25, cfg=7.0,
    sampler="euler_ancestral", scheduler="normal", seed=-1,
    loras=None, vae_name="",
):
    """Build ControlNet guided generation workflow."""
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    CN_MODELS = {
        "openpose": "control_v11p_sd15_openpose.pth",
        "depth": "control_v11f1p_sd15_depth.pth",
        "lineart": "control_v11p_sd15_lineart.pth",
    }
    cn_model = CN_MODELS.get(control_type, CN_MODELS["openpose"])

    lora_list = [(n, s) for n, s in (loras or []) if n and n not in ("None", "")]
    if lora_list:
        last_id = str(10 + (len(lora_list) - 1) * 2)
        model_out, clip_out = [last_id, 0], [last_id, 1]
    else:
        model_out, clip_out = ["4", 0], ["4", 1]
    vae_out = ["4", 2] if not vae_name else ["11", 0]

    workflow = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "40": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cn_model}},
        "41": {"class_type": "LoadImage", "inputs": {"image": control_image_path}},
        "42": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["6", 0], "negative": ["7", 0],
                "control_net": ["40", 0], "image": ["41", 0],
                "strength": control_strength, "start_percent": 0.0, "end_percent": 1.0,
            },
        },
        "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": height, "width": width}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_out, "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_out, "text": negative_prompt}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg, "denoise": 1.0, "latent_image": ["5", 0],
                "model": model_out, "negative": ["42", 1], "positive": ["42", 0],
                "sampler_name": sampler, "scheduler": scheduler, "seed": seed, "steps": steps,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": vae_out}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "controlnet", "images": ["8", 0]}},
    }
    prev_m, prev_c = ["4", 0], ["4", 1]
    for i, (ln, ls) in enumerate(lora_list):
        nid = str(10 + i * 2)
        workflow[nid] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": ln, "model": prev_m, "clip": prev_c, "strength_model": ls, "strength_clip": ls,
        }}
        prev_m, prev_c = [nid, 0], [nid, 1]
    if vae_name:
        workflow["11"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}}
    return workflow


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IP-Adapter Workflow (reference image style/face transfer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_ipadapter_workflow(
    prompt, negative_prompt="", model="",
    reference_image_path="", ip_weight=0.8,
    width=512, height=768, steps=25, cfg=7.0,
    sampler="euler_ancestral", scheduler="normal", seed=-1,
    loras=None, vae_name="",
    ip_model="ip-adapter_sd15.safetensors",
    clip_vision_model="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
):
    """Build IP-Adapter workflow — uses reference image to preserve style/face."""
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    lora_list = [(n, s) for n, s in (loras or []) if n and n not in ("None", "")]
    if lora_list:
        last_id = str(10 + (len(lora_list) - 1) * 2)
        model_out, clip_out = [last_id, 0], [last_id, 1]
    else:
        model_out, clip_out = ["4", 0], ["4", 1]
    vae_out = ["4", 2] if not vae_name else ["11", 0]

    workflow = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "50": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": ip_model}},
        "51": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision_model}},
        "52": {"class_type": "LoadImage", "inputs": {"image": reference_image_path}},
        "53": {
            "class_type": "IPAdapterApply",
            "inputs": {
                "ipadapter": ["50", 0], "clip_vision": ["51", 0],
                "image": ["52", 0], "model": model_out,
                "weight": ip_weight, "noise": 0.0, "weight_type": "original",
                "start_at": 0.0, "end_at": 1.0,
            },
        },
        "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": height, "width": width}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_out, "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_out, "text": negative_prompt}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg, "denoise": 1.0, "latent_image": ["5", 0],
                "model": ["53", 0], "negative": ["7", 0], "positive": ["6", 0],
                "sampler_name": sampler, "scheduler": scheduler, "seed": seed, "steps": steps,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": vae_out}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ipadapter", "images": ["8", 0]}},
    }
    prev_m, prev_c = ["4", 0], ["4", 1]
    for i, (ln, ls) in enumerate(lora_list):
        nid = str(10 + i * 2)
        workflow[nid] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": ln, "model": prev_m, "clip": prev_c, "strength_model": ls, "strength_clip": ls,
        }}
        prev_m, prev_c = [nid, 0], [nid, 1]
    if vae_name:
        workflow["11"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}}
    return workflow


# ──────────────────────────────────────────────
# Wan 2.2 NSFW Lightning (Enhanced SVI Camera Prompt Adherence)
# Two-stage MoE sampler: HIGH-noise → LOW-noise GGUF checkpoint pair
# Requires ComfyUI-GGUF custom node + umt5-xxl text encoder + wan 2.1 VAE
# ──────────────────────────────────────────────

WAN22_DEFAULTS = {
    "high_model": "wan22_nsfw_fm_v2_high_Q4_K_M.gguf",
    "low_model":  "wan22_nsfw_fm_v2_low_Q4_K_M.gguf",
    "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan_2.1_vae.safetensors",
    "clip_vision": "clip_vision_h.safetensors",
    "steps": 4,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "shift": 5.0,
    "frame_count": 81,
    "fps": 16,
    "width": 832,
    "height": 480,
}


def build_wan22_t2v_workflow(
    prompt, negative_prompt="",
    high_model=None, low_model=None,
    text_encoder=None, vae=None,
    width=None, height=None,
    frame_count=None, fps=None,
    steps=None, cfg=None,
    sampler=None, scheduler=None,
    seed=-1, shift=None,
):
    """Build Wan 2.2 T2V workflow (text-to-video, GGUF 2-stage HIGH->LOW MoE)."""
    import random
    if seed == -1 or seed is None:
        seed = random.randint(0, 2**31 - 1)
    d = WAN22_DEFAULTS
    high_model = high_model or d["high_model"]
    low_model = low_model or d["low_model"]
    text_encoder = text_encoder or d["text_encoder"]
    vae = vae or d["vae"]
    width = int(width or d["width"])
    height = int(height or d["height"])
    frame_count = int(frame_count or d["frame_count"])
    fps = int(fps or d["fps"])
    steps = int(steps or d["steps"])
    cfg = float(cfg or d["cfg"])
    sampler = sampler or d["sampler"]
    scheduler = scheduler or d["scheduler"]
    shift = float(shift if shift is not None else d["shift"])
    half = max(1, steps // 2)
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": high_model}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": low_model}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "wan", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": negative_prompt or ""}},
        "7": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": width, "height": height, "length": frame_count, "batch_size": 1}},
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": shift}},
        "9": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": shift}},
        "10": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["8", 0], "add_noise": "enable", "noise_seed": seed,
                "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
                "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
                "start_at_step": 0, "end_at_step": half,
                "return_with_leftover_noise": "enable",
            },
        },
        "11": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["9", 0], "add_noise": "disable", "noise_seed": seed,
                "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
                "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["10", 0],
                "start_at_step": half, "end_at_step": steps,
                "return_with_leftover_noise": "disable",
            },
        },
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "13": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["12", 0], "frame_rate": fps, "loop_count": 0,
                "filename_prefix": "wan22_t2v", "format": "video/h264-mp4",
                "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True,
                "pingpong": False, "save_output": True,
            },
        },
    }


def build_wan22_i2v_workflow(
    prompt, image_name, negative_prompt="",
    high_model=None, low_model=None,
    text_encoder=None, vae=None, clip_vision=None,
    width=None, height=None,
    frame_count=None, fps=None,
    steps=None, cfg=None,
    sampler=None, scheduler=None,
    seed=-1, shift=None,
):
    """Build Wan 2.2 I2V workflow (image-to-video, GGUF 2-stage HIGH->LOW)."""
    import random
    if seed == -1 or seed is None:
        seed = random.randint(0, 2**31 - 1)
    d = WAN22_DEFAULTS
    high_model = high_model or "wan22_nsfw_fm_v2_i2v_high_Q4_K_M.gguf"
    low_model = low_model or "wan22_nsfw_fm_v2_i2v_low_Q4_K_M.gguf"
    text_encoder = text_encoder or d["text_encoder"]
    vae = vae or d["vae"]
    clip_vision = clip_vision or d["clip_vision"]
    width = int(width or d["width"])
    height = int(height or d["height"])
    frame_count = int(frame_count or d["frame_count"])
    fps = int(fps or d["fps"])
    steps = int(steps or d["steps"])
    cfg = float(cfg or d["cfg"])
    sampler = sampler or d["sampler"]
    scheduler = scheduler or d["scheduler"]
    shift = float(shift if shift is not None else d["shift"])
    half = max(1, steps // 2)
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": high_model}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": low_model}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "wan", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": negative_prompt or ""}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "8": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision}},
        "9": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["8", 0], "image": ["7", 0], "crop": "none"}},
        "10": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["5", 0], "negative": ["6", 0], "vae": ["4", 0],
                "clip_vision_output": ["9", 0], "start_image": ["7", 0],
                "width": width, "height": height, "length": frame_count, "batch_size": 1,
            },
        },
        "11": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": shift}},
        "12": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": shift}},
        "13": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["11", 0], "add_noise": "enable", "noise_seed": seed,
                "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
                "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["10", 2],
                "start_at_step": 0, "end_at_step": half,
                "return_with_leftover_noise": "enable",
            },
        },
        "14": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["12", 0], "add_noise": "disable", "noise_seed": seed,
                "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
                "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["13", 0],
                "start_at_step": half, "end_at_step": steps,
                "return_with_leftover_noise": "disable",
            },
        },
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["4", 0]}},
        "16": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["15", 0], "frame_rate": fps, "loop_count": 0,
                "filename_prefix": "wan22_i2v", "format": "video/h264-mp4",
                "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True,
                "pingpong": False, "save_output": True,
            },
        },
    }
