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
    def __init__(self, server_url="http://127.0.0.1:8188"):
        self.server_url = server_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def is_server_running(self):
        try:
            urllib.request.urlopen(f"{self.server_url}/system_stats", timeout=3)
            return True
        except (urllib.error.URLError, TimeoutError):
            return False

    def queue_prompt(self, workflow):
        """Queue a workflow prompt and return the prompt_id."""
        data = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read())["prompt_id"]

    def get_history(self, prompt_id):
        resp = urllib.request.urlopen(f"{self.server_url}/history/{prompt_id}")
        return json.loads(resp.read())

    def get_image(self, filename, subfolder="", folder_type="output"):
        params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        resp = urllib.request.urlopen(f"{self.server_url}/view?{params}")
        return resp.read()

    def get_queue_status(self):
        """Check ComfyUI queue status."""
        try:
            resp = urllib.request.urlopen(f"{self.server_url}/queue", timeout=5)
            return json.loads(resp.read())
        except Exception:
            return {}

    def wait_for_result(self, prompt_id, timeout=300):
        """Poll until the prompt completes, return list of image bytes."""
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
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
):
    """Build a ComfyUI txt2img workflow JSON. With optional Hires Fix."""
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    model_out = ["4", 0] if not lora_name else ["10", 0]
    clip_out = ["4", 1] if not lora_name else ["10", 1]
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
    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "AI-diffusion", "images": ["8", 0]},
    }

    if lora_name:
        workflow["10"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora_name,
                "model": ["4", 0],
                "clip": ["4", 1],
                "strength_clip": lora_strength,
                "strength_model": lora_strength,
            },
        }

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
):
    """Build AnimateDiff txt2vid workflow for ComfyUI."""
    if seed == -1:
        import random
        seed = random.randint(0, 2**63)

    model_source = "4"
    clip_source = "4"

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
                "model": ["4", 0] if not lora_name else ["10", 0],
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
                "clip": ["4", 1] if not lora_name else ["10", 1],
                "text": prompt,
            },
        },
        # Negative prompt
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1] if not lora_name else ["10", 1],
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

    if lora_name:
        workflow["10"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora_name,
                "model": ["4", 0],
                "clip": ["4", 1],
                "strength_clip": lora_strength,
                "strength_model": lora_strength,
            },
        }

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
