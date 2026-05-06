"""Vast.ai Cloud GPU Manager - Search/create/manage instances with ComfyUI + CivitAI model support."""
import json
import urllib.request
import urllib.error
import time

VAST_API_BASE = "https://console.vast.ai/api/v0"

# ── ComfyUI Docker images for vast.ai ──
COMFYUI_IMAGES = {
    "ComfyUI (ai-dock, 推奨)": {
        "image": "ghcr.io/ai-dock/comfyui:latest-cuda",
        "desc": "ComfyUI + CUDA、カスタムノード簡単追加",
    },
    "ComfyUI (公式軽量)": {
        "image": "comfyanonymous/comfyui:latest",
        "desc": "公式イメージ、最小構成",
    },
}

# ── Recommended Realistic NSFW Models (CivitAI) ──
# Format: name -> {civitai_model_id, civitai_version_id, filename, type, size_gb, desc}
RECOMMENDED_MODELS = {
    # ━━ Checkpoints (SDXL / Pony) ━━
    "RealVisXL V5.0 (SDXL最強リアル)": {
        "url": "https://civitai.com/api/download/models/789646",
        "filename": "realvisxlV50.safetensors",
        "type": "checkpoint",
        "path": "models/checkpoints/",
        "size": "6.5GB",
        "desc": "SDXL系で最もリアルなフォトリアル。NSFW対応。",
    },
    "Juggernaut XL V9 (フォトリアル)": {
        "url": "https://civitai.com/api/download/models/456194",
        "filename": "juggernautXL_v9.safetensors",
        "type": "checkpoint",
        "path": "models/checkpoints/",
        "size": "6.9GB",
        "desc": "超リアル。ポートレート・ヌード両方得意。",
    },
    "epiCRealism XL (芸術的リアル)": {
        "url": "https://civitai.com/api/download/models/354959",
        "filename": "epicrealismXL.safetensors",
        "type": "checkpoint",
        "path": "models/checkpoints/",
        "size": "6.5GB",
        "desc": "芸術的でリアル。ファインアート風ヌードに最適。",
    },
    "CyberRealistic Pony (無検閲リアル)": {
        "url": "https://civitai.com/api/download/models/637041",
        "filename": "cyberrealisticPony.safetensors",
        "type": "checkpoint",
        "path": "models/checkpoints/",
        "size": "6.5GB",
        "desc": "Ponyベース。完全無検閲のリアル系。NSFW全開。",
    },
    "Pony Diffusion V6 XL (無検閲ベース)": {
        "url": "https://civitai.com/api/download/models/290640",
        "filename": "ponyDiffusionV6XL.safetensors",
        "type": "checkpoint",
        "path": "models/checkpoints/",
        "size": "6.5GB",
        "desc": "Pony系ベースモデル。NSFW LoRAとの組み合わせが豊富。",
    },
    "AutismMix SDXL (アニメ+リアル)": {
        "url": "https://civitai.com/api/download/models/324524",
        "filename": "autismmixSDXL.safetensors",
        "type": "checkpoint",
        "path": "models/checkpoints/",
        "size": "6.5GB",
        "desc": "アニメとリアルの中間。2.5Dスタイルが得意。",
    },
    # ━━ LoRA (NSFW特化) ━━
    "Detail Tweaker XL (ディテール強化)": {
        "url": "https://civitai.com/api/download/models/135867",
        "filename": "detailTweakerXL.safetensors",
        "type": "lora",
        "path": "models/loras/",
        "size": "0.4GB",
        "desc": "ディテール・肌質感向上。全モデル共通。",
    },
    "Skin Texture (肌テクスチャ)": {
        "url": "https://civitai.com/api/download/models/159384",
        "filename": "skinTexture_xl.safetensors",
        "type": "lora",
        "path": "models/loras/",
        "size": "0.2GB",
        "desc": "リアルな肌の毛穴・質感を追加。",
    },
    "NSFW Body XL (NSFW強化)": {
        "url": "https://civitai.com/api/download/models/274039",
        "filename": "nsfwBody_xl.safetensors",
        "type": "lora",
        "path": "models/loras/",
        "size": "0.3GB",
        "desc": "体のディテール・ポーズ精度向上。",
    },
    # ━━ VAE ━━
    "SDXL VAE (標準)": {
        "url": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
        "filename": "sdxl_vae.safetensors",
        "type": "vae",
        "path": "models/vae/",
        "size": "0.3GB",
        "desc": "SDXL標準VAE。色の正確性向上。",
    },
    # ━━ Upscaler ━━
    "4x-UltraSharp (アップスケーラー)": {
        "url": "https://civitai.com/api/download/models/125843",
        "filename": "4x-UltraSharp.pth",
        "type": "upscaler",
        "path": "models/upscale_models/",
        "size": "0.07GB",
        "desc": "最高品質のアップスケーラー。",
    },
}

# Starter pack: minimum set for realistic NSFW
STARTER_PACK = [
    "RealVisXL V5.0 (SDXL最強リアル)",
    "Detail Tweaker XL (ディテール強化)",
    "SDXL VAE (標準)",
    "4x-UltraSharp (アップスケーラー)",
]

# Full pack: all recommended models
FULL_PACK = list(RECOMMENDED_MODELS.keys())


class VastAIManager:
    def __init__(self, api_key=""):
        self.api_key = api_key
        self._instance_id = None
        self._comfyui_url = None

    def set_key(self, api_key):
        self.api_key = api_key

    def _request(self, method, path, data=None):
        """Make a request to the Vast.ai API."""
        if not self.api_key:
            raise ValueError("Vast.ai API Key が設定されていません")

        url = f"{VAST_API_BASE}{path}"
        body = json.dumps(data).encode("utf-8") if data else None

        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "AI-diffusion/1.0")

        try:
            resp = urllib.request.urlopen(req, timeout=30)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Vast.ai API Error ({e.code}): {body}")

    # ── Search GPU Offers ──
    def search_offers(self, min_gpu_ram=16, max_price=1.0, gpu_names=None, limit=20):
        """Search available GPU offers on Vast.ai."""
        filters = {
            "gpu_ram": {"gte": min_gpu_ram * 1024},  # MB
            "dph_total": {"lte": max_price},
            "rentable": {"eq": True},
            "verified": {"eq": True},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": limit,
        }
        if gpu_names:
            filters["gpu_name"] = {"in": gpu_names}

        result = self._request("POST", "/bundles/", filters)
        return result.get("offers", [])

    def format_offers(self, offers):
        """Format GPU offers for display."""
        if not offers:
            return "利用可能なGPUが見つかりません。価格上限を上げるか、後で再試行してください。"

        lines = [f"{'GPU':<30} {'VRAM':>6} {'$/hr':>6} {'信頼度':>6} {'DL速度':>8} {'ID':>10}"]
        lines.append("-" * 80)
        for o in offers[:20]:
            gpu = o.get("gpu_name", "?")
            vram = f"{o.get('gpu_ram', 0) / 1024:.0f}GB"
            price = f"${o.get('dph_total', 0):.3f}"
            rel = f"{o.get('reliability', 0) * 100:.0f}%"
            dl = f"{o.get('inet_down', 0):.0f}Mbps"
            oid = str(o.get("id", "?"))
            lines.append(f"{gpu:<30} {vram:>6} {price:>6} {rel:>6} {dl:>8} {oid:>10}")
        return "\n".join(lines)

    # ── Instance Management ──
    def get_instances(self):
        """List all instances."""
        result = self._request("GET", "/instances/")
        return result.get("instances", [])

    def get_instance(self, instance_id):
        """Get details of a specific instance."""
        result = self._request("GET", f"/instances/{instance_id}/")
        return result

    def create_instance(self, offer_id, image_key="ComfyUI (ai-dock, 推奨)",
                        disk_gb=50, label="AI-diffusion-ComfyUI",
                        model_pack="starter"):
        """Create a new instance from an offer."""
        image_info = COMFYUI_IMAGES.get(image_key, list(COMFYUI_IMAGES.values())[0])
        docker_image = image_info["image"]

        # Build onstart script to install models
        onstart_script = self._build_onstart_script(model_pack)

        data = {
            "image": docker_image,
            "disk": disk_gb,
            "runtype": "ssh_direc",
            "label": label,
            "env": {
                "-p 8188:8188": "1",
                "-p 22:22": "1",
            },
            "onstart": onstart_script,
        }

        result = self._request("PUT", f"/asks/{offer_id}/", data)
        instance_id = result.get("new_contract")
        if instance_id:
            self._instance_id = instance_id
        return result

    def stop_instance(self, instance_id):
        """Stop an instance."""
        return self._request("PUT", f"/instances/{instance_id}/", {"state": "stopped"})

    def start_instance(self, instance_id):
        """Start a stopped instance."""
        return self._request("PUT", f"/instances/{instance_id}/", {"state": "running"})

    def destroy_instance(self, instance_id):
        """Permanently destroy an instance."""
        return self._request("DELETE", f"/instances/{instance_id}/")

    # ── ComfyUI Connection ──
    def get_comfyui_url(self, instance=None):
        """Extract ComfyUI URL from instance info."""
        if instance is None and self._instance_id:
            instance = self.get_instance(self._instance_id)
        if not instance:
            return None

        # Try to find the public IP and port mapping
        public_ip = instance.get("public_ipaddr")
        ports = instance.get("ports", {})

        # Look for port 8188 mapping
        port_8188 = ports.get("8188/tcp")
        if port_8188 and isinstance(port_8188, list) and len(port_8188) > 0:
            host_port = port_8188[0].get("HostPort", "8188")
            host_ip = port_8188[0].get("HostIp", public_ip)
            return f"http://{host_ip}:{host_port}"

        # Fallback: direct IP
        if public_ip:
            direct_port = instance.get("direct_port_start", 8188)
            return f"http://{public_ip}:{direct_port}"

        return None

    def wait_for_ready(self, instance_id, timeout=300):
        """Wait for instance to be running and ComfyUI accessible."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                inst = self.get_instance(instance_id)
                status = inst.get("actual_status", "")
                if status == "running":
                    url = self.get_comfyui_url(inst)
                    if url:
                        try:
                            urllib.request.urlopen(f"{url}/system_stats", timeout=5)
                            self._comfyui_url = url
                            return url
                        except (urllib.error.URLError, TimeoutError, OSError):
                            pass
            except Exception:
                pass
            time.sleep(10)
        return None

    # ── Model Download Script ──
    def _build_onstart_script(self, model_pack="starter"):
        """Build startup script that downloads CivitAI models."""
        if model_pack == "starter":
            model_keys = STARTER_PACK
        elif model_pack == "full":
            model_keys = FULL_PACK
        else:
            model_keys = []

        lines = [
            "#!/bin/bash",
            "set -e",
            "",
            "# Wait for ComfyUI to be available",
            "COMFY_DIR=/workspace/ComfyUI",
            "if [ ! -d $COMFY_DIR ]; then",
            "  COMFY_DIR=/opt/ComfyUI",
            "fi",
            "if [ ! -d $COMFY_DIR ]; then",
            "  echo 'ComfyUI not found, skipping model download'",
            "  exit 0",
            "fi",
            "",
            "echo '=== Downloading CivitAI Models ==='",
            "",
        ]

        for key in model_keys:
            model = RECOMMENDED_MODELS.get(key)
            if not model:
                continue
            url = model["url"]
            path = f"$COMFY_DIR/{model['path']}"
            filename = model["filename"]
            lines.extend([
                f"# {key}",
                f"mkdir -p {path}",
                f"if [ ! -f {path}{filename} ]; then",
                f"  echo 'Downloading {key} ({model['size']})...'",
                f"  wget -q --show-progress -O {path}{filename} '{url}'",
                f"  echo 'Done: {filename}'",
                f"else",
                f"  echo 'Already exists: {filename}'",
                f"fi",
                "",
            ])

        # Install custom nodes
        lines.extend([
            "# Install custom nodes",
            "cd $COMFY_DIR/custom_nodes",
            "if [ ! -d ComfyUI-Impact-Pack ]; then",
            "  git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git 2>/dev/null || true",
            "fi",
            "",
            "echo '=== Model Download Complete ==='",
        ])

        return "\n".join(lines)

    def generate_download_script(self, model_keys):
        """Generate a download script for specific models (for SSH execution)."""
        lines = ["#!/bin/bash", "set -e", ""]
        lines.append("COMFY_DIR=${1:-/workspace/ComfyUI}")
        lines.append("")

        for key in model_keys:
            model = RECOMMENDED_MODELS.get(key)
            if not model:
                continue
            url = model["url"]
            path = f"$COMFY_DIR/{model['path']}"
            filename = model["filename"]
            lines.extend([
                f"# {key} ({model['size']})",
                f"mkdir -p {path}",
                f"if [ ! -f {path}{filename} ]; then",
                f"  echo 'Downloading {key}...'",
                f"  wget -q --show-progress -O {path}{filename} '{url}'",
                f"else",
                f"  echo 'Skip (exists): {filename}'",
                f"fi",
                "",
            ])
        return "\n".join(lines)

    def generate_civitai_download_command(self, civitai_url, model_type="checkpoint"):
        """Generate wget command for any CivitAI model URL."""
        type_to_path = {
            "checkpoint": "models/checkpoints/",
            "lora": "models/loras/",
            "vae": "models/vae/",
            "embedding": "models/embeddings/",
            "controlnet": "models/controlnet/",
            "upscaler": "models/upscale_models/",
        }
        path = type_to_path.get(model_type, "models/checkpoints/")
        return f"cd /workspace/ComfyUI/{path} && wget -q --show-progress --content-disposition '{civitai_url}'"


def format_instance_status(instance):
    """Format instance status for display."""
    if not instance:
        return "インスタンスが見つかりません"

    gpu = instance.get("gpu_name", "Unknown")
    status = instance.get("actual_status", "Unknown")
    label = instance.get("label", "")
    price = instance.get("dph_total", 0)
    uptime = instance.get("duration", 0)

    lines = [f"GPU: {gpu} | Status: {status}"]
    if label:
        lines.append(f"Label: {label}")
    lines.append(f"料金: ${price:.3f}/hr (約¥{int(price * 150)}/hr)")

    if uptime and uptime > 0:
        hours = uptime / 3600
        cost = hours * price
        lines.append(f"稼働: {hours:.1f}時間 | コスト: ${cost:.2f} (約¥{int(cost * 150)})")
        if hours > 2:
            lines.append("⚠ 2時間以上稼働中！不要なら停止してください。")

    # Connection info
    public_ip = instance.get("public_ipaddr")
    ssh_port = instance.get("ssh_port")
    if public_ip and ssh_port:
        lines.append(f"SSH: ssh -p {ssh_port} root@{public_ip}")

    return "\n".join(lines)


def format_cost_summary(instances):
    """Format cost summary for all running instances."""
    total_cost = 0
    running = 0
    for inst in instances:
        if inst.get("actual_status") == "running":
            running += 1
            hrs = (inst.get("duration", 0) or 0) / 3600
            total_cost += hrs * (inst.get("dph_total", 0) or 0)

    return (
        f"稼働中: {running}台 | 合計コスト: ${total_cost:.2f} (約¥{int(total_cost * 150)})\n"
        f"⚠ 使い終わったら必ず停止(Stop)してください。Destroyで完全削除。"
    )
