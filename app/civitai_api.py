"""CivitAI API Client - Model search, download, image upload, generation, and training."""
import json
import os
import urllib.request
import urllib.error
import urllib.parse
import time
import threading

CIVITAI_API_URL = "https://civitai.com/api/v1"
CIVITAI_ORCHESTRATION_URL = "https://orchestration.civitai.com"

# ── iCloud model → CivitAI URN cache ──
_URN_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icloud_urn_cache.json")
_urn_cache = {}


def _load_urn_cache():
    global _urn_cache
    if os.path.exists(_URN_CACHE_FILE):
        try:
            with open(_URN_CACHE_FILE, "r") as f:
                _urn_cache = json.load(f)
        except Exception:
            _urn_cache = {}


def _save_urn_cache():
    with open(_URN_CACHE_FILE, "w") as f:
        json.dump(_urn_cache, f, indent=2, ensure_ascii=False)


def _base_model_to_urn_prefix(base_model_str):
    """Map CivitAI baseModel string to URN prefix (sdxl, sd1, flux1, etc.)."""
    bm = (base_model_str or "").lower()
    if "flux" in bm:
        return "flux1"
    if "sdxl" in bm or "pony" in bm or "illustrious" in bm:
        return "sdxl"
    if "sd 1" in bm or "sd1" in bm:
        return "sd1"
    if "sd 2" in bm or "sd2" in bm:
        return "sd2"
    return "sdxl"


def _base_model_to_api_format(base_model_str):
    """Map CivitAI baseModel string to Orchestration API format (SDXL, SD_1_5, etc.)."""
    bm = (base_model_str or "").lower()
    if "flux" in bm:
        return "Flux1"
    if "sdxl" in bm or "pony" in bm or "illustrious" in bm:
        return "SDXL"
    if "sd 1" in bm or "sd1" in bm:
        return "SD_1_5"
    if "sd 2" in bm or "sd2" in bm:
        return "SD_2_1"
    return "SDXL"


# Load cache on import
_load_urn_cache()

# Popular NSFW-capable models (pre-configured for easy generation)
CIVITAI_GENERATION_MODELS = {
    # ── SDXL系 (高品質・1024px推奨) ──
    "Juggernaut XL Ragnarok (SDXL 最高品質)": {
        "urn": "urn:air:sdxl:checkpoint:civitai:133005@1759168",
        "base": "SDXL",
        "cost": "~4 Buzz",
    },
    "epiCRealism XL (SDXL フォトリアル)": {
        "urn": "urn:air:sdxl:checkpoint:civitai:25694@2514955",
        "base": "SDXL",
        "cost": "~4 Buzz",
    },
    "RealVisXL V5.0 Lightning (SDXL 高速リアル)": {
        "urn": "urn:air:sdxl:checkpoint:civitai:139562@798204",
        "base": "SDXL",
        "cost": "~4 Buzz",
    },
    "DreamShaper XL Lightning (SDXL 万能)": {
        "urn": "urn:air:sdxl:checkpoint:civitai:4384@354657",
        "base": "SDXL",
        "cost": "~4 Buzz",
    },
    "Anything XL (SDXL アニメ)": {
        "urn": "urn:air:sdxl:checkpoint:civitai:9409@384264",
        "base": "SDXL",
        "cost": "~4 Buzz",
    },
    # ── Pony系 (NSFW特化・高品質) ──
    "Pony Diffusion V6 XL (NSFW アニメ)": {
        "urn": "urn:air:sdxl:checkpoint:civitai:257749@290640",
        "base": "SDXL",
        "cost": "~4 Buzz",
    },
    "CyberRealistic Pony (NSFW リアル)": {
        "urn": "urn:air:sdxl:checkpoint:civitai:443821@2727742",
        "base": "SDXL",
        "cost": "~4 Buzz",
    },
    "Pony Realism (NSFW フォトリアル)": {
        "urn": "urn:air:sdxl:checkpoint:civitai:372465@914390",
        "base": "SDXL",
        "cost": "~4 Buzz",
    },
    # ── SD1.5系 (軽量・安い) ──
    "DreamShaper 8 (SD1.5 万能)": {
        "urn": "urn:air:sd1:checkpoint:civitai:4384@128713",
        "base": "SD_1_5",
        "cost": "~1 Buzz",
    },
    "MeinaMix V12 (SD1.5 アニメ)": {
        "urn": "urn:air:sd1:checkpoint:civitai:7240@948574",
        "base": "SD_1_5",
        "cost": "~1 Buzz",
    },
}


class CivitAIClient:
    def __init__(self, api_key=""):
        self.api_key = api_key

    def _headers(self, content_type=None):
        h = {"User-Agent": "AI-diffusion/1.0"}
        if content_type:
            h["Content-Type"] = content_type
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _get(self, endpoint, params=None):
        url = f"{CIVITAI_API_URL}{endpoint}"
        if params is None:
            params = {}
        if self.api_key:
            params["token"] = self.api_key
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "AI-diffusion/1.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())

    def _post_multipart(self, url, fields, files):
        """Post multipart form data (for image uploads)."""
        boundary = "----AI-diffusion-boundary"
        body = b""
        for key, value in fields.items():
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
            body += f"{value}\r\n".encode()
        for key, (filename, data, content_type) in files.items():
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode()
            body += f"Content-Type: {content_type}\r\n\r\n".encode()
            body += data + b"\r\n"
        body += f"--{boundary}--\r\n".encode()

        headers = self._headers(content_type=f"multipart/form-data; boundary={boundary}")
        req = urllib.request.Request(url, data=body, headers=headers)
        resp = urllib.request.urlopen(req, timeout=60)
        return json.loads(resp.read())

    # ── Model Search ──

    def search_models(self, query="", model_type="Checkpoint", sort="Highest Rated",
                      nsfw=True, limit=20):
        """Search CivitAI models."""
        params = {"limit": limit, "sort": sort, "nsfw": "true"}
        if query:
            params["query"] = query
        if model_type:
            params["types"] = model_type
        if not nsfw:
            params["nsfw"] = "false"
        return self._get("/models", params)

    def get_model(self, model_id):
        """Get model details by ID."""
        return self._get(f"/models/{model_id}")

    def get_model_version(self, version_id):
        """Get specific model version details."""
        return self._get(f"/model-versions/{version_id}")

    # ── Model Download ──

    def get_download_url(self, version_id):
        """Get download URL for a model version."""
        url = f"{CIVITAI_API_URL}/model-versions/{version_id}"
        data = self._get(f"/model-versions/{version_id}")
        if data.get("files"):
            primary = data["files"][0]
            dl_url = primary.get("downloadUrl", "")
            if self.api_key and dl_url:
                dl_url += f"&token={self.api_key}" if "?" in dl_url else f"?token={self.api_key}"
            return {
                "url": dl_url,
                "filename": primary.get("name", "model.safetensors"),
                "size_kb": primary.get("sizeKB", 0),
                "type": primary.get("type", "Model"),
            }
        return None

    def download_model(self, version_id, dest_dir, progress_callback=None):
        """Download a model file to dest_dir. Returns filepath."""
        info = self.get_download_url(version_id)
        if not info:
            raise ValueError("Download URL not found")

        filepath = os.path.join(dest_dir, info["filename"])
        url = info["url"]

        req = urllib.request.Request(url, headers={"User-Agent": "AI-diffusion/1.0"})
        resp = urllib.request.urlopen(req, timeout=600)
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(filepath, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total > 0:
                    progress_callback(downloaded / total)

        return filepath

    # ── Image Upload ──

    def upload_image(self, image_path, meta=None):
        """Upload an image to CivitAI."""
        if not self.api_key:
            raise ValueError("API Key required for upload")

        with open(image_path, "rb") as f:
            image_data = f.read()

        filename = os.path.basename(image_path)
        ext = filename.rsplit(".", 1)[-1].lower()
        content_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")

        fields = {}
        if meta:
            fields["meta"] = json.dumps(meta)

        files = {"file": (filename, image_data, content_type)}

        return self._post_multipart(f"{CIVITAI_API_URL}/images", fields, files)

    def create_post(self, title, description="", image_ids=None, model_version_id=None, nsfw=False):
        """Create a post on CivitAI."""
        if not self.api_key:
            raise ValueError("API Key required")

        payload = {"title": title}
        if description:
            payload["description"] = description
        if image_ids:
            payload["imageIds"] = image_ids
        if model_version_id:
            payload["modelVersionId"] = model_version_id
        if nsfw:
            payload["nsfw"] = True

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{CIVITAI_API_URL}/posts",
            data=data,
            headers=self._headers(content_type="application/json"),
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())

    # ── Image Generation (Cloud GPU) ──

    def _orchestration_request(self, method, path, data=None):
        """Make request to CivitAI Orchestration API."""
        if not self.api_key:
            raise ValueError("CivitAI API Key が設定されていません")
        import httpx
        url = f"{CIVITAI_ORCHESTRATION_URL}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=300) as client:
            if method == "POST":
                resp = client.post(url, json=data, headers=headers)
            else:
                resp = client.get(url, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"CivitAI Generation Error ({resp.status_code}): {resp.text}")
        return resp.json()

    def generate_image(self, model_key, prompt, negative_prompt="",
                       width=512, height=768, steps=25, cfg_scale=7,
                       scheduler="EulerA", seed=-1, clip_skip=2, quantity=1,
                       lora_urns=None):
        """Generate image via CivitAI Cloud GPU. Returns list of image URLs.

        lora_urns: list of (urn_string, strength) tuples for additionalNetworks.
        """
        model_info = CIVITAI_GENERATION_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"不明なモデル: {model_key}")

        model_urn = model_info["urn"]
        base_model = model_info["base"]

        # Clamp dimensions for API limits
        w = min(max(64, width), 1024)
        h = min(max(64, height), 1024)

        job_input = {
            "$type": "textToImage",
            "baseModel": base_model,
            "model": model_urn,
            "params": {
                "prompt": prompt,
                "negativePrompt": negative_prompt or "(worst quality, low quality, blurry:1.3)",
                "scheduler": scheduler,
                "steps": steps,
                "cfgScale": cfg_scale,
                "width": w,
                "height": h,
                "seed": seed if seed >= 0 else -1,
                "clipSkip": clip_skip,
            },
            "quantity": quantity,
        }

        # LoRA support via additionalNetworks
        if lora_urns:
            networks = {}
            for lora_urn, strength in lora_urns:
                networks[lora_urn] = {"type": "Lora", "strength": float(strength)}
            job_input["additionalNetworks"] = networks

        # Submit job
        result = self._orchestration_request("POST", "/v1/consumer/jobs", job_input)
        token = result.get("token", "")

        if not token:
            raise RuntimeError("生成ジョブの送信に失敗しました")
        return self._poll_generation(token, timeout=300)

    def _extract_blob_urls(self, jobs):
        """Extract completed blobUrls from job results."""
        urls = []
        all_done = True
        for job in jobs:
            results = job.get("result", [])
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict) and r.get("available") and r.get("blobUrl"):
                        urls.append(r["blobUrl"])
                    elif isinstance(r, dict) and not r.get("available"):
                        all_done = False
            elif isinstance(results, dict):
                if results.get("available") and results.get("blobUrl"):
                    urls.append(results["blobUrl"])
                elif not results.get("available"):
                    all_done = False
        return urls, all_done

    def _poll_generation(self, token, timeout=300):
        """Poll for generation job completion."""
        import httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        start = time.time()
        while time.time() - start < timeout:
            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    f"{CIVITAI_ORCHESTRATION_URL}/v1/consumer/jobs",
                    params={"token": token},
                    headers=headers,
                )
            if resp.status_code != 200:
                time.sleep(2)
                continue

            data = resp.json()
            jobs = data.get("jobs", [])
            urls, all_done = self._extract_blob_urls(jobs)

            if urls and all_done:
                return urls
            time.sleep(2)

        raise TimeoutError(f"CivitAI生成が{timeout}秒以内に完了しませんでした")

    def generate_image_by_urn(self, model_urn, base_model, prompt, negative_prompt="",
                               width=512, height=768, steps=25, cfg_scale=7,
                               scheduler="EulerA", seed=-1, quantity=1,
                               lora_urns=None):
        """Generate using a custom model URN (for search results).

        lora_urns: list of (urn_string, strength) tuples for additionalNetworks.
        """
        w = min(max(64, width), 1024)
        h = min(max(64, height), 1024)
        job_input = {
            "$type": "textToImage",
            "baseModel": base_model,
            "model": model_urn,
            "params": {
                "prompt": prompt,
                "negativePrompt": negative_prompt or "(worst quality, low quality:1.3)",
                "scheduler": scheduler,
                "steps": steps,
                "cfgScale": cfg_scale,
                "width": w,
                "height": h,
                "seed": seed if seed >= 0 else -1,
            },
            "quantity": quantity,
        }

        # LoRA support via additionalNetworks
        if lora_urns:
            networks = {}
            for lora_urn, strength in lora_urns:
                networks[lora_urn] = {"type": "Lora", "strength": float(strength)}
            job_input["additionalNetworks"] = networks

        result = self._orchestration_request("POST", "/v1/consumer/jobs", job_input)
        token = result.get("token", "")
        if not token:
            raise RuntimeError("生成ジョブの送信に失敗しました")
        return self._poll_generation(token, timeout=300)

    # ── Training ──

    def get_training_status(self, model_version_id):
        """Check training status for a model version."""
        return self._get(f"/model-versions/{model_version_id}/training")

    def get_training_cost_estimate(self):
        """Get current training pricing info."""
        return {
            "info": "CivitAI Training uses Buzz (CivitAI currency).",
            "estimate_sd15": "~500-1000 Buzz for SD1.5 LoRA training",
            "estimate_sdxl": "~1000-2000 Buzz for SDXL LoRA training",
            "note": "Training is done on CivitAI's servers (GPU not needed locally).",
            "url": "https://civitai.com/models/train",
        }


    # ── Vault (cloud model storage for Supporter+) ──

    def _trpc_query(self, procedure, input_data=None):
        """Call CivitAI tRPC query endpoint."""
        if not self.api_key:
            raise ValueError("CivitAI API Key が必要です (Vault にはログインが必要)")
        url = f"https://civitai.com/api/trpc/{procedure}"
        if input_data is not None:
            input_json = json.dumps({"json": input_data})
            url += f"?input={urllib.parse.quote(input_json)}"
        req = urllib.request.Request(url, headers=self._headers())
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        return data.get("result", {}).get("data", {}).get("json", data)

    def _trpc_mutation(self, procedure, input_data):
        """Call CivitAI tRPC mutation endpoint."""
        if not self.api_key:
            raise ValueError("CivitAI API Key が必要です")
        url = f"https://civitai.com/api/trpc/{procedure}"
        body = json.dumps({"json": input_data}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers=self._headers(content_type="application/json"),
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        return data.get("result", {}).get("data", {}).get("json", data)

    def vault_get(self):
        """Get vault info (storage used, limit, etc.)."""
        return self._trpc_query("vault.get")

    def vault_list(self, limit=60, query="", page=1):
        """List items in vault with optional search."""
        params = {"limit": limit, "page": page}
        if query:
            params["query"] = query
        return self._trpc_query("vault.getItemsPaged", params)

    def vault_add(self, model_version_id):
        """Add a model version to vault."""
        return self._trpc_mutation("vault.toggleModelVersion", {"modelVersionId": model_version_id})

    def vault_remove(self, model_version_ids):
        """Remove model versions from vault."""
        return self._trpc_mutation("vault.removeItemsFromVault", {"modelVersionIds": model_version_ids})

    def vault_download_url(self, vault_item):
        """Extract download URL from a vault item's files list."""
        files = vault_item.get("files", [])
        if files:
            return files[0].get("url", "")
        return ""

    # ── iCloud model URN resolution ──

    def resolve_icloud_model_urn(self, filename):
        """Resolve an iCloud model filename to a CivitAI URN.

        Searches CivitAI by filename, matches exactly, caches the result.
        Returns dict with urn, base, type, name, version, cost or None.
        """
        global _urn_cache
        if filename in _urn_cache:
            return _urn_cache[filename]

        # Build search queries from filename (strip extension, try variants)
        base_name = filename.rsplit(".", 1)[0]
        # Remove common suffixes for better search
        search_queries = [base_name]
        # Try shorter variants (e.g., "realismIllustriousBy_v55FP16" → "realismIllustrious")
        for sep in ["_v", "_V", "-v", "-V", "FP16", "FP32", "fp16", "fp32"]:
            if sep in base_name:
                shorter = base_name.split(sep)[0]
                if len(shorter) >= 4:
                    search_queries.append(shorter)

        for query in search_queries:
            try:
                results = self.search_models(query=query, limit=20)
                items = results.get("items", [])
                for model in items:
                    for version in model.get("modelVersions", []):
                        for file_info in version.get("files", []):
                            if file_info.get("name") == filename:
                                # Exact filename match found
                                model_id = model["id"]
                                version_id = version["id"]
                                raw_base = version.get("baseModel", "SDXL 1.0")
                                urn_prefix = _base_model_to_urn_prefix(raw_base)
                                api_base = _base_model_to_api_format(raw_base)
                                model_type = model.get("type", "Checkpoint").lower()
                                type_key = "lora" if "lora" in model_type else "checkpoint"
                                urn = f"urn:air:{urn_prefix}:{type_key}:civitai:{model_id}@{version_id}"

                                info = {
                                    "urn": urn,
                                    "base": api_base,
                                    "type": model.get("type", "Checkpoint"),
                                    "name": model.get("name", base_name),
                                    "version": version.get("name", ""),
                                    "cost": "~4 Buzz" if api_base == "SDXL" else "~1 Buzz",
                                }
                                _urn_cache[filename] = info
                                _save_urn_cache()
                                return info
            except Exception:
                continue

        # Not found - cache as None to avoid repeated lookups
        _urn_cache[filename] = None
        _save_urn_cache()
        return None

    def clear_urn_cache(self, filename=None):
        """Clear URN cache. If filename given, clear only that entry."""
        global _urn_cache
        if filename:
            _urn_cache.pop(filename, None)
        else:
            _urn_cache.clear()
        _save_urn_cache()


def format_model_result(model):
    """Format a model search result for display."""
    name = model.get("name", "Unknown")
    model_type = model.get("type", "")
    stats = model.get("stats", {})
    downloads = stats.get("downloadCount", 0)
    rating = stats.get("rating", 0)
    nsfw = model.get("nsfw", False)

    versions = model.get("modelVersions", [])
    latest = versions[0] if versions else {}
    version_name = latest.get("name", "")
    version_id = latest.get("id", "")

    files = latest.get("files", [])
    size_gb = files[0].get("sizeKB", 0) / 1024 / 1024 if files else 0

    nsfw_tag = " [NSFW]" if nsfw else ""
    return (
        f"{name} ({version_name}){nsfw_tag}\n"
        f"  Type: {model_type} | Size: {size_gb:.1f}GB | DL: {downloads:,} | Rating: {rating:.1f}\n"
        f"  Version ID: {version_id}"
    )


def format_search_results(data):
    """Format search results for display."""
    items = data.get("items", [])
    if not items:
        return "モデルが見つかりませんでした"
    lines = []
    for i, model in enumerate(items, 1):
        lines.append(f"[{i}] {format_model_result(model)}")
    meta = data.get("metadata", {})
    total = meta.get("totalItems", "?")
    page = meta.get("currentPage", 1)
    lines.append(f"\n--- Page {page} | Total: {total} models ---")
    return "\n".join(lines)
