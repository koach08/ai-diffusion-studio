"""RunPod Cloud GPU Manager - Start/stop/manage RunPod instances with ComfyUI."""
import json
import urllib.request
import urllib.error
import time

RUNPOD_API_URL = "https://api.runpod.io/graphql"

# ComfyUI template on RunPod
COMFYUI_TEMPLATE_ID = "runpod-comfyui"


class RunPodManager:
    def __init__(self, api_key=""):
        self.api_key = api_key

    def _graphql(self, query, variables=None):
        if not self.api_key:
            raise ValueError("RunPod API Key が設定されていません")
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            RUNPOD_API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "AI-diffusion/1.0",
            },
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        if "errors" in result:
            raise RuntimeError(f"RunPod API Error: {result['errors']}")
        return result["data"]

    def get_pods(self):
        query = """
        query {
            myself {
                pods {
                    id
                    name
                    desiredStatus
                    runtime { uptimeInSeconds gpus { id gpuUtilPercent memoryUtilPercent } ports { ip isIpPublic privatePort publicPort type } }
                    machine { gpuDisplayName }
                    imageName
                }
            }
        }
        """
        data = self._graphql(query)
        return data["myself"]["pods"]

    def get_pod(self, pod_id):
        pods = self.get_pods()
        for pod in pods:
            if pod["id"] == pod_id:
                return pod
        return None

    def get_comfyui_url(self, pod):
        """Extract ComfyUI URL from pod runtime info."""
        if not pod or not pod.get("runtime") or not pod["runtime"].get("ports"):
            return None
        for port_info in pod["runtime"]["ports"]:
            if port_info["privatePort"] == 8188 and port_info.get("ip"):
                public_port = port_info.get("publicPort", 8188)
                ip = port_info["ip"]
                proto = "https" if port_info.get("type") == "https" else "http"
                return f"{proto}://{ip}:{public_port}"
        # Fallback: RunPod proxy URL
        return f"https://{pod['id']}-8188.proxy.runpod.net"

    # GPU fallback order: コスパ順、16GB以上VRAM
    GPU_FALLBACK_ORDER = [
        "NVIDIA RTX A5000",                    # 24GB $0.16/hr
        "NVIDIA RTX A4000",                    # 16GB $0.17/hr
        "NVIDIA RTX 4000 SFF Ada Generation",  # 20GB $0.18/hr
        "NVIDIA RTX A4500",                    # 20GB $0.19/hr
        "NVIDIA RTX 4000 Ada Generation",      # 20GB $0.20/hr
        "NVIDIA A30",                          # 24GB $0.22/hr
        "NVIDIA GeForce RTX 3090",             # 24GB $0.22/hr
        "NVIDIA GeForce RTX 3090 Ti",          # 24GB $0.27/hr
        "NVIDIA GeForce RTX 4080",             # 16GB $0.27/hr
        "NVIDIA GeForce RTX 4080 SUPER",       # 16GB $0.28/hr
        "NVIDIA RTX A6000",                    # 48GB $0.33/hr
        "NVIDIA GeForce RTX 4090",             # 24GB $0.34/hr
        "NVIDIA A40",                          # 48GB $0.35/hr
        "NVIDIA GeForce RTX 5080",             # 16GB $0.39/hr
        "NVIDIA L4",                           # 24GB $0.44/hr
    ]

    def check_gpu_availability(self):
        """Check which GPUs are currently available on RunPod."""
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                communityPrice
                securePrice
                communitySpotPrice
            }
        }
        """
        data = self._graphql(query)
        gpus = data["gpuTypes"]
        available = []
        for g in gpus:
            if g.get("memoryInGb", 0) >= 16 and (g.get("communityPrice") or g.get("securePrice")):
                available.append(g)
        return sorted(available, key=lambda g: g.get("communityPrice") or g.get("securePrice") or 999)

    # RunPod template options
    TEMPLATES = {
        "ComfyUI + Flux (推奨)": {"id": "rzg5z3pls5", "desc": "Flux.1 dev プリインストール済み"},
        "ComfyUI (公式)": {"id": "cw3nka7d08", "desc": "基本ComfyUI、モデルは自分でDL"},
        "SD + ComfyUI": {"id": "yfe4bnwbz9", "desc": "Stable Diffusion + ComfyUI"},
        "ComfyUI 軽量版": {"id": "aomdggbx0y", "desc": "Flux無し、軽量起動"},
    }

    def create_pod(self, name="AI-diffusion-ComfyUI", gpu_type_id="NVIDIA RTX A5000",
                   volume_size=20, auto_fallback=True, template_key="ComfyUI + Flux (推奨)"):
        query = """
        mutation ($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                desiredStatus
                imageName
                machine { gpuDisplayName }
            }
        }
        """

        # Build list of GPUs to try
        if auto_fallback:
            gpu_list = [gpu_type_id] + [g for g in self.GPU_FALLBACK_ORDER if g != gpu_type_id]
        else:
            gpu_list = [gpu_type_id]

        # Get template ID
        template_info = self.TEMPLATES.get(template_key, self.TEMPLATES["ComfyUI + Flux (推奨)"])
        template_id = template_info["id"]

        last_error = None
        tried = []
        for gpu in gpu_list:
            pod_input = {
                "name": name,
                "templateId": template_id,
                "gpuTypeId": gpu,
                "cloudType": "ALL",
                "volumeInGb": volume_size,
                "containerDiskInGb": 10,
                "minVcpuCount": 2,
                "minMemoryInGb": 16,
                "ports": "8188/http,22/tcp",
                "volumeMountPath": "/workspace",
            }
            variables = {"input": pod_input}
            try:
                data = self._graphql(query, variables)
                return data["podFindAndDeployOnDemand"]
            except RuntimeError as e:
                last_error = e
                if "SUPPLY_CONSTRAINT" in str(e):
                    tried.append(gpu)
                    continue
                raise

        raise RuntimeError(
            f"全GPU在庫切れ（{len(set(tried))}種類試行）。\n"
            f"RunPodが混雑しています。数分待って再試行してください。"
        )

    def start_pod(self, pod_id):
        query = """
        mutation ($input: PodResumeInput!) {
            podResume(input: $input) {
                id
                desiredStatus
            }
        }
        """
        variables = {"input": {"podId": pod_id, "gpuCount": 1}}
        data = self._graphql(query, variables)
        return data["podResume"]

    def stop_pod(self, pod_id):
        query = """
        mutation ($input: PodStopInput!) {
            podStop(input: $input) {
                id
                desiredStatus
            }
        }
        """
        variables = {"input": {"podId": pod_id}}
        data = self._graphql(query, variables)
        return data["podStop"]

    def terminate_pod(self, pod_id):
        query = """
        mutation ($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        variables = {"input": {"podId": pod_id}}
        self._graphql(query, variables)
        return True

    def get_gpu_types(self):
        query = """
        query {
            gpuTypes {
                id
                displayName
                memoryInGb
                communityPrice
                securePrice
            }
        }
        """
        data = self._graphql(query)
        gpus = data["gpuTypes"]
        # Filter to relevant GPUs and sort by price
        relevant = [g for g in gpus if g.get("communityPrice") and g["communityPrice"] > 0]
        return sorted(relevant, key=lambda g: g["communityPrice"])

    def wait_for_ready(self, pod_id, timeout=300):
        """Wait for pod to be running and ComfyUI to be accessible."""
        start = time.time()
        while time.time() - start < timeout:
            pod = self.get_pod(pod_id)
            if pod and pod["desiredStatus"] == "RUNNING" and pod.get("runtime"):
                url = self.get_comfyui_url(pod)
                if url:
                    # Check if ComfyUI is actually responding
                    try:
                        urllib.request.urlopen(f"{url}/system_stats", timeout=5)
                        return url
                    except (urllib.error.URLError, TimeoutError):
                        pass
            time.sleep(5)
        return None


def format_pod_status(pod):
    if not pod:
        return "Pod が見つかりません"
    gpu = pod.get("machine", {}).get("gpuDisplayName", "Unknown")
    status = pod.get("desiredStatus", "Unknown")
    uptime = ""
    if pod.get("runtime") and pod["runtime"].get("uptimeInSeconds"):
        mins = pod["runtime"]["uptimeInSeconds"] // 60
        uptime = f" (稼働: {mins}分)"
    return f"Pod: {pod['name']} | GPU: {gpu} | Status: {status}{uptime}"


# GPU hourly rates (community pricing, approximate)
GPU_HOURLY_RATES = {
    "NVIDIA RTX A5000": 0.16,
    "NVIDIA RTX A4000": 0.17,
    "NVIDIA RTX 4000 SFF Ada Generation": 0.18,
    "NVIDIA RTX A4500": 0.19,
    "NVIDIA RTX 4000 Ada Generation": 0.20,
    "NVIDIA A30": 0.22,
    "NVIDIA GeForce RTX 3090": 0.22,
    "NVIDIA GeForce RTX 3090 Ti": 0.27,
    "NVIDIA GeForce RTX 4080": 0.27,
    "NVIDIA GeForce RTX 4080 SUPER": 0.28,
    "NVIDIA RTX A6000": 0.33,
    "NVIDIA GeForce RTX 4090": 0.34,
    "NVIDIA A40": 0.35,
    "NVIDIA GeForce RTX 5080": 0.39,
    "NVIDIA L4": 0.44,
}


def format_pod_cost(pod):
    """Format pod status with cost information."""
    if not pod:
        return "Pod が見つかりません"
    gpu = pod.get("machine", {}).get("gpuDisplayName", "Unknown")
    status = pod.get("desiredStatus", "Unknown")
    rate = GPU_HOURLY_RATES.get(gpu, 0.30)

    lines = [f"GPU: {gpu} | Status: {status}"]
    lines.append(f"料金: ${rate:.2f}/時間 (約¥{int(rate * 150)}/時間)")

    if pod.get("runtime") and pod["runtime"].get("uptimeInSeconds"):
        secs = pod["runtime"]["uptimeInSeconds"]
        hours = secs / 3600
        mins = secs // 60
        cost = hours * rate
        lines.append(f"稼働時間: {int(mins)}分 ({hours:.1f}時間)")
        lines.append(f"今回のコスト: ${cost:.3f} (約¥{int(cost * 150)})")
        if hours > 1:
            lines.append(f"⚠ 1時間以上稼働中！不要なら停止してください。")
    elif status == "RUNNING":
        lines.append("稼働時間: 起動直後")

    return "\n".join(lines)
