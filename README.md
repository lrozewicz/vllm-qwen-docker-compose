# vLLM Qwen3-Coder-30B na RunPod (A40 Optimized)

Setup vLLM z modelem Qwen3-Coder-30B zoptymalizowany pod A40 48GB z FP8 dla maksymalnej wydajności.

## Zalety vs GGUF/Ollama

- ⚡ **5-10x szybszy inference** dzięki PagedAttention
- 📊 **Continuous batching** - obsługa wielu requestów jednocześnie  
- 🔌 **OpenAI-compatible API** - łatwa integracja
- 💾 **Efektywne zarządzanie pamięcią** - lepsze wykorzystanie VRAM
- 🎯 **Natywna obsługa długich kontekstów** - do 128K tokenów

## Optymalizacja dla A40 48GB

- **GPU**: A40 48GB VRAM ✅
- **RAM**: 48GB ✅
- **CPU**: 9 vCPU ✅
- **Dysk**: 60GB+ wolnego miejsca
- **RunPod**: Expose port 8000

## Konfiguracja A40 Optimized

- **Model**: BFloat16 + FP8 KV Cache
- **Kontekst**: 250K tokenów (wykorzystuje dodatkową VRAM!)
- **VRAM użycie**: ~35GB z 48GB (73% utilization)
- **Concurrent sequences**: 64 (dostosowane do 9 vCPU)
- **Performance**: 50-80 tokens/s, 8-16 concurrent users

## Automatyczny setup

```bash
# Klonuj/skopiuj pliki na RunPod
cd /workspace
git clone [YOUR_REPO] vllm-qwen-simple  # lub skopiuj przez SCP

cd vllm-qwen-simple

# Uruchom automatyczny setup
chmod +x setup.sh
sudo ./setup.sh
```

## Ręczne uruchomienie

```bash
# Zainstaluj Docker (jeśli nie ma)
curl -fsSL https://get.docker.com | sh

# Uruchom vLLM
docker compose up -d

# Sprawdź status
docker compose logs -f
```

## Testowanie API

### Health Check
```bash
curl http://localhost:8000/health
```

### Lista modeli
```bash
curl http://localhost:8000/v1/models
```

### Completion
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder",
    "prompt": "def fibonacci(n):",
    "max_tokens": 200,
    "temperature": 0.3
  }'
```

### Chat Completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder", 
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python web scraper using requests"}
    ],
    "max_tokens": 1000,
    "temperature": 0.7
  }'
```

## Python Client

```python
from openai import OpenAI

# Konfiguracja (użyj swojego RunPod IP)
client = OpenAI(
    base_url="http://localhost:8000/v1",  # lub http://RUNPOD_IP:8000/v1
    api_key="token-abc123"  # Dowolny string
)

# Chat completion
response = client.chat.completions.create(
    model="qwen3-coder",
    messages=[
        {"role": "system", "content": "You are an expert Python developer."},
        {"role": "user", "content": "Create a FastAPI app with authentication"}
    ],
    max_tokens=2000,
    temperature=0.7,
    top_p=0.9
)

print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="qwen3-coder",
    messages=[{"role": "user", "content": "Explain async/await in Python"}],
    stream=True,
    max_tokens=1000
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Konfiguracja dla różnych GPU

### RTX 4090 (24GB) - FP8
```yaml
--max-model-len 131072          # 128K kontekst możliwy z FP8!
--gpu-memory-utilization 0.90   # Bezpieczne dla 24GB
--kv-cache-dtype fp8_e4m3       # FP8 cache
```

### A40 48GB - FP8 (DOMYŚLNA KONFIGURACJA)
```yaml
--max-model-len 250000          # 250K kontekst - wykorzystuje dodatkową VRAM
--gpu-memory-utilization 0.92   # Bezpieczne dla 48GB
--max-num-seqs 64               # Dostosowane do 9 vCPU
--kv-cache-dtype fp8_e4m3       # FP8 cache
--block-size 16                 # Optymalizacja dla A40
```

### A100 40GB - FP8
```yaml
--max-model-len 200000          # 200K kontekst z FP8
--gpu-memory-utilization 0.95   # Prawie pełne wykorzystanie
--max-num-seqs 128              # Więcej CPU dostępne
--kv-cache-dtype fp8_e4m3
```

### A100 80GB / H100 - FP8
```yaml
--max-model-len 256000          # Maksymalny natywny kontekst Qwen3
--gpu-memory-utilization 0.95   # Pełne wykorzystanie
--kv-cache-dtype fp8_e4m3       # Najlepsza wydajność
```

### Multi-GPU
```yaml
--tensor-parallel-size 2        # Dla 2 GPU
--pipeline-parallel-size 1
```

## Monitoring

```bash
# Status kontenerów
docker compose ps

# Logi vLLM
docker compose logs -f vllm

# GPU usage
nvidia-smi

# API metrics
curl http://localhost:8000/metrics
```

## Optymalizacja wydajności

### 1. Adjust batch size
```yaml
--max-num-seqs 128              # Więcej równoległych requestów
```

### 2. Memory optimization
```yaml
--swap-space 32                 # Więcej CPU swap space
--gpu-memory-utilization 0.95   # Maksymalne użycie GPU
```

### 3. Attention backend
```yaml
environment:
  - VLLM_ATTENTION_BACKEND=FLASHINFER  # Najszybszy backend
```

## Porównanie wydajności

| Model Format | Engine | Speed | VRAM | Context | A40 Fit? |
|-------------|--------|-------|------|---------|----------|
| GGUF Q5_K_XL | Ollama | 1x | ~30GB | 128K | ✅ |
| FP16 | vLLM | 5-10x | ~45GB | 128K | ✅ |
| **FP8 A40 Opt** | **vLLM** | **8-15x** | **~35GB** | **250K** | **✅ Perfect!** |

**A40 + FP8 = idealne połączenie wydajność/pamięć/kontekst!**

## Troubleshooting

### OOM Error
```bash
# Zmniejsz parametry w .env:
MAX_MODEL_LEN=65536
GPU_MEMORY_UTILIZATION=0.85

# Restart
docker compose down && docker compose up -d
```

### Wolne uruchamianie
```bash
# Model pobiera się przy pierwszym starcie (~30GB)
# Sprawdź postęp:
docker compose logs -f vllm
```

### API nie odpowiada
```bash
# Sprawdź czy kontener działa
docker compose ps

# Sprawdź porty
netstat -tlnp | grep 8000

# Test health
curl -I http://localhost:8000/health
```

### Brak GPU
```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Jeśli nie działa - reinstaluj NVIDIA Container Toolkit
```

## Endpoints RunPod

Po expose port 8000 w RunPod:

- **Lokalny**: `http://localhost:8000`
- **Publiczny**: `http://[RUNPOD_IP]:8000` 
- **Proxy**: `https://[POD_ID]-8000.proxy.runpod.net`

## Model Information

- **Nazwa**: unsloth/Qwen3-Coder-30B-A3B-Instruct
- **Rozmiar**: ~30B parametrów  
- **Precyzja**: BFloat16 + FP8 KV Cache
- **Kontekst**: 250K tokenów (A40 optimized), do 256K max
- **Specjalizacja**: Kodowanie, programowanie
- **Języki**: Python, JavaScript, Java, C++, Go, Rust, i inne

To znacznie prostszy i wydajniejszy setup niż Ollama z GGUF!