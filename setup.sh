#!/bin/bash
set -e

echo "🚀 vLLM Qwen3-Coder Setup dla RunPod"
echo "=================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_requirements() {
    log_info "Sprawdzanie wymagań..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        log_error "Uruchom jako root: sudo ./setup.sh"
        exit 1
    fi
    
    # Check GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA GPU nie wykryte"
        exit 1
    fi
    
    # Check VRAM
    local vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    log_info "Wykryto GPU z ${vram}MB VRAM"
    
    if [[ $vram -gt 45000 ]]; then
        log_info "Świetnie! ${vram}MB VRAM - pełna konfiguracja A40"
    elif [[ $vram -gt 35000 ]]; then
        log_info "Dobry VRAM (${vram}MB) - zmniejszę kontekst do 200K"
        sed -i 's/MAX_MODEL_LEN=250000/MAX_MODEL_LEN=200000/' .env
    elif [[ $vram -gt 30000 ]]; then
        log_warn "Średni VRAM (${vram}MB) - kontekst 128K"
        sed -i 's/MAX_MODEL_LEN=250000/MAX_MODEL_LEN=131072/' .env
    elif [[ $vram -lt 24000 ]]; then
        log_error "Za mało VRAM (${vram}MB). Minimum 24GB dla FP8"
        exit 1
    fi
    
    # Check disk space
    local space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ $space -lt 60 ]]; then
        log_error "Za mało miejsca na dysku: ${space}GB (potrzeba 60GB+)"
        exit 1
    fi
    
    log_info "✅ Wymagania spełnione"
}

# Install Docker if needed
install_docker() {
    if command -v docker &> /dev/null; then
        log_info "Docker już zainstalowany"
        return 0
    fi
    
    log_info "Instalowanie Docker..."
    
    # Install Docker
    curl -fsSL https://get.docker.com | sh
    
    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    apt update
    apt install -y nvidia-container-toolkit docker-compose-plugin
    
    # Configure and restart Docker
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    
    # Test
    if docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi; then
        log_info "✅ Docker z GPU zainstalowany"
    else
        log_error "❌ Problem z Docker GPU"
        exit 1
    fi
}

# Start vLLM
start_vllm() {
    log_info "Uruchamianie vLLM..."
    
    # Pull latest image
    docker compose pull
    
    # Start service
    docker compose up -d
    
    log_info "Czekam na uruchomienie vLLM..."
    
    # Wait for health check
    local timeout=600  # 10 minutes
    local count=0
    
    while [[ $count -lt $timeout ]]; do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            log_info "✅ vLLM uruchomiony!"
            break
        fi
        
        sleep 5
        count=$((count + 5))
        
        if [[ $((count % 60)) -eq 0 ]]; then
            log_info "Nadal czekam... (${count}/${timeout}s)"
            log_info "Sprawdź logi: docker compose logs -f"
        fi
    done
    
    if [[ $count -ge $timeout ]]; then
        log_error "vLLM nie uruchomił się w czasie"
        log_error "Sprawdź logi: docker compose logs vllm"
        exit 1
    fi
}

# Test API
test_api() {
    log_info "Testowanie API..."
    
    # Health check
    if ! curl -sf http://localhost:8000/health > /dev/null; then
        log_error "Health check failed"
        return 1
    fi
    
    # Model list
    local models=$(curl -s http://localhost:8000/v1/models | grep -o '"qwen3-coder"' || echo "")
    if [[ -z "$models" ]]; then
        log_error "Model nie jest dostępny"
        return 1
    fi
    
    # Test completion
    local response=$(curl -s http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "qwen3-coder",
            "prompt": "def hello():",
            "max_tokens": 50,
            "temperature": 0.1
        }' | grep -o '"text"' || echo "")
    
    if [[ -n "$response" ]]; then
        log_info "✅ API działa poprawnie"
        return 0
    else
        log_error "Test completion failed"
        return 1
    fi
}

# Show completion info
show_completion() {
    local public_ip=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_RUNPOD_IP")
    
    echo ""
    echo "🎉 Setup zakończony!"
    echo "==================="
    echo ""
    echo "📡 Endpointy:"
    echo "• Lokalny:    http://localhost:8000"
    echo "• Zewnętrzny: http://${public_ip}:8000"
    echo "• RunPod:     https://[POD_ID]-8000.proxy.runpod.net"
    echo ""
    echo "🧪 Test API:"
    echo "curl http://localhost:8000/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{"
    echo '    "model": "qwen3-coder",'
    echo '    "messages": [{"role": "user", "content": "Write hello world in Python"}],'
    echo '    "max_tokens": 100'
    echo "  }'"
    echo ""
    echo "📊 Monitoring:"
    echo "• Status:  docker compose ps"
    echo "• Logi:    docker compose logs -f"
    echo "• GPU:     nvidia-smi"
    echo ""
    echo "🔧 Zarządzanie:"
    echo "• Restart: docker compose restart"
    echo "• Stop:    docker compose down"
    echo ""
    echo "⚠️  Pamiętaj: Expose port 8000 w RunPod!"
}

# Main execution
main() {
    cd "$(dirname "$0")"
    
    echo "Czy kontynuować setup vLLM z Qwen3-Coder-30B? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Setup anulowany"
        exit 0
    fi
    
    check_requirements
    install_docker
    start_vllm
    test_api
    show_completion
    
    log_info "🚀 vLLM Qwen3-Coder gotowy do użycia!"
}

main "$@"