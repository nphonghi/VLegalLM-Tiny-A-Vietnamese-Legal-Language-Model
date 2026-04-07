#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src:${PYTHONPATH:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CONFIG="configs/config.yaml"
DEVICE="auto"
CHECKPOINT=""

print_header() {
    echo -e "${BLUE}"
    echo "======================================================================"
    echo "                    VLex-RIA DeepSeek Pipeline                        "
    echo "       Modes: Pretrain -> SFT -> RL + Inference + Web Chat            "
    echo "======================================================================"
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 <command> [options] [command-args]"
    echo ""
    echo "Commands:"
    echo "  test             Run all tests"
    echo "  test-quick       Run quick model test"
    echo "  pretrain         Start pretraining"
    echo "  pretrain-test    Quick pretrain test mode"
    echo "  sft              Start supervised fine-tuning"
    echo "  sft-test         Quick SFT test mode"
    echo "  rl               Start reinforcement learning (default grpo)"
    echo "  rl-test          Quick RL test mode"
    echo "  inference        Run inference demo"
    echo "  chat             Run interactive CLI chat"
    echo "  web-chat         Start web chat interface"
    echo "  tensorboard      Start TensorBoard from artifacts/runs"
    echo "  full             Run pretrain -> sft -> rl -> inference"
    echo "  full-test        Run quick full pipeline"
    echo "  clean            Clean artifacts and python cache"
    echo "  help             Show this help"
    echo ""
    echo "Options:"
    echo "  --config <path>       Config file (default: configs/config.yaml)"
    echo "  --device <device>     Device: auto, cuda, mps, cpu (default: auto)"
    echo "  --checkpoint <path>   Checkpoint path for sft/rl/inference/chat"
}

ensure_dirs() {
    mkdir -p artifacts/checkpoints artifacts/runs artifacts/logs artifacts/eval
}

check_environment() {
    echo -e "${YELLOW}Checking environment...${NC}"
    if ! command -v python3 >/dev/null 2>&1; then
        echo -e "${RED}Error: Python 3 is not installed${NC}"
        exit 1
    fi
    python3 -c "import torch" >/dev/null 2>&1 || {
        echo -e "${RED}Error: PyTorch not installed${NC}"
        exit 1
    }
    echo -e "${GREEN}Environment OK${NC}"
}

run_tests() {
    python3 tests/test_all.py --test all
}

run_quick_test() {
    python3 tests/test_all.py --test model
}

run_pretrain() {
    python3 src/vlex_ria/training/engine.py --mode pretrain --config "$CONFIG" --device "$DEVICE" "$@"
}


run_pretrain_test() {
    python3 src/vlex_ria/training/engine.py --mode pretrain --config "$CONFIG" --device "$DEVICE" --test "$@"
}

run_sft() {
    local checkpoint_path="${1:-artifacts/checkpoints/pretrain/legal_vn/best.pt}"
    python3 src/vlex_ria/training/engine.py --mode sft --config "$CONFIG" --device "$DEVICE" --checkpoint "$checkpoint_path"
}

run_sft_test() {
    python3 src/vlex_ria/training/engine.py --mode sft --config "$CONFIG" --device "$DEVICE" --test
}

run_rl() {
    local checkpoint_path="${1:-artifacts/checkpoints/sft/legal_vn/best.pt}"
    python3 src/vlex_ria/training/engine.py --mode rl --config "$CONFIG" --device "$DEVICE" --checkpoint "$checkpoint_path"
}

run_rl_test() {
    python3 src/vlex_ria/training/engine.py --mode rl --config "$CONFIG" --device "$DEVICE" --test
}

run_inference() {
    local checkpoint_path="${1:-artifacts/checkpoints/sft/legal_vn/best.pt}"
    python3 src/vlex_ria/inference/inference.py --config "$CONFIG" --checkpoint "$checkpoint_path" --device "$DEVICE"
}

run_chat() {
    local checkpoint_path="${1:-artifacts/checkpoints/sft/legal_vn/best.pt}"
    python3 src/vlex_ria/inference/inference.py --config "$CONFIG" --checkpoint "$checkpoint_path" --device "$DEVICE" --interactive
}

run_web_chat() {
    python3 src/vlex_ria/serving/web/app.py
}

run_tensorboard() {
    tensorboard --logdir artifacts/runs --port 6006
}

run_full_pipeline() {
    run_pretrain
    run_sft "artifacts/checkpoints/pretrain/best.pt"
    run_rl "artifacts/checkpoints/sft/best.pt"
    run_inference "artifacts/checkpoints/rl/final.pt"
}

run_full_pipeline_test() {
    run_pretrain_test
    run_sft_test
    run_rl_test
    run_inference "artifacts/checkpoints/rl/final.pt"
}

clean() {
    rm -rf artifacts/checkpoints artifacts/runs artifacts/logs artifacts/eval data/__pycache__ __pycache__
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

print_header

case "$COMMAND" in
    test)
        check_environment
        run_tests
        ;;
    test-quick)
        check_environment
        run_quick_test
        ;;
    pretrain)
        check_environment
        ensure_dirs
        run_pretrain "$@"
        ;;

    pretrain-test)
        check_environment
        ensure_dirs
        run_pretrain_test "$@"
        ;;
    sft)
        check_environment
        ensure_dirs
        run_sft "${CHECKPOINT:-${1:-}}"
        ;;
    sft-test)
        check_environment
        ensure_dirs
        run_sft_test
        ;;
    rl)
        check_environment
        ensure_dirs
        run_rl "${CHECKPOINT:-${1:-}}"
        ;;
    rl-test)
        check_environment
        ensure_dirs
        run_rl_test
        ;;
    inference)
        check_environment
        run_inference "${CHECKPOINT:-${1:-}}"
        ;;
    chat)
        check_environment
        run_chat "${CHECKPOINT:-${1:-}}"
        ;;
    web-chat)
        check_environment
        ensure_dirs
        run_web_chat
        ;;
    tensorboard)
        run_tensorboard
        ;;
    full)
        check_environment
        ensure_dirs
        run_full_pipeline
        ;;
    full-test)
        check_environment
        ensure_dirs
        run_full_pipeline_test
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        print_usage
        exit 1
        ;;
esac
