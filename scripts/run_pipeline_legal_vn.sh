#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src:${PYTHONPATH:-}"

CONFIG_FILE="configs/config.yaml"
RUNNER="$PROJECT_DIR/scripts/run.sh"

echo "=========================================================================="
echo "                  BẮT ĐẦU PIPELINE HUẤN LUYỆN LEGAL VN                    "
echo "=========================================================================="
echo "  Config:   $CONFIG_FILE"
echo "  Datasets:"
echo "    Pretrain: undertheseanlp/UTS_VLC (Bộ luật dân sự, hình sự...)"
echo "    SFT:      YuITC/Vietnamese-Legal-Documents (Q&A pháp lý)"
echo "    RL:       YuITC/Vietnamese-Legal-Documents (GRPO)"
echo "  Tokenizer: vinai/phobert-base"
echo "=========================================================================="

echo ""
echo ">>> Giai đoạn 1/3: PRETRAIN — Học ngôn ngữ pháp lý Việt Nam"
echo "    Dataset: undertheseanlp/UTS_VLC (Hiến pháp, Bộ luật dân sự, hình sự...)"
echo "---"
bash "$RUNNER" --config "$CONFIG_FILE" --device cuda pretrain

echo ""
echo ">>> Giai đoạn 2/3: SFT — Supervised Fine-Tuning trên Q&A pháp lý"
echo "    Dataset: YuITC/Vietnamese-Legal-Documents"
echo "---"
if [ -f "artifacts/checkpoints/pretrain/legal_vn/best.pt" ]; then
    PRETRAIN_CKPT="artifacts/checkpoints/pretrain/legal_vn/best.pt"
elif [ -f "artifacts/checkpoints/pretrain/legal_vn/final.pt" ]; then
    PRETRAIN_CKPT="artifacts/checkpoints/pretrain/legal_vn/final.pt"
else
    echo "WARN: Không tìm thấy pretrain checkpoint, SFT sẽ train from scratch"
    PRETRAIN_CKPT=""
fi
bash "$RUNNER" --config "$CONFIG_FILE" --device cuda --checkpoint "$PRETRAIN_CKPT" sft

echo ""
echo ">>> Giai đoạn 3/3: RL (GRPO) — Reinforcement Learning alignment"  
echo "    Dataset: YuITC/Vietnamese-Legal-Documents (GRPO)"
echo "---"
if [ -f "artifacts/checkpoints/sft/legal_vn/best.pt" ]; then
    SFT_CKPT="artifacts/checkpoints/sft/legal_vn/best.pt"
elif [ -f "artifacts/checkpoints/sft/legal_vn/final.pt" ]; then
    SFT_CKPT="artifacts/checkpoints/sft/legal_vn/final.pt"
else
    echo "WARN: Không tìm thấy SFT checkpoint, RL sẽ train from scratch"
    SFT_CKPT=""
fi
bash "$RUNNER" --config "$CONFIG_FILE" --device cuda --checkpoint "$SFT_CKPT" rl

echo ""
echo "=========================================================================="
echo "                      HOÀN TẤT PIPELINE LEGAL VN                          "
echo "  Checkpoints:"
echo "    Pretrain: artifacts/checkpoints/pretrain/legal_vn/"
echo "    SFT:      artifacts/checkpoints/sft/legal_vn/"
echo "    RL:       artifacts/checkpoints/rl/legal_vn/"
echo "  TensorBoard: artifacts/runs/"
echo "=========================================================================="
