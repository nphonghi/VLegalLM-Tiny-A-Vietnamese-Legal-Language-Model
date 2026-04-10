<div align="center">
  <img src="src/vlex_ria/serving/web/static/logo.png" alt="VLex-RIA" width="200"/>
  <h1>VLex-RIA: Vietnamese Legal Regulatory Impact Assessment</h1>
  <p>
    <b>Efficient LLM Optimized for Vietnamese Legal Systems</b>
  </p>


</div>

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Key Milestones](#key-milestones)
3. [Model Architecture](#2-model-architecture)
   - [Multi-Head Latent Attention](#multi-head-latent-attention-mla)
   - [DeepSeekMoE](#deepseekmoe)
   - [Multi-Token Prediction](#multi-token-prediction-mtp)
4. [Vietnamese Legal Adaptation](#3-vietnamese-legal-adaptation)
5. [Software Stack](#4-software-stack)
   - [Training Pipeline](#training-pipeline)
   - [Inference & Serving](#inference--serving)
6. [Installation](#5-installation)
7. [Quick Start & Usage](#6-quick-start--usage)
   - [Inference with Web Chat](#inference-with-web-chat)
   - [Training Pipeline](#training-pipeline-1)
8. [Project Structure](#7-project-structure)

---

## 1. Introduction
**VLex-RIA** là một LLM được phát triển cho việc học tập và nghiên cứu, hiểu rõ quy trình xây dựng mô hình ngôn ngữ, cùng với đó là mong muốn xây dựng một LLM có thể giải quyết bài toán RIA với các văn bản quy phạm pháp luật Việt Nam. Dựa trên kiến trúc **DeepSeek-V3**, dự án triển khai một pipeline đào tạo end-to-end trên dữ liệu pháp luật Việt Nam.

---

## 2. Model Architecture

### Multi-Head Latent Attention

### DeepSeekMoE

### Multi-Token Prediction 

---

## 3. Vietnamese Legal Adaptation
Tập trung vào việc tinh chỉnh pipeline để phù hợp với đặc thù ngôn ngữ và văn bản Việt Nam:

| Component | Optimization Detail |
| :--- | :--- |
| **Tokenizer** | Tối ưu dựa trên `vinai/phobert-base` với word segmentation `underthesea`. |
| **LER Adapter** | Tích hợp **Legal Entity Recognition** để trích xuất các căn cứ pháp luật trực tiếp từ input, giảm thiểu tình trạng hallucination. |

---

## 4. Software Stack

### Training Pipeline
1. **Pre-training**: Tiếp nhận tri thức pháp luật trên diện rộng.
2. **SFT**: Huấn luyện theo các cặp dữ liệu Hội thoại - Giải đáp pháp lý của chuyên gia.
3. **RL**: Hỗ trợ thuật toán **GRPO** giúp căn chỉnh mô hình hiệu quả hơn mà không cần mô hình Critic phức tạp.

### Inference & Serving
- **Streaming Engine**: Hỗ trợ truyền luồng token thời gian thực (SSE).
- **Web Interface**: Giao diện UI/UX trực quan dành cho người dùng cuối.

---

## 5. Installation

Yêu cầu **Python >= 3.10** và **PyTorch >= 2.0**.

```bash
# Clone the repository
git clone https://github.com/nphonghi/VLegalLM-Tiny-A-Vietnamese-Legal-Language-Model.git
cd VLegalLM-Tiny-A-Vietnamese-Legal-Language-Model

# Install in editable mode with all dependencies
pip install -e ".[all]"
```

---

## 6. Quick Start & Usage

### Inference with Web Chat
Dự án tích hợp sẵn một Web Server hoàn chỉnh để tương tác trực tiếp:
```bash
./scripts/run.sh web-chat
```
Truy cập tại: `http://localhost:5001`

### Training Pipeline
Để bắt đầu đào tạo mô hình cho dữ liệu pháp lý Việt Nam:
```bash
# Chạy toàn bộ pipeline (Pretrain -> SFT -> RL)
./scripts/run_pipeline_legal_vn.sh
```

---

## 7. Project Structure
```text
VLex-RIA/
├── src/vlex_ria/           # Core Source Code
│   ├── model/              # MLA, MoE, MTP implementations
│   ├── training/           # Training engines (SFT, DPO, GRPO, PPO)
│   ├── data/               # Vietnamese Legal NLP Pipeline
│   └── serving/web/        # Streaming Web Application
├── configs/                # System & Model Configurations
├── scripts/                # Launch & Automation scripts
├── tests/                  # Exhaustive Test Suite
└── tools/                  # Tokenizer & Data utilities
```

---
