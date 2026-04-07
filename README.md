<div align="center">
  <img src="src/vlex_ria/serving/web/static/logo.png" alt="VLex-RIA" width="200"/>
  <h1>VLex-RIA: Vietnamese Legal Regulatory Impact Assessment</h1>
  <p>
    <b>Efficient LLM Architecture Optimized for Vietnamese Legal Systems</b>
  </p>

[**Paper (Coming Soon)**] | [**Project Page**] | [**Hugging Face**]

</div>

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Key Milestones](#key-milestones)
3. [Model Architecture](#2-model-architecture)
   - [Multi-Head Latent Attention (MLA)](#multi-head-latent-attention-mla)
   - [DeepSeekMoE](#deepseekmoe)
   - [Multi-Token Prediction (MTP)](#multi-token-prediction-mtp)
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
**VLex-RIA** (Vietnamese Legal Reasoning Intelligent Assistant) là một hệ thống mô hình ngôn ngữ lớn (LLM) đột phá được thiết kế chuyên biệt cho hệ thức pháp luật Việt Nam. Dựa trên những cải tiến từ kiến trúc **DeepSeek-V3**, dự án triển khai một pipeline đào tạo end-to-end, tối ưu hóa cả về tri thức pháp lý lẫn hiệu suất tính toán trên các thiết bị hiện đại.

### Key Milestones
- **Efficiency**: Hybrid MoE with MLA KV compression.
- **Domain Specialization**: Deep integration of Vietnamese Legal Codes, Case Laws, and Administrative circulars.
- **Advanced Alignment**: Full support for SFT, DPO, and **GRPO** (Group Relative Policy Optimization).
- **Environment**: Native optimization for Apple Silicon (MPS) and high-performance CUDA clusters.

---

## 2. Model Architecture
VLex-RIA kế thừa những tinh hoa kỹ thuật từ DeepSeek-V3 để đạt được sự cân bằng hoàn hảo giữa tham số mô hình và hiệu quả thực thi:

### Multi-Head Latent Attention (MLA)
Sử dụng cơ chế nén KV mã hóa thấp (low-rank compression) để giảm tối đa dung lượng KV cache. Điều này cho phép mô hình duy trì tốc độ nội suy cao ngay cả khi xử lý các hồ sơ pháp lý có độ dài văn bản lớn.

### DeepSeekMoE
Kiến trúc Mixture-of-Experts thế hệ mới:
- **Shared Experts**: Nắm bắt các tri thức ngôn ngữ chung, đảm bảo tính ổn định của mô hình.
- **Routed Experts**: Các chuyên gia chuyên biệt hóa cho từng khía cạnh kiến thức pháp lý (Dân sự, Hình sự, Hành chính...).
- **Load Balancing**: Sử dụng cơ chế cân bằng tải tiên tiến để tối ưu hóa hiệu suất song song.

### Multi-Token Prediction (MTP)
Huấn luyện mô hình dự đoán đồng thời nhiều token kế tiếp. Kỹ thuật này không chỉ giúp mô hình học được cấu trúc câu pháp lý chặt chẽ hơn mà còn là tiền đề cho việc tăng tốc suy luận thông qua **Speculative Decoding**.

---

## 3. Vietnamese Legal Adaptation
Chúng tôi tập trung vào việc tinh chỉnh pipeline để phù hợp với đặc thù ngôn ngữ và văn bản Việt Nam:

| Component | Optimization Detail |
| :--- | :--- |
| **Tokenizer** | Tối ưu dựa trên `vinai/phobert-base` với word segmentation `underthesea`, xử lý tốt các thuật ngữ ghép phức tạp. |
| **LER Adapter** | Tích hợp **Legal Entity Recognition** để trích xuất các căn cứ pháp luật trực tiếp từ input, giảm thiểu tình trạng "ảo giác" (hallucination). |
| **Numerical Stability** | Khắc phục triệt để lỗi `NaN loss` trên Apple Silicon bằng cách thay thế các giá trị `-inf` trong attention maps. |

---

## 4. Software Stack

### Training Pipeline
Hỗ trợ đầy đủ quy trình căn chỉnh mô hình chuyên sâu:
1. **Pre-training**: Tiếp nhận tri thức pháp luật trên diện rộng.
2. **SFT (Supervised Fine-Tuning)**: Huấn luyện theo các cặp dữ liệu Hội thoại - Giải đáp pháp lý của chuyên gia.
3. **Alignment**: Hỗ trợ thuật toán **GRPO** (DeepSeek's signature) giúp căn chỉnh mô hình hiệu quả hơn mà không cần mô hình Critic phức tạp.

### Inference & Serving
- **Streaming Engine**: Hỗ trợ truyền luồng token thời gian thực (SSE).
- **Web Interface**: Giao diện UI/UX trực quan dành cho người dùng cuối.

---

## 5. Installation

Yêu cầu **Python >= 3.10** và **PyTorch >= 2.0**.

```bash
# Clone the repository
git clone https://github.com/nphonghi/LegalLM.git
cd LegalLM

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
