<div align="center">
  <img src="src/vlex_ria/serving/web/static/logo.png" alt="VLex-RIA" width="200"/>
</div>

# VLex-RIA: Vietnamese Legal LLM Architecture
VLex-RIA (Vietnamese Legal Reasoning Intelligent Assistant) là một kiến trúc mô hình ngôn ngữ lớn (LLM) đột phá được tối ưu hóa đặc biệt cho lĩnh vực pháp lý Việt Nam. Dự án này triển khai một pipeline đào tạo end-to-end (từ đầu đến cuối), kết hợp các công nghệ tiên tiến nhất từ kiến trúc DeepSeek V3 với các cải tiến NLP đặc thù cho tiếng Việt.

## 🌟 Tính Năng Cốt Lõi

Kiến trúc VLex-RIA được xây dựng trên ba trụ cột chính của DeepSeek V3:

1. **Multi-Head Latent Attention (MLA)**:
   - Tối ưu hóa KV Cache bằng cách nén (compress) Key và Value thành một không gian vector (latent vector) chung.
   - Cải thiện tốc độ nội suy và giảm đáng kể mức sử dụng bộ nhớ trong quá trình suy luận văn bản dài.

2. **Mixture of Experts (MoE) với Shared Experts**:
   - Kiến trúc MoE tối giản sử dụng một số chuyên gia chia sẻ (shared experts) nắm bắt tri thức chung và các chuyên gia định tuyến (routed experts) để xử lý tri thức chuyên biệt.
   - Cơ chế load-balancing loss giúp cân bằng hệ số phân phối token trên các chuyên gia, ngăn chặn tình trạng nghẽn nghẽn nhánh phụ.

3. **Multi-Token Prediction (MTP)**:
   - Thay vì chỉ dự đoán 1 token duy nhất trong mỗi bước, mô hình học cách dự đoán nhiều token tiếp theo cùng lúc thông qua các lớp dự đoán (prediction heads) tuần tự.
   - Làm tiền đề quan trọng cho kỹ thuật **Speculative Decoding** tại thời điểm inference, mang lại khả năng tăng tốc vượt trội.

---

## 🛠️ Cải tiến đặc thù cho Tiếng Việt (VN-Specific Features)

Để xử lý cấu trúc ngôn ngữ và văn bản pháp lý Việt Nam phức tạp, VLex-RIA đã trang bị:

### 1. Tokenizer Tối Ưu Hóa (PhoBERT)
- Tích hợp chuẩn tokenizer `vinai/phobert-base` dành riêng cho tiếng Việt.
- Tự động nhận diện và tiền xử lý văn bản dựa trên `underthesea` word segmentation (nối từ ghép bằng `_` như `trách_nhiệm hình_sự`), giữ nguyên ngữ nghĩa.
- **Tập Token Cạnh Tranh (Token Competitors):** Tokenizer được điều chỉnh linh hoạt độ dài seq length, xử lý tốt các token kết dính và văn bản nhiễu (noise) trong tài liệu hành chính.

### 2. Legal Entity Recognition (LER)
- Giai đoạn Data Adapter (cho SFT và RLHF) tự động sử dụng regex và logic trích xuất để dò tìm các thực thể pháp lý (điều khoản luât, nghị định, thông tư...).
- Tự động tiêm (inject) các "ngữ cảnh pháp lý" (legal contexts) vào câu hỏi để dẫn dắt mô hình trả lời chính xác, tránh hallucination cơ bản.

### 3. Numerical Stability trên Apple Silicon (MPS)
- Được tinh chỉnh thủ công để huấn luyện ổn định (stable) trên kiến trúc phân tán macOS/Apple Silicon.
- Khắc phục lỗi `NaN Loss` kinh điển trong attention map bằng ranh giới `-1e9` thay vì `-inf` thô bạo.

---

## 📦 Kiến Trúc Tệp (Project Structure)

```text
VLex-RIA/
├── src/vlex_ria/           # Mã nguồn chính
│   ├── model/              # Kiến trúc Model cốt lõi (MLA, MoE, MTP, Transformer)
│   ├── core/               # File cấu hình (Config classes), Logging
│   ├── data/               # Xử lý Dataset, Tokenizer, Vietnamese NLP Cleaning
│   ├── training/           # Pipeline huấn luyện (Pretrain, SFT, DPO, GRPO, PPO)
│   └── inference/          # Lớp Inference, Speculative Decoding hỗ trợ MTP
│   └── serving/web/        # 🌐 Giao diện Chat Web (Streaming, Chọn Model)
├── configs/                # File cấu hình YAML (Default & Legal_VN)
├── scripts/                # Launch scripts (Pretrain, RLHF, Web Server)
├── tests/                  # Bộ Test toàn diện (Unit Test & E2E)
├── pyproject.toml          # Quản lý Package và Dependency hiện đại
└── requirements.txt        # Liệt kê môi trường cài đặt cho người dùng
```

---

## 🚀 Hướng Dẫn Cài Đặt

Mô hình hiện tương thích với **Python >= 3.10** và PyTorch >= 2.0.

### 1. Clone & Cài Đặt 
```bash
# Cài đặt dưới dạng package có thể chỉnh sửa (editable mode)
pip install -e .

# (Tùy chọn) Cài thêm gói NLP tiếng Việt cho phần tách từ
pip install -e ".[vn]"

# Hoặc cài toàn bộ dependencies có sẵn
pip install -e ".[all]"
```

### 2. File Môi Trường
Đảm bảo bạn có đủ quyền đối với dataset. Nếu dùng HuggingFace dataset, tạo hoặc login qua `huggingface-cli login`:
```bash
export HF_TOKEN="your_hf_token_here" # Tùy chọn nếu cần load repo kín
```

---

## 💻 Hướng Dẫn Sử Dụng

### Giao Diện Web Chat (Inference Streaming)
Dự án đi kèm với một giao diện Web Chat hoàn chỉnh sử dụng Server-Sent Events (SSE) để truyền luồng token mượt mà:
```bash
# Khởi chạy server tại localhost:5001
./scripts/run.sh web-chat
```
- Khả năng tự động tìm kiếm các model `.pt` trong cấu trúc `artifacts/checkpoints`.
- Cấu hình linh hoạt Hyperparameters: Temperature, Top-K, Top-P, Repetition Penalty. 

### Chạy Pipeline Huấn Luyện (Training)
Kịch bản chung bao quát cả 3 giai đoạn: Pretraining, Supervised-Finetuning (SFT) & Reinforcement Learning (RL):
```bash
# Chạy bộ test hệ thống toàn diện trước khi train
python3 tests/test_all.py 

# Chạy pipeline tổng hợp mẫu cho Pháp lý Việt Nam (Cần tải Dataset)
./scripts/run_pipeline_legal_vn.sh
```

---

## 🔬 Giai Đoạn Reinforcement Learning (RL)

Khác biệt nổi bật của VLex-RIA là hỗ trợ lên tới 3 thuật toán RL tối ưu hóa trải nghiệm người dùng cuối ở giai đoạn Alignment:

1. **DPO (Direct Preference Optimization)**: Tối ưu hóa trọng số mô hình trực tiếp qua các cặp dữ liệu tương phản (Cặp Chosen vs Rejected), hoàn toàn không cần Reward Model rời.
2. **PPO (Proximal Policy Optimization)**: Tiếp cận truyền thống của RLHF, cân bằng Actor/Critic model. Cho kiểm soát hành vi tinh tế nhất nhưng đòi hỏi tài nguyên cao nhất. 
3. **GRPO (Group Relative Policy Optimization)**: Phiên bản tối giản PPO (Không Critic model), chỉ cần tính luân phiên đánh giá điểm (Reward) tương đối trên 1 sample group. 

> *Lưu ý*: Hàm thưởng (Reward Functions) của chúng tôi tính hợp **Legal Rule-Based Reward** (Thưởng điểm khi trích xuất đúng điều khoản pháp luật, chống chém gió) bên cạnh thưởng độ dài.
