# VLex-RIA Web Chat Interface

## Tính năng đã hoàn thiện

### Chọn & chuyển đổi model động
- Tự động quét toàn bộ file `.pt` trong thư mục `artifacts/checkpoints/`
- Phân nhóm checkpoint theo loại: **Pretrained**, **SFT**, **RL**
- Hiển thị animation loading khi đang tải model
- Cơ chế cache model — model đã tải sẽ chuyển đổi nhanh hơn (không load lại từ disk)
- Tự động huỷ generation đang chạy khi chuyển model

### Sinh văn bản theo luồng (Streaming)
- Sử dụng **Server-Sent Events** — mỗi token hiển thị ngay khi sinh ra
- Hỗ trợ đầy đủ các tham số sinh: `temperature`, `top_p`, `top_k`, `max_new_tokens`, `repetition_penalty`
- Áp dụng repetition penalty để giảm lặp từ
- Lọc Top-K và Top-P (nucleus sampling)

### Huỷ generation
- Nút Huỷ hiện ra ngay khi bắt đầu sinh văn bản
- Dừng ngay lập tức khi người dùng nhấn huỷ (qua `threading.Event`)
- Chuyển model cũng tự động kích hoạt huỷ

### Giao diện dark theme
- Thiết kế dark mode hiện đại
- Layout responsive
- Khung chat dạng bubble (phân biệt người dùng / AI)
- Sidebar điều chỉnh tham số sinh
- Thanh chọn model tích hợp ngay trên giao diện
- Chức năng xoá lịch sử hội thoại
- Hiển thị lỗi thân thiện


## Cấu trúc file

```
src/vlex_ria/serving/web/
├── app.py                  # Flask backend — API và logic sinh văn bản
├── templates/
│   └── index.html          # Template HTML chính
├── static/
│   ├── style.css           # Stylesheet dark theme
│   ├── app.js              # Frontend JavaScript (SSE, UI logic)
│   ├── logo.png            # Logo VLex-RIA
│   └── user.png            # Avatar người dùng
└── IMPLEMENTATION.md       # Tài liệu này
```

## Khởi động

### Cách 1: Script tổng hợp
```bash
./scripts/run.sh web-chat
```

### Cách 2: Chạy trực tiếp
```bash
# Từ thư mục gốc LegalLM/
python src/vlex_ria/serving/web/app.py
```

### Sau khi khởi động, truy cập: **http://localhost:5001**

## API Endpoints

### `GET /api/checkpoints`
Lấy danh sách tất cả checkpoint có sẵn trong `artifacts/checkpoints/`.

**Response:**
```json
{
    "checkpoints": [
        {
            "name": "pretrain/final.pt",
            "path": "/absolute/path/to/final.pt",
            "size_mb": 245.3,
            "category": "Pretrained"
        }
    ],
    "default": "/path/to/default.pt",
    "current": "/path/to/loaded.pt"
}
```

### `POST /api/load_model`
Tải một checkpoint vào bộ nhớ.

**Request:**
```json
{
    "checkpoint_path": "/absolute/path/to/model.pt"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Model loaded: /path/to/model.pt",
    "current": "/path/to/model.pt"
}
```

### `GET /api/model_status`
Kiểm tra trạng thái model hiện tại.

**Response:**
```json
{
    "current_model": "/path/to/model.pt",
    "is_generating": false,
    "cached_models": ["/path/to/model1.pt", "/path/to/model2.pt"]
}
```

### `POST /api/generate`
Sinh văn bản theo luồng (SSE). Response là `text/event-stream`.

**Request:**
```json
{
    "prompt": "Hỏi về luật doanh nghiệp",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "checkpoint_path": "/path/to/model.pt"
}
```

**SSE Events:**
```
data: {"type": "token", "content": "Theo"}
data: {"type": "token", "content": " quy"}
data: {"type": "done"}
data: {"type": "cancelled"}
data: {"type": "error", "message": "..."}
```

### `POST /api/cancel`
Huỷ generation đang chạy.

**Response:**
```json
{
    "success": true,
    "message": "Cancellation requested"
}
```

## Cấu hình model

Khi tải checkpoint, server tự động tìm config theo thứ tự ưu tiên:

Phân tích cấu hình mặc định:
1. `configs/config.yaml` — Toàn quyền kiểm soát cấu hình cho Web Chat

Config phải khớp với kiến trúc model đã dùng khi huấn luyện (hidden size, số lớp, v.v.).

## Lưu ý kỹ thuật

| Vấn đề | Chi tiết |
|---|---|
| **CPU mode** | Web server luôn chạy trên CPU (`device="cpu"`) — tránh xung đột thread với MPS trên macOS |
| **Model cache** | Model đã tải được giữ trong RAM (`_model_cache`); restart Python để giải phóng bộ nhớ |
| **Tokenizer** | Sử dụng `vinai/phobert-base` (PhoBERT) theo cấu hình `DataConfig.tokenizer_name` |
| **Thread safety** | Dùng `threading.Lock` cho load model, `threading.Event` cho cancel generation |
| **Browser** | Yêu cầu trình duyệt hiện đại hỗ trợ SSE (Chrome, Firefox, Safari, Edge) |
