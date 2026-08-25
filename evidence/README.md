# Bằng chứng - Day 22: LangSmith + Prompt Versioning

Học viên: Trần Minh Hạnh - 2A202601232
LangSmith project: `day22-lab`
https://smith.langchain.com/o/311850db-4bef-4f14-91e8-ff2eac2b3c16/projects/p/05856170-3972-4eca-942e-86c675da00e6?timeModel=%7B%22duration%22%3A%221d%22%7D

## Danh mục tệp

| Tệp | Nội dung |
|---|---|
| `01_langsmith_traces.png` | Giao diện LangSmith, project `day22-lab` |
| `02_prompt_hub.png` | Prompt Hub với `tranminhhanh-rag-v1` và `-v2` |
| `02_ab_routing_log.txt` | Log Bước 2: push/pull Hub, 50 câu có nhãn v1/v2 |
| `03_ragas_scores.png` | Bảng so sánh RAGAS V1 vs V2 |
| `03_ragas_report.json` | Bản sao của `data/ragas_report.json` |
| `04_pii_demo_log.txt` | 6 test case PII |
| `04_json_demo_log.txt` | 5 test case JSON repair |

## Cấu hình

- Knowledge base: 29.894 ký tự
- Chunking: `chunk_size=500`, `overlap=50` - 107 chunks
- Vector store: FAISS, retriever `k=3`
- Bộ đánh giá: 50 cặp QA có đáp án chuẩn

## Kết quả RAGAS

| Chỉ số | V1 (ngắn gọn) | V2 (chuyên gia) | Chênh lệch |
|---|---|---|---|
| faithfulness | **0.9342** | 0.8827 | 0.0514 |
| answer_relevancy | **0.9163** | 0.8957 | 0.0206 |
| context_recall | 1.0000 | 1.0000 | 0.0000 |
| context_precision | **0.9450** | 0.9417 | 0.0033 |

Mục tiêu faithfulness >= 0.8: đạt ở cả hai phiên bản.

## Phân tích V1 vs V2

Khác biệt giữa hai prompt: V1 yêu cầu trả lời trực tiếp trong 2-4 câu, V2 yêu cầu xác định facts liên quan rồi viết có tổ chức trong 3-5 câu. Cả hai giữ nguyên ràng buộc "chỉ dùng context, không suy đoán".

Đo trên 100 câu trả lời trong `data/rag_outputs_v*.json`:

| | V1 | V2 |
|---|---|---|
| Độ dài trung bình | 293 ký tự | 508 ký tự |
| Số câu trung bình | 2.0 | 3.5 |

V2 dài hơn 73%, và số câu của mỗi phiên bản rơi đúng khoảng được yêu cầu. Hai prompt tạo ra hai hành vi khác nhau thật, không phải khác trên giấy.

### Vì sao V1 thắng faithfulness

Chính độ dài giải thích khoảng cách 0.0514. `faithfulness` tính tỷ lệ mệnh đề trong câu trả lời được context chống lưng. Câu trả lời càng dài thì càng nhiều mệnh đề, mỗi mệnh đề thêm vào là một cơ hội trượt khỏi context. V1 trung bình 2.0 câu nên bề mặt rủi ro nhỏ; V2 trung bình 3.5 câu, và yêu cầu "viết có tổ chức" đẩy mô hình bổ sung câu dẫn, câu nối, câu tổng kết - những chỗ dễ nói thêm điều context không có.

Cùng cơ chế đó giải thích `answer_relevancy` (chênh 0.0206): phần mở rộng của V2 mang thêm ngữ cảnh phụ, làm loãng mức độ đi thẳng vào câu hỏi.

### Vì sao không đọc `context_recall` và `context_precision`

Hai chỉ số này chấm retriever chứ không chấm prompt. Retriever ở V1 và V2 là một - cùng FAISS index, cùng `k=3`, cùng câu hỏi - nên giá trị lý thuyết phải bằng nhau. `context_recall` đúng là 1.0000 ở cả hai. Phần lệch 0.0033 của `context_precision` là nhiễu từ LLM giám khảo vốn không tất định, không phải bằng chứng V1 truy xuất tốt hơn.

### Kết luận

V1 thắng ở hai chỉ số thật sự đo prompt, và thắng vì lý do có thể chỉ ra được chứ không phải may rủi: trả lời ngắn thì ít cơ hội bịa hơn. Đánh đổi là V2 cho câu trả lời đầy đủ và dễ đọc hơn, thứ mà 4 chỉ số RAGAS không đo.

Tiêu chí chọn nên là bối cảnh dùng. Ưu tiên độ chính xác và chatbot cần gọn thì V1. Cần tài liệu kỹ thuật đầy đủ thì V2, kèm chấp nhận faithfulness thấp hơn khoảng 5 điểm phần trăm.

## Tái lập

```bash
cd src
python run_all.py
```
