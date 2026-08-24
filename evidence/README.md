# Bằng chứng — Day 22: LangSmith + Prompt Versioning

**Học viên:** Trần Minh Hạnh · **Mã:** 2A202601232
**LangSmith project:** `day22-lab`

---

## Danh mục tệp

| Tệp | Nội dung |
|---|---|
| `01_langsmith_traces.png` | Giao diện LangSmith, project `day22-lab` |
| `02_prompt_hub.png` | Prompt Hub với 2 phiên bản `tranminhhanh-rag-v1` và `-v2` |
| `02_ab_routing_log.txt` | Log console Bước 2: push/pull Hub + 50 câu có nhãn v1/v2 |
| `03_ragas_scores.png` | Bảng so sánh RAGAS V1 vs V2 trên terminal |
| `03_ragas_report.json` | Bản sao của `data/ragas_report.json` |
| `04_pii_demo_log.txt` | 6 test case PII detection & redaction |
| `04_json_demo_log.txt` | 5 test case JSON repair |

---

## Cấu hình chạy

| Hạng mục | Giá trị |
|---|---|
| Knowledge base | `data/knowledge_base.txt`, 29.894 ký tự |
| Chunking | `RecursiveCharacterTextSplitter`, `chunk_size=500`, `overlap=50` → 107 chunks |
| Vector store | FAISS, retriever `k=3` |
| Bộ dữ liệu đánh giá | 50 cặp QA có đáp án chuẩn (`src/qa_pairs.py`) |
| Chỉ số RAGAS | faithfulness, answer_relevancy, context_recall, context_precision |

---

## Kết quả RAGAS — V1 vs V2

| Chỉ số | V1 (ngắn gọn) | V2 (chuyên gia) | Chênh lệch |
|---|---|---|---|
| **faithfulness** | 0.9630 | **0.9680** | +0.0050 nghiêng V2 |
| **answer_relevancy** | **0.9085** | 0.9069 | +0.0016 nghiêng V1 |
| **context_recall** | 1.0000 | 1.0000 | bằng nhau |
| **context_precision** | **0.9833** | 0.9752 | +0.0081 nghiêng V1 |

**Độ phủ mẫu:** V1 chấm được 50/50 ở cả 4 chỉ số. V2 chấm được 48/50 (faithfulness), 49/50 (answer_relevancy, context_recall), 47/50 (context_precision) — số mẫu hụt là do rate limit của nhà cung cấp LLM khi chấm, không phải lỗi pipeline.

Mục tiêu faithfulness ≥ 0.8: **đạt ở cả hai phiên bản**, và cả hai đều vượt mốc 0.9.

---

## Phân tích: vì sao V1 và V2 chênh nhau như vậy

### Hai prompt khác nhau ở đâu

- **V1 — ngắn gọn:** yêu cầu trả lời trực tiếp trong 2-4 câu.
- **V2 — chuyên gia:** yêu cầu đọc kỹ context, xác định facts liên quan, rồi viết câu trả lời có tổ chức trong 3-5 câu.

Cả hai đều có cùng ràng buộc "chỉ dùng context, không suy đoán".

### Prompt có được LLM tuân thủ không

Đo trực tiếp trên 100 câu trả lời đã sinh (`data/rag_outputs_v*.json`):

| | V1 | V2 |
|---|---|---|
| Độ dài trung bình | 293 ký tự | 508 ký tự |
| Trung vị | 309 | 512 |
| Số câu trung bình | 2.0 | 3.5 |

V2 dài hơn V1 **73%**, và số câu (2.0 vs 3.5) rơi đúng vào khoảng mà mỗi prompt yêu cầu (2-4 và 3-5). Hai prompt thực sự tạo ra hai hành vi khác nhau, không phải khác biệt trên giấy.

### Đọc các chỉ số

**`context_recall` và `context_precision` không phản ánh prompt.** Hai chỉ số này chấm chất lượng *retriever*, mà retriever ở V1 và V2 là **một** — cùng FAISS index, cùng `k=3`, cùng câu hỏi. Về lý thuyết chúng phải bằng nhau tuyệt đối. `context_recall` đúng là 1.0000 ở cả hai. Riêng `context_precision` lệch 0.0081, và phần lệch đó đến từ hai nguồn: LLM giám khảo không hoàn toàn tất định, và V2 chỉ chấm được 47/50 mẫu nên mẫu số khác nhau. Đây là nhiễu đo lường, không phải bằng chứng V1 truy xuất tốt hơn.

**`faithfulness` — V2 nhỉnh hơn 0.0050.** Cách diễn giải hợp lý: prompt V2 buộc mô hình "xác định các facts liên quan" trước khi viết, nên câu trả lời bám sát câu chữ của context hơn. Nhưng 0.0050 trên 48-50 mẫu là quá nhỏ để kết luận chắc chắn.

**`answer_relevancy` — V1 nhỉnh hơn 0.0016.** Chỉ số này đo mức độ câu trả lời đi thẳng vào câu hỏi. Câu trả lời dài hơn của V2 mang thêm ngữ cảnh phụ, về nguyên tắc có thể làm loãng độ liên quan. Nhưng 0.0016 thì nhỏ hơn cả sai số của một lần chạy lại.

### Kết luận

**Không có phiên bản nào thắng.** Cả 4 chênh lệch đều dưới 0.01, trong khi bản thân LLM giám khảo đã dao động hơn thế giữa các lần chạy — hai lần chạy trước đó trên cùng bộ dữ liệu cho faithfulness V1 lần lượt 0.9355 và 0.9342. Nói "V1 tốt hơn vì context_precision cao hơn 0.0081" là đọc nhiễu thành tín hiệu.

Điều rút ra được thì rõ ràng hơn: **cả hai prompt đều đạt yêu cầu chống bịa đặt** (faithfulness > 0.96), và câu ràng buộc "chỉ dùng context, không suy đoán" — thứ duy nhất giống nhau ở cả hai — nhiều khả năng mới là yếu tố quyết định con số đó, chứ không phải phần văn phong khác nhau.

Khác biệt thật sự giữa V1 và V2 là **độ dài và cấu trúc câu trả lời** (293 vs 508 ký tự), và 4 chỉ số RAGAS này không đo thứ đó. Muốn chọn giữa hai prompt thì tiêu chí nên là bối cảnh sử dụng — chatbot cần gọn thì chọn V1, tài liệu kỹ thuật cần đầy đủ thì chọn V2 — chứ không phải bảng điểm trên.

Muốn phân định bằng số thì cần: bộ dữ liệu lớn hơn 50 câu, chạy lặp nhiều lần để ước lượng phương sai, và thêm chỉ số đo được độ đầy đủ của câu trả lời.

---

## Tái lập kết quả

```bash
cd src
python 01_langsmith_rag_pipeline.py
python 02_prompt_hub_ab_routing.py
python 03_ragas_evaluation.py      # dùng cache trong data/rag_outputs_v*.json nếu có
python 04_guardrails_validator.py

# hoặc chạy tất cả
python run_all.py
```
