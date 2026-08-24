"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import importlib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Phải import TRƯỚC ragas — vá lỗi ragas 0.4.x import module vertexai đã bị gỡ
from utils import ragas_compat  # noqa: F401

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# Import trực tiếp 2 system prompt từ Bước 2 — đảm bảo giống hệt bản đã push lên
# Prompt Hub. Copy tay chỉ cần lệch một ký tự là phép so sánh V1/V2 mất ý nghĩa.
_step2 = importlib.import_module("02_prompt_hub_ab_routing")

SYSTEM_V1 = _step2.SYSTEM_V1
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = _step2.SYSTEM_V2
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

# Số request RAGAS được phép chạy song song. Tăng lên nếu tài khoản LLM
# của bạn có hạn mức cao; để 2 cho free tier.
RAGAS_MAX_WORKERS = 1


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=800, chunk_overlap=80)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    # LIST of strings — RAGAS cần từng đoạn riêng để chấm context_recall
    # và context_precision. Ghép sớm là hỏng cả hai chỉ số đó.
    contexts = [doc.page_content for doc in docs]

    # Chuỗi ghép chỉ dùng để đổ vào placeholder {context} của prompt
    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    # str() vì StrOutputParser trả về TextAccessor, RAGAS cần chuỗi thật
    return {"answer": str(answer), "contexts": contexts}


def _cache_path(prompt_version: str):
    """Đường dẫn file cache kết quả RAG cho một prompt version."""
    return Path(__file__).parent.parent / "data" / f"rag_outputs_{prompt_version}.json"


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts

    Kết quả được cache ra data/rag_outputs_<version>.json. Giai đoạn sinh câu
    trả lời tốn 50 lượt gọi LLM mỗi version; nếu bước chấm điểm phía sau hỏng
    (rate limit, hết credit) thì lần chạy lại dùng luôn cache thay vì đốt thêm
    100 lượt gọi. Xóa file cache nếu bạn đổi prompt.
    """
    cache = _cache_path(prompt_version)
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if len(cached) == len(QA_PAIRS):
            print(f"\n♻️  Dùng lại {len(cached)} kết quả đã cache cho prompt {prompt_version}")
            print(f"   ({cache})")
            return cached
        print(f"\n⚠️  Cache {prompt_version} có {len(cached)}/{len(QA_PAIRS)} mẫu — chạy lại từ đầu")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    cache.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Đã cache {len(results)} kết quả vào {cache}")
    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    # Mặc định RAGAS bắn 16 request song song. Tài khoản free tier của
    # OpenRouter/Gemini có hạn mức in-flight rất thấp → 402/429 hàng loạt và
    # mọi điểm trả về NaN. Hạ xuống 2 luồng: chậm hơn nhưng chạy trọn.
    run_config = RunConfig(max_workers=RAGAS_MAX_WORKERS, timeout=180, max_retries=10)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=run_config,
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores   = {}
    coverage = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw   = result[key]
        valid = [v for v in raw if v is not None and not np.isnan(v)]
        coverage[key] = {"scored": len(valid), "total": len(raw)}
        if not valid:
            # Toàn bộ mẫu lỗi (thường do rate limit) — ghi None thay vì NaN,
            # vì NaN không phải JSON hợp lệ và sẽ làm hỏng ragas_report.json
            print(f"  ⚠️  {key}: không có mẫu nào chấm được ({len(raw)} mẫu đều lỗi)")
            scores[key] = None
        else:
            if len(valid) < len(raw):
                print(f"  ⚠️  {key}: chỉ {len(valid)}/{len(raw)} mẫu chấm được")
            scores[key] = float(np.mean(valid))

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        if v is None:
            print(f"  {k:30s}: n/a")
            continue
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores, coverage


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    """Chạy toàn bộ Bước 3: sinh câu trả lời cho cả 2 version, chấm 4 chỉ số RAGAS, lưu báo cáo."""
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores, v1_coverage = run_ragas_eval(v1_results, "v1")
    v2_scores, v2_coverage = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        if s1 is None or s2 is None:
            f1 = "n/a" if s1 is None else f"{s1:.4f}"
            f2 = "n/a" if s2 is None else f"{s2:.4f}"
            print(f"  {metric:30s}  {f1:>8}  {f2:>8}  —")
            continue
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    faiths = [s for s in (v1_scores["faithfulness"], v2_scores["faithfulness"])
              if s is not None]
    best_faith = max(faiths) if faiths else None
    if best_faith is None:
        print("\n❌  Không chấm được mẫu nào — nhiều khả năng bị rate limit.")
        print("   Hạ RAGAS_MAX_WORKERS xuống 1, hoặc đổi PROVIDER sang tài khoản có hạn mức cao hơn.")
    elif best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} >= 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # Lưu báo cáo vào data/ragas_report.json và evidence/
    # Số mẫu thực sự chấm được. Nếu thấp hơn tổng số QA pairs thì điểm bên trên
    # KHÔNG đại diện cho cả bộ dữ liệu — thường do rate limit hoặc hết credit.
    min_scored = min(c["scored"] for c in list(v1_coverage.values()) + list(v2_coverage.values()))
    complete   = min_scored == len(QA_PAIRS)
    if not complete:
        print(f"⚠️  CẢNH BÁO: có chỉ số chỉ chấm được {min_scored}/{len(QA_PAIRS)} mẫu.")
        print("   Điểm bên trên KHÔNG đại diện cho cả bộ 50 câu — đừng nộp kết quả này.")

    report = {
        "n_qa_pairs": len(QA_PAIRS),
        "evaluation_complete": complete,
        "coverage_v1": v1_coverage,
        "coverage_v2": v2_coverage,
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": bool(best_faith is not None and best_faith >= 0.8),
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")

    # Chép luôn sang evidence/ — rubric yêu cầu cả 2 nơi
    evidence_path = Path(__file__).parent.parent / "evidence" / "03_ragas_report.json"
    evidence_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Đã chép sang {evidence_path}")


if __name__ == "__main__":
    main()
