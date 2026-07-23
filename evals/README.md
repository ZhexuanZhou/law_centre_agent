# 法律检索评测集

`benchmark_v0.jsonl` 是当前语料上的第一版检索基准，共 24 条：

| 任务 | dev | test | 合计 |
|---|---:|---:|---:|
| `exact_law` | 4 | 2 | 6 |
| `risk` | 4 | 2 | 6 |
| `compare` | 4 | 2 | 6 |
| `case_search` | 4 | 2 | 6 |
| 合计 | 16 | 8 | 24 |

所有样本当前都是 `gold`，复核日期为 2026-07-23。标签依据当前 `data/corpus_v3.sqlite3` 的法规全文、案例事实与裁决摘要、案例—法条关系，以及现有 Agent 的真实检索候选池逐条裁决。复核同时修正了风险题的适用法域歧义、同等相关案例漏标和 coverage group 的替代证据关系。

该状态的准确含义是“当前语料快照上的 AI 辅助检索金标”，不是两名法律专业人员独立复核后的专家法律意见。`annotation.method` 固定记录为 `ai_assisted_pooled_corpus_review`；新增或更新语料后必须重新 pooling 和裁决。

## 数据结构

每行是一个独立 JSON 对象，正式结构见 `benchmark.schema.json`。

```json
{
  "id": "risk_001",
  "split": "dev",
  "task": "risk",
  "query": "用户输入",
  "language": "zh-CN",
  "difficulty": "hard",
  "retrieval_k": 8,
  "relevance": [
    {
      "evidence_id": "case:gdprhub:10052",
      "grade": 3,
      "required": true,
      "rationale": "事实高度相似"
    }
  ],
  "coverage_groups": [
    {
      "name": "case",
      "evidence_ids": ["case:gdprhub:10052"],
      "min_hits": 1
    }
  ],
  "expected_limitations": ["GDPRhub 是二手案例摘要。"],
  "tags": ["consent"],
  "annotation": {
    "status": "gold",
    "method": "ai_assisted_pooled_corpus_review",
    "reviewer": "OpenAI Codex",
    "reviewed_at": "2026-07-23"
  }
}
```

### 相关性等级

| grade | 含义 |
|---:|---|
| 3 | 直接回答 query 或事实高度相似，是核心证据 |
| 2 | 明确支持部分结论，但单独不足以完整回答 |
| 1 | 有用背景或案例引用法条，不是主要答案 |

`required=true` 表示完整回答必须召回该证据。`coverage_groups` 表达集合约束，例如比较三个法域时，每个法域至少命中一个证据。这样不会把“命中某法域的任意一个充分材料”错误建模成“必须召回该法域的全部材料”。

## dev 与 test

- 只使用 `dev` 调整 planner prompt、query expansion、过滤、rerank prompt、候选规模和阈值；
- `test` 在最终确认方案前保持不可见，不应逐条观察后再调参；
- 如果根据 test 失败样本修改系统，该 test 已经泄漏，应把它移入 dev，并补充新的盲测样本。

## 验证数据

以下命令会检查 JSONL 结构、重复 ID、grade、coverage group，以及所有 evidence 是否真实存在于索引：

```bash
python -m legal_agentic_retrieval.eval_cli validate \
  --dataset evals/benchmark_v0.jsonl \
  --index data/corpus_v3.sqlite3
```

## 运行 Agent

先只运行 dev：

```bash
python -m legal_agentic_retrieval.eval_cli run \
  --dataset evals/benchmark_v0.jsonl \
  --split dev \
  --index data/corpus_v3.sqlite3 \
  --env-file .env \
  --output evals/results/dev.jsonl
```

运行中断后可添加 `--resume`。使用 `--task exact_law` 或 `--limit 2` 可以做小规模 smoke test。默认使用 `reference-only`，只评估检索；添加 `--with-answer` 才会调用答案生成。

## 计算指标

```bash
python -m legal_agentic_retrieval.eval_cli score \
  --dataset evals/benchmark_v0.jsonl \
  --results evals/results/dev.jsonl \
  --split dev
```

输出包括整体、按任务和按 split 的：

- `Recall@K`：所有分级相关证据的召回比例；
- `RequiredRecall@K`：必须证据的召回比例；
- `MRR@K`：第一个相关证据的倒数排名；
- `nDCG@K`：考虑 1–3 级相关性的排序质量；
- `Coverage@K`：法域、来源或对比对象的覆盖组满足比例；
- `Precision@K`：返回结果中已标相关的比例。

小数据集上的绝对分数波动较大，应先看任务分组和具体失败样本，不要只优化总体均值。

## Gold 复核记录

本版执行了以下步骤：

1. 核对 24 条 query 的任务类型、法域和预期回答范围；
2. 阅读每条已标法规/案例的完整索引文本，而不只检查 `evidence_id` 是否存在；
3. 检查 `case_law_relations` 中的结构化引用，补充能直接支持问题的法条；
4. 用当前 planner、embedding 和 Qwen rerank 链路分别运行 dev/test，形成待裁决候选池；
5. 对候选池中的未标证据逐条判为 1–3 级相关或无关；
6. 修正替代案例的 coverage group、`required` 约束、rationale 和 answer limitation；
7. 运行 Schema、索引存在性、重复 ID 和 coverage 一致性验证。

本次实质修订包括：

- 风险题明确写入挪威、比利时、克罗地亚、意大利或法国等适用背景，避免 planner 在缺少法域时作出另一种合理解释；
- `case_search_003` 和 `compare_006` 加入同样满足条件的 `case:gdprhub:7843`；
- 风险题加入真实检索池中确认相关的补充案例和法条，并使用 2 级标签区分“部分支持”；
- 对可由多个案例满足的问题使用 `coverage_groups` 表达替代关系，不再把每个候选都错误设为 required；
- 对二手案例摘要、跨法域案例和事实差异写入 `expected_limitations`。

`test` 候选只用于本次标签裁决，没有据此修改检索算法。gold 标签冻结后，后续调参仍只能使用 `dev`；如果根据 test 失败样本修改系统，应将相应样本移入 dev 并补充新的盲测样本。

若需要“专家级 gold”，建议再由两名法律专业人员独立盲审 grade、required 和 rationale，并对分歧进行仲裁。`expected_limitations` 当前用于人工检查答案边界，尚未自动计算文本生成指标。
