# 法律检索评测集

`benchmark_v0.jsonl` 是当前语料上的第一版检索基准，共 24 条：

| 任务 | dev | test | 合计 |
|---|---:|---:|---:|
| `exact_law` | 4 | 2 | 6 |
| `risk` | 4 | 2 | 6 |
| `compare` | 4 | 2 | 6 |
| `case_search` | 4 | 2 | 6 |
| 合计 | 16 | 8 | 24 |

所有样本当前都是 `silver`：query 和标签由现有法规、案例事实及结构化关系反向构造，并已自动确认 `evidence_id` 存在，但尚未经过法律专业人员的独立复核。因此它可以用于建立 baseline 和调试，不能直接宣称为人工金标。

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
    "status": "silver",
    "method": "evidence_anchored_manual_draft",
    "reviewer": null,
    "reviewed_at": null
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

## 从 silver 升级为 gold

建议由两名复核者独立完成：

1. 不看当前标签，只阅读 query 并从候选池判断哪些证据相关；
2. 检查 query 是否泄露案例标题、是否存在多个同等相关但漏标的证据；
3. 分别给出 grade、required 和 rationale；
4. 对分歧进行仲裁；
5. 写入真实 `reviewer`、ISO 日期 `reviewed_at`，将 `status` 改为 `gold`。

复核尤其要关注风险题是否把“相似案例”错误标成确定违法，以及比较题是否遗漏某个法域的同等相关条款。`expected_limitations` 当前用于人工检查答案边界，尚未自动计算文本生成指标。
