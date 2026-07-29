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

## 导出服务标注 CSV

以下命令把嵌套的 gold JSONL 展开为长表：一行对应一个
`query × evidence` 判断，并从 SQLite 补齐法规全文或案例事实与裁决。
这条命令不调用 LLM、embedding 或 reranker，也不需要 `.env`：

```bash
python -m legal_agentic_retrieval.eval_cli export-csv \
  --dataset evals/benchmark_v0.jsonl \
  --index data/corpus_v3.sqlite3 \
  --output evals/benchmark_v0.service_annotation.csv
```

CSV 使用带 BOM 的 UTF-8 编码，方便 Excel 和外部标注服务识别中文。
为了让法务只判断“这个 evidence 对当前 query 是否相关、是否不可替代”，
表中固定只有 7 列：

| 列 | 用途 |
|---|---|
| `sample_id` | 用于把法务结论准确回写到原测试样本 |
| `task` | `exact_law`、`risk`、`compare` 或 `case_search` |
| `query` | 用户问题 |
| `evidence_id` | 用于把结论准确回写到具体法规或案例 |
| `evidence` | 合并后的标题、引用、法域和完整证据正文 |
| `is_relevant` | 该证据是否支持 query，只允许 `true` 或 `false` |
| `is_required` | 缺少该证据是否无法完整回答 query，只允许 `true` 或 `false` |

表中不暴露 `gold_grade`、`required`、coverage group 或原标注理由，避免影响法务的
独立判断。原始 `benchmark_v0.jsonl` 仍是包含完整 gold 标签的内部答案文件。

法务填写时应按 `sample_id` 查看同一 query 下的全部 evidence，并遵守：

- `is_relevant=false` 时，`is_required` 必须填写 `false`；
- 证据相关且没有同等替代证据时，填写 `is_required=true`；
- 多条证据可以相互替代时，它们可以都是 relevant，但单条应填写
  `is_required=false`；
- coverage group 不要求法务填写，收回结果后由评测维护者根据同一 query 的
  可替代证据关系更新。

默认导出完整证据。只有标注平台存在单元格长度限制时才使用
`--text-limit 30000`，但截断可能影响法务判断，因此不建议常规使用。
可通过 `--split dev`、`--split test` 或 `--task risk` 单独导出子集。

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

## 外部法务二元复核记录（2026-07-29）

外部法务使用 `benchmark_v0.service_annotation.csv` 对 93 条
`query × evidence` 记录填写了 `is_relevant` 和 `is_required`。维护者将结果与
现有 Gold、coverage group 和父子条款关系逐项仲裁，采用以下规则：

- `is_relevant` 是法务判断：证据可以是直接答案，也可以是支持案例认定的相关
  法规；因此 `case_search` 中的相关法规继续保留，不要求结果只能包含案例；
- `required` 是评测结构判断，不等同于“法律上有用”。它表示缺少该条具体
  evidence 后，现有相关证据集合无法完整覆盖 query；
- 多条证据可互相替代时，单条保持 `required=false`，由 coverage group 的
  `min_hits` 表达集合要求；
- 父条文和子条款覆盖同一命题时，不同时标为 required；
- coverage group 只有一条 evidence 时，该 evidence 必须
  `required=true`。加载 benchmark 时会自动校验这一不变量；
- 后续法务主要复核 relevance 和法律理由；required 与 coverage group 由评测
  维护者结合全部候选统一建模，不再直接批量导入逐行 required 结果。

本轮接受两项修正：

- `compare_001` 的 `PIPL Article 13` 改为 required；
- `risk_005` 的 `GDPR Article 83` 改为 required。

本轮没有接受三项“唯一案例改为非 required”的建议：

- `case_search_002` 的 `case:gdprhub:10052`；
- `risk_004` 的 `case:gdprhub:9868`；
- `compare_006` 的 `case:gdprhub:9994`。

其余 required 分歧涉及替代证据、父子条款重复或同组多候选，保持原 Gold。
每次接受复核修正后都必须重新运行 benchmark validate，并重新计算 dev、test 和
综合指标；旧结果不得继续作为当前 Gold 的正式指标。
