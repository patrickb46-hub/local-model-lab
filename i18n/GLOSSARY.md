# Translation Glossary & Style Guide · 中译英统一规范

本文件是 Local-Model-Lab 英文版的**唯一术语标准**。翻译任何页面时必须遵守，
否则 11 个页面之间会出现术语打架。

---

## 一、硬性约束（违反即返工）

1. **保留全部代码、命令、路径、文件名、URL 原样**，不要翻译：
   `ask.sh` · `scripts/30_evaluate.py` · `bench_stratified.jsonl` · `eval_raw.jsonl` ·
   `~/Desktop/AI/Local-Model-Lab` · `python3 -m http.server` · `pip install` 等。
2. **数字、百分比、表格结构一个都不能改。** 56.2% 就是 56.2%，52 题就是 52 题。
3. **HTML 文件**：只替换**可见文字文本节点**（含 `<title>`、`<svg><title>` 提示、
   `alt`、`aria-label`）。**绝对不要动** `<style>` 里的 CSS、SVG 的坐标/宽高/fill/
   class/d 属性、表格的 colspan/rowspan、任何标签结构。译文不得增删任何 HTML 标签。
4. **Markdown 文件**：保留全部标题层级、表格、代码块围栏、链接目标、引用块结构。
   中文标题译成英文标题。表格表头与单元格内容翻译，但对齐结构不动。
5. **专有名词保持原样**：Qwen3.5-2B · Qwen3.5-9B · DeepSeek V4 Pro · Ollama · LoRA ·
   GRPO · Qwen · Hugging Face · PyPI · conda-forge · PyTorch / torch · transformers ·
   PEFT · MPS · Intel MacBook Pro（2018, i7-8850H）· macOS · Aliyun mirror · WorkBuddy。
6. 不要加译者注、不要加解释性括号、不要"improve"原文。**忠实是第一位的。**
7. 中文标点（，。、：；「」）换成英文标点（, . : ; " "），但中文引号「」内若是专有名词
   或命令，保持原样。

---

## 二、核心术语表

| 中文 | 英文（统一用这个） |
|---|---|
| 本地模型实验室 | Local Model Lab |
| 蒸馏 | distillation |
| 强化学习 | reinforcement learning (RL) |
| 老师模型 / 大模型 | teacher model |
| 学生模型 / 小模型 | student model |
| 原版 / base | base (the original) |
| 补丁 / adapter | LoRA adapter |
| 混合 / 先抄后刷 | hybrid (distill-then-RL) |
| 考卷 / 评测集 | evaluation set / benchmark |
| 冻结（考卷） | frozen |
| 分层 | tiers |
| L1 基础题 | L1 basics |
| L2 老师不会 | L2 teacher-fails |
| L3 都不会 | L3 both-fail |
| 横评 | head-to-head comparison |
| 一致性 | consistency |
| 多样性 | diversity |
| 格式合规 | format compliance |
| 准确率 | accuracy |
| 与教师模型相似度 | teacher similarity |
| 采样 | sampling |
| 作答 / 一次作答 | attempt / response |
| 配对检验 | paired test |
| McNemar 精确检验 | McNemar's exact test |
| 显著 / 不显著 | significant / not significant |
| p 值 | p-value |
| 云基线 | cloud baseline |
| token 预算 | token budget |
| 思考块 / 思考模式 | thinking block / thinking mode |
| 推理模型 | reasoning model |
| 量化 | quantization |
| 微调 | fine-tuning |
| 管线 | pipeline |
| 探针 | probe |
| 判分 / 评分 | grading |
| 主指标 | primary metric |
| 出题 | problem generation |
| 筛选 | screening |
| 训练轮次 | epochs |
| 学习率 | learning rate |
| 数据集 | dataset |
| 提示词 | prompt |
| 结构化输出 | structured output |
| 纯 CPU | CPU-only |
| 踩坑记录 | Pitfalls Log |
| 经验教训总纲 | Key Lessons Learned |
| 原理笔记 | How It Works (Technical Notes) |
| 完整计划 | The Full Plan |
| 成果展示 | Results Showcase |
| 项目全貌 | Project Overview |
| 评测报告 | Evaluation Report |
| 文档笔记 | Docs & Notes |

### 图表与角色标签（短标签，直接用于 SVG / 表格）

| 中文 | 英文短标签 |
|---|---|
| 原版 | Base |
| A1 蒸馏 | A1 · Distilled |
| B 强化 | B · RL |
| C 混合 | C · Hybrid |
| 老师 | Teacher |
| 云端 / dsv4pro | Cloud |
| L1 基础题 | L1 Basics |
| L2 老师不会 | L2 Teacher-fails |
| L3 都不会 | L3 Both-fail |
| 总分 | Overall |

### 导航栏标签

首页 → **Home** ｜ 项目全貌 → **Overview** ｜ 评测报告 → **Report** ｜
文档笔记 → **Docs** ｜ 源码 → **Source**

---

## 三、语气与风格

- **项目全貌（overview）**：原文是"讲给非技术读者听"的白话，带「小学生 / 大学生」的比喻。
  英文要保持这个**平易近人的讲故事语气**，不要翻译成学术腔。
  比喻统一为：小学生 = **the small student**，大学生 = **the big student**（首次出现时
  可写 "the small model — think of it as an elementary-school student"）。
- **评测报告（report）与技术文档**：**技术、精确、克制**。术语用标准英文表达，
  不要口语化。表格与数字密集，行文要短促清楚。
- **不要用营销腔**（"revolutionary"、"game-changing"）。原文是什么力度就译什么力度。
- 中文的破折号 —— 在英文里用 em dash（—），或用逗号/冒号拆句，读顺为准。
- 中文的"⚠️ 注意"这类提示保留 emoji。
- 第一人称：原文用"我们"时英文可用 "we"；泛指时用被动或 "the experiment"。

---

## 四、输出约定

- 输出文件路径由调用方指定，写入 `i18n/` 目录。
- 只输出译文文件本身，**不要**在文件里附加"Translated by..."之类的署名或说明。
- 完成后回报：输出路径 + 字数 + 是否有任何原文内容无法直译（需要人工判断的地方）。
