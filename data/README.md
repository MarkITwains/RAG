# Data directory

公开仓库不包含原始语料、标准文档全文或向量缓存。

请将你有权使用的 PCB 文档放入：

```text
data/clear_docs/
```

支持的文档类型由入库脚本和 LlamaIndex Reader 决定，常见格式包括 `.pdf`、`.txt`、`.docx`、`.md` 等。

不要将业务数据、内部资料、版权受限全文、模型缓存或 `lexical_corpus.jsonl` 提交到 GitHub。
