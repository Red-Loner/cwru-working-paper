# Prompt Log

> 记录所有与 AI 助手的交互：用了什么 prompt、得到了什么输出、做了什么决策。

| # | Date | Time | Tool | Prompt Summary | Output Summary | Decision Made |
|---|------|------|------|---------------|----------------|---------------|
| 1 | 2026-07-20 | — | Claude (opencode) | Generated project repository skeleton, dataset card, split design, and leakage audit | Repository created at E:\Git\cwru-working-paper, pushed to GitHub | Adopted file-level split design; leaked protocol documented |

## Design Prompts (to be executed with LLM)

### §4.1 Research Question Discovery
```text
I am designing a CWRU-based bearing fault diagnosis working paper.
[Paste: manual.md §4.1 prompt]
```

### §5.1 Dataset Discovery (Second Dataset)
```text
Find public datasets similar to the CWRU bearing dataset for machinery fault diagnosis.
[Paste: manual.md §5.1 prompt]
```

### §6.3 Leakage Audit
```text
Audit this proposed split for leakage risk.
[Paste: manual.md §6.3 prompt with dataset card and split design]
```

### §7.3 Method Design
```text
I have the following research question:
[Paste: research question brief]
Propose 3 candidate methods that could address the bottleneck.
[Paste: manual.md §7.3 prompt]
```

### §8 Baseline Design
```text
Given my research question and method card, design fair baselines.
[Paste: manual.md §8 prompt]
```

### §9.2 Mechanism Validation
```text
Design mechanism validation tests for this method.
[Paste: manual.md §9.2 prompt]
```
