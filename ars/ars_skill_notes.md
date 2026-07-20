# ARS Skill Notes

> 来源: https://github.com/Imbad0202/academic-research-skills

## 1. ARS 能帮助什么？

- **Deep Research**（13-agent 研究团队）：文献检索、系统综述（PRISMA）、苏格拉底式引导研究、事实核查、论文对比
- **Academic Paper**（12-agent 论文写作）：风格校准、写作质量检查、LaTeX 硬化、可视化、修订指导、引用转换
- **Academic Paper Reviewer**（7-agent 审稿）：多视角同行评审、0-100 质量评分、魔鬼代言人质疑、R&R 追溯矩阵
- **Academic Pipeline**（10 阶段编排）：完整性验证、Material Passport、可复现锁定、跨模型验证
- 文献组织、论文大纲、论证图谱、引用格式、审稿模拟、修回规划、披露起草

## 2. ARS 不能做什么？

- 不能发明实验结果
- 不能在没有证据的情况下决定研究声明
- 不能替代人类解释（human interpretation）
- 不能隐藏 AI 使用
- 不能跳过引用验证
- 不能自己跑实验（需要搭配 experiment-agent）
- 受限于 frame-lock（在同一认知框架内运作）、sycophancy（对反驳过快让步）、意图误判

## 3. ARS 的完整性关卡（Integrity Gates）

- **Stage 2.5**（写作后预审前）：7 模式阻塞检查清单（AI 研究失败模式），检查引用真实性、数据造假、捷径依赖
- **Stage 4.5**（修回后终审前）：零回归确认，最终完整性审计
- 两个关卡都是 **MANDATORY**，不可跳过
- v3.8 新增：可选引用审计（`ARS_CLAIM_AUDIT=1`），逐条拉取被引来源验证
- Material Passport 记录实验出处（experiment provenance）

## 4. ARS 如何减少幻觉引用和未支持声明？

- 引用验证关卡：检查每一条引用是否存在、是否支持对应论述
- Locator anchor 基础设施（v3.7.3）：每条引用携带三级定位锚点
- Style Calibration：从历史作品中学习作者文风，减少机器写作痕迹
- Writing Quality Check：检测机器写作模式
- 反泄漏协议（anti-leakage protocol）
- VLM 图形验证

## 5. 人类研究者必须决定什么？

- **研究问题的定义**（选择什么瓶颈、提出什么假设）
- **数据集的选择**（用哪些数据集、为什么）
- **切分设计**（如何避免泄漏，选择什么切分策略）
- **实验证据的收集与判断**（什么算有效证据）
- **引用验证**（每一条引用的真实性和相关性）
- **结果解释**（物理含义、因果关系）
- **结论边界**（适用范围、局限性）
- **AI 使用披露**（哪些部分用了 AI、怎么用的）

## 本项目的 ARS 使用计划

1. 使用 ARS 研究模式精炼 RQ brief
2. 使用 ARS 文献综述支持构建文献矩阵
3. 使用 ARS 论文写作从证据包产出大纲和初稿
4. 运行完整性检查
5. 运行审稿模拟
6. 逐条修回回应
7. 终审完整性检查
8. 发布工作组包
