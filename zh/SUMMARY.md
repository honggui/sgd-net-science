# Summary

* [首页](README.md)

## 入门

* [从这里开始](guide/start-here.md)
* [术语与约定](guide/glossary.md)
* [研究状态](guide/status.md)

## 研究定位

* [01 · 论文与研究问题](docs/01.md)
* [32 · 融合评估与创新边界](docs/32.md)
* [47 · 通用性与能力边界](docs/47.md)

## 核心模型架构

* [36 · 统一架构与组件全集](docs/36.md)
  * [摘要与详细版阅读方法](docs/chapters/36/section-01.md)
  * [1. 架构决策与阅读入口](docs/chapters/36/section-02.md)
  * [2. 与已有材料的对应](docs/chapters/36/section-03.md)
  * [3. 系统连接图](docs/chapters/36/section-04.md)
  * [4. 组件与单元全集：9 组、72 个登记单元](docs/chapters/36/section-05.md)
  * [5. 张量、状态和时钟合同](docs/chapters/36/section-06.md)
  * [6. 一个块到底如何计算](docs/chapters/36/section-07.md)
  * [7. 初始配置与可复现前向顺序](docs/chapters/36/section-08.md)
  * [8. 从计算图到芯片](docs/chapters/36/section-09.md)
  * [9. 验收与当前交付边界](docs/chapters/36/section-10.md)
  * [10. 李亚普洛夫修订记录](docs/chapters/36/section-11.md)
  * [11. 隐式求解与持续演化的显式接入（2026-09-12）](docs/chapters/36/section-12.md)
  * [12. 能力优先与可选兼容扩展（2026-09-12 修订）](docs/chapters/36/section-13.md)
  * [13. 函数/算子空间与数值误差接口（2026-09-12）](docs/chapters/36/section-14.md)
  * [反思、回顾与复盘补充（2026-09-12）](docs/chapters/36/section-15.md)
  * [14. 统一数学建模：任务、模型与适应过程](docs/chapters/36/section-16.md)
  * [15. A 组详细构建：把观测变成可解释的输入](docs/chapters/36/section-17.md)
  * [16. B 组详细构建：模态编码与可测量状态的连接](docs/chapters/36/section-18.md)
  * [17. C 组详细构建：图、几何与表示层级](docs/chapters/36/section-19.md)
  * [18. D 组详细构建：时空块的可训练内核](docs/chapters/36/section-20.md)
  * [19. E 组详细构建：预测、生成与任务读出](docs/chapters/36/section-21.md)
  * [20. F 组详细构建：误差驱动的有界适应](docs/chapters/36/section-22.md)
  * [21. G 组详细构建：训练、复盘和持续学习](docs/chapters/36/section-23.md)
  * [22. H 组详细构建：工具、执行与证据](docs/chapters/36/section-24.md)
  * [23. L 组详细构建：稳定性、耗散与修正](docs/chapters/36/section-25.md)
  * [24. 内部扩展的数学职责与接入位置](docs/chapters/36/section-26.md)
  * [25. 完整模型如何构建和执行](docs/chapters/36/section-27.md)
  * [26. 可以论证的性质、条件和反例](docs/chapters/36/section-28.md)
  * [27. 复杂度、资源与实验构建](docs/chapters/36/section-29.md)
* [13 · 层与算子详解](docs/13.md)
  * [1. 核心结论](docs/chapters/13/section-01.md)
  * [2. SGD-Net 的架构边界](docs/chapters/13/section-02.md)
  * [3. 整体分层架构](docs/chapters/13/section-03.md)
  * [4. 统一符号与张量定义](docs/chapters/13/section-04.md)
  * [5. Layer 0：输入编码层](docs/chapters/13/section-05.md)
  * [6. Layer 1：图构造层](docs/chapters/13/section-06.md)
  * [7. Layer 2：节点与边嵌入层](docs/chapters/13/section-07.md)
  * [8. Layer 3：SSM 状态演化层](docs/chapters/13/section-08.md)
  * [9. Layer 4：GNN 拓扑传播层](docs/chapters/13/section-09.md)
  * [9b. Layer 4b：等变几何传播层](docs/chapters/13/section-10.md)
  * [9c. Layer 4c：多度量/流形传播层（远期可选）](docs/chapters/13/section-11.md)
  * [10. Layer 5：动态树路由层](docs/chapters/13/section-12.md)
  * [11. Layer 6：后验误差估计层](docs/chapters/13/section-13.md)
  * [12. Layer 7：拓扑自适应重构层](docs/chapters/13/section-14.md)
  * [13. Layer 8：稳定性投影层](docs/chapters/13/section-15.md)
  * [14. Layer 9：任务输出头](docs/chapters/13/section-16.md)
  * [15. SGD-Net Block：可堆叠基本单元](docs/chapters/13/section-17.md)
  * [16. 完整前向流程](docs/chapters/13/section-18.md)
  * [17. 总损失函数设计](docs/chapters/13/section-19.md)
  * [18. 算子与 Layer 的关系](docs/chapters/13/section-20.md)
  * [19. 最小可实现版本 MVP](docs/chapters/13/section-21.md)
  * [20. 标准版 SGD-Net](docs/chapters/13/section-22.md)
  * [21. 完整版 SGD-Net](docs/chapters/13/section-23.md)
  * [22. 与其他架构的关系](docs/chapters/13/section-24.md)
  * [23. 实现建议目录](docs/chapters/13/section-25.md)
  * [24. 核心类接口草案](docs/chapters/13/section-26.md)
  * [25. 复杂度分析](docs/chapters/13/section-27.md)
  * [26. 设计不变量](docs/chapters/13/section-28.md)
  * [27. 推荐开发顺序](docs/chapters/13/section-29.md)
  * [29. 总结](docs/chapters/13/section-30.md)
* [24 · 等变 GNN 与状态模型](docs/24.md)
* [18 · JEPA 与世界模型桥接](docs/18.md)
* [33 · 拓扑接触世界模型提案](docs/33.md)

## 数学、学习与持续演化

* [44 · 李亚普洛夫稳定性](docs/44.md)
* [45 · 隐式求解](docs/45.md)
* [46 · 动态生长与持续演化](docs/46.md)
* [48 · 函数空间与数值误差](docs/48.md)
* [27 · 混合求解与训练](docs/27.md)
* [14 · 双闭环控制与认知类比](docs/14.md)
* [17 · 复盘与自进化设计](docs/17.md)
* [50 · 反思、回顾与复盘](docs/50.md)
* [25 · 物理约束与执行检查](docs/25.md)

## 领域模型与科学应用

* [35 · 跨领域扩展路线](docs/35.md)
* [37 · 世界模型](docs/37.md)
* [38 · 物理 AI 与灵巧手](docs/38.md)
* [39 · 制药与分子](docs/39.md)
* [40 · 生物与细胞](docs/40.md)
* [41 · 材料](docs/41.md)
* [42 · 神经行为](docs/42.md)
* [43 · 信号与突触](docs/43.md)
* [07 · 科学大模型扩展](docs/07.md)
* [08 · 科研 Agent 与材料筛选](docs/08.md)
* [15 · 具身实验与数字孪生](docs/15.md)

## 实现与实验验证

* [03 · 实现技术白皮书](docs/03.md)
* [04 · 实现概要设计](docs/04.md)
* [05 · 实现详细设计](docs/05.md)
* [34 · 跨本体验证与成本](docs/34.md)
* [49 · 任务卡与评估路线](docs/49.md)

## 附录

* [组件登记索引](architecture/README.md)
* [公式与图示检查](guide/rendering-check.md)
* [许可说明](NOTICE.md)
* [接入与维护](GITBOOK_SETUP.md)

* [在线阅读与发布维护](guide/publishing.md)

* [双语文档结构与维护](guide/bilingual.md)
