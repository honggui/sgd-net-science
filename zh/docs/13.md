# SGD-Net 模型架构与层算子详解

> 公开研究版 · 2026-09-12。保留模型方法、公式和组件设计；本文描述研究方案，未据此声称已有训练结果、完整实现或芯片性能。章节编号保留原研究索引，缺号表示未随包发布的独立规划内容。


> 版本：v1.2
> 日期：2026-06-28
> 定位：独立解释 SGD-Net 到底是整体 AI Model 架构，还是由多个子模块构成，并给出可实现的层级结构、张量形状、核心算子、前向流程，以及对 `00`～`31` 号文档中应用场景和算法路线的覆盖边界。


## 分章阅读 <a href="#section-001" id="section-001"></a>

按原稿章节拆分，公式与算法内容保留。可以按顺序阅读，也可直接选择关注的单元。

- [1. 核心结论](chapters/13/section-01.md)
- [2. SGD-Net 的架构边界](chapters/13/section-02.md)
- [3. 整体分层架构](chapters/13/section-03.md)
- [4. 统一符号与张量定义](chapters/13/section-04.md)
- [5. Layer 0：输入编码层](chapters/13/section-05.md)
- [6. Layer 1：图构造层](chapters/13/section-06.md)
- [7. Layer 2：节点与边嵌入层](chapters/13/section-07.md)
- [8. Layer 3：SSM 状态演化层](chapters/13/section-08.md)
- [9. Layer 4：GNN 拓扑传播层](chapters/13/section-09.md)
- [9b. Layer 4b：等变几何传播层](chapters/13/section-10.md)
- [9c. Layer 4c：多度量/流形传播层（远期可选）](chapters/13/section-11.md)
- [10. Layer 5：动态树路由层](chapters/13/section-12.md)
- [11. Layer 6：后验误差估计层](chapters/13/section-13.md)
- [12. Layer 7：拓扑自适应重构层](chapters/13/section-14.md)
- [13. Layer 8：稳定性投影层](chapters/13/section-15.md)
- [14. Layer 9：任务输出头](chapters/13/section-16.md)
- [15. SGD-Net Block：可堆叠基本单元](chapters/13/section-17.md)
- [16. 完整前向流程](chapters/13/section-18.md)
- [17. 总损失函数设计](chapters/13/section-19.md)
- [18. 算子与 Layer 的关系](chapters/13/section-20.md)
- [19. 最小可实现版本 MVP](chapters/13/section-21.md)
- [20. 标准版 SGD-Net](chapters/13/section-22.md)
- [21. 完整版 SGD-Net](chapters/13/section-23.md)
- [22. 与其他架构的关系](chapters/13/section-24.md)
- [23. 实现建议目录](chapters/13/section-25.md)
- [24. 核心类接口草案](chapters/13/section-26.md)
- [25. 复杂度分析](chapters/13/section-27.md)
- [26. 设计不变量](chapters/13/section-28.md)
- [27. 推荐开发顺序](chapters/13/section-29.md)
- [29. 总结](chapters/13/section-30.md)

---

[← 上一页](chapters/36/section-29.md) · [全书目录](../SUMMARY.md) · [下一页 →](chapters/13/section-01.md)
