# 21. G 组详细构建：训练、复盘和持续学习

**本页目录**

- [G01 损失装配](#section-001)
- [G02 数据采样与切分](#section-002)
- [G03 混合优化器](#section-003)
- [G04 JEPA 教师/预测器](#section-004)
- [G05 回放与复盘](#section-005)
- [G06 适配/蒸馏/合并](#section-006)
- [G07 独立评估](#section-007)
- [G08 主动实验选择](#section-008)

---

## G01 损失装配 <a href="#section-001" id="section-001"></a>

**作用与数学。** 组合按任务定义的无量纲损失：

$$
\mathcal L=\sum_{r\in\mathcal T_{enabled}}\lambda_r\frac{\sum_iM_{ir}\ell_r(\hat y_{ir},y_{ir})}{\max(1,\sum_iM_{ir})}
+\lambda_{reg}\Omega(\theta).
$$

**构建与使用。** r 可含似然、几何、力、JEPA、迁移与稳定正则，未启用目标不参与。尺度化与权重均登记，测试集不用于调 λ。把损失称为泛函是数学表达选择，不自动赋予它真实物理能量、凸性或唯一极小点；冲突目标需监控分项指标。

## G02 数据采样与切分 <a href="#section-002" id="section-002"></a>

**作用与数学。** 以来源组 g 划分集合：

$$
\mathcal G_{train}\cap\mathcal G_{test}=\varnothing,\qquad
D_{test}=\{x:g(x)\in\mathcal G_{test}\}.
$$

**构建与使用。** 分组可以是动物、供体、分子骨架、实验批次或时间窗口；增强视图及衍生记录继承原组。归一化在训练集拟合，校准集单列；不能把近重复实验随机打散后声称域外泛化。样本量不足的领域明确评估不确定性，不以增加随机切分次数创造独立样本。

## G03 混合优化器 <a href="#section-003" id="section-003"></a>

**作用与数学。** 对欧氏参数，以 AdamW 参考更新为例：

$$
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\quad
v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2,
$$
$$
\theta_{k+1}=(1-\eta\lambda)\theta_k-\eta\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon},
\quad \hat m_k=m_k/(1-\beta_1^k),\quad\hat v_k=v_k/(1-\beta_2^k).
$$

**构建与使用。** 梯度来源可为显式反传或 IM05；流形参数另用声明的 retraction，硬树结构在外环选择。新参数的优化器槽须初始化，旧槽按迁移映射处理。在线训练仅在候选域更新，不能边反传边改活动形状。优化器可运行于伴随 CPU/加速器，位置不改变数学责任。

## G04 JEPA 教师/预测器 <a href="#section-004" id="section-004"></a>

**作用与数学。** 以匹配视图目标为例：

$$
\mathcal L_J=\|p_\theta(f_\theta(x_{context}))-\operatorname{sg}(f_{\bar\theta}(x_{target}))\|^2,
\quad\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta.
$$

**构建与使用。** 上式仅是教师—预测器骨架，必须采用完整的视图、掩码、目标归一化与防塌缩策略并验证表征方差；EMA 或 stop-gradient 单独不保证非塌缩。目标支路不泄漏到在线输入，教师不接该项梯度。桥接到物理状态仍依 B08 的观测监督，不从潜损失小推出物理正确。

## G05 回放与复盘 <a href="#section-005" id="section-005"></a>

**作用与数学。** 经验记录应至少保留

$$
b_t=(o_{\leq t}^{available},\hat y_t^-,a_t,y_t,versions,source,cost),\quad
P(sample=b)\propto\epsilon+priority(b).
$$

**构建与使用。** priority 可考虑重复误差、代表性和稀缺性，不应只回放失败或高损失样本；若目标是原分布风险，非均匀采样需相应加权或明确改了训练分布。原始事件、原因假设和模拟反事实分层保存。复盘先检验错误来源，再提出修订；写出解释不作为正确归因证据。

## G06 适配/蒸馏/合并 <a href="#section-006" id="section-006"></a>

**作用与数学。** 在当前任务训练与旧任务保持间构造

$$
\mathcal L_{adapt}=\mathcal L_{new}(\theta')+
\lambda\mathbb E_{x\sim D_{replay}}D(p_{old}(\cdot|x),p_{new}(\cdot|x)).
$$

**构建与使用。** D 可为适用分布上的 KL 或结构一致性，旧模型只是行为参考，不能强迫复制已知错误。adapter 可低秩参数化 ΔW=UV^T，秩与初始化登记。拓扑不同的模型不能逐元素平均参数；先对齐或蒸馏到目标结构。持久接纳使用新数据和旧任务门槛，回放损失低不保证无遗忘。

## G07 独立评估 <a href="#section-007" id="section-007"></a>

**作用与数学。** 在独立测试单元上估计风险与资源：

$$
\hat R=\frac1{n}\sum_{i=1}^{n}\ell_i,\qquad
Report=(\hat R,CI,latency,memory,energy,failure).
$$

**构建与使用。** CI 应按真正独立单元或分组重采样计算，不能把同一动物的相邻时间点当独立重复。报告固定预算消融、域外切分和多随机种子；没有能量测量时不可伪造 J/task。持续多次用同一留出集选择模型会过拟合，需要后续数据或独立最终集。

## G08 主动实验选择 <a href="#section-008" id="section-008"></a>

**作用与数学。** 在已允许的实验/观测集合中，候选评分可写为

$$
a^*=\arg\max_{a\in\mathcal A_{allowed}}
\left[\mathbb E(\Delta Q\mid a,D)-\lambda C(a)\right].
$$

**构建与使用。** 增益可来自信息获取、终点改进或减少模型分歧，但其估计依赖当前模型，不能当已知真实效应。重复实验、阴性对照和多保真成本需进入方案。该单元提出建议，H06 根据已有实验授权执行；“主动”不意味着自行改变生物协议或设备限值。


---

[← 上一页](section-22.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-24.md)
