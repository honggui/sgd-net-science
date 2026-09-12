# 18. D 组详细构建：时空块的可训练内核

**本页目录**

- [D01 类型保持归一化/投影](#section-001)
- [D02 消息生成](#section-002)
- [D03 消息归约](#section-003)
- [D04 实体时序 SSM](#section-004)
- [D05 观测创新校正](#section-005)
- [D06 节点残差 FFN](#section-006)
- [D07 EGNN 几何读出/更新](#section-007)
- [D08 度量/流形插槽](#section-008)

---

## D01 类型保持归一化/投影 <a href="#section-001" id="section-001"></a>

**作用与数学。** 标量可用 LayerNorm；向量使用不变范数归一：

$$
\tilde h=\operatorname{LN}(h),\quad
s_v=\sqrt{\epsilon+q^{-1}\sum_a\|v_a\|^2},\quad
\tilde v_a=\sum_bW_{ab}v_b/s_v.
$$

**构建与使用。** W 混合通道轴而不随意混合空间轴；向量偏置不能是固定方向。标量路径只允许真正不变量使用普通 LN。参数训练由下游任务驱动；增宽时 LN 的统计维度会变化，因此复制旧权重不保证函数不变，迁移后需重验。

## D02 消息生成 <a href="#section-002" id="section-002"></a>

**作用与数学。** 对具有关系类型 r 的边：

$$
m_{ij}=\phi_{m,r}([h_i,h_j,I_{ij},a_{ij},c]),\qquad m_{ij}\in\mathbb R^{d_m}.
$$

**构建与使用。** 参考 φ 为 Linear→SiLU→Linear，输入宽 2d+d_I+d_e+d_c；关系可用共享主干加类型嵌入，避免每种边复制大网络。几何模式只将不变量送入标量 MLP；非几何图依 schema 使用属性。局部消息训练不能自动修复一阶图的表达限制，需按 U05 的任务反例选择更高阶结构。

## D03 消息归约 <a href="#section-003" id="section-003"></a>

**作用与数学。** 默认节点聚合为

$$
\bar m_i=\frac{\sum_{j\in\mathcal N(i)}M_{ij}m_{ij}}{\max(1,\sum_jM_{ij})},
$$

或直接求和；场积分型配置改为 Σ_jw_jκ_θ(i,j)v_j，w_j 为求积权重。

**构建与使用。** mean、sum、积分权重含义不同，应在模型包登记，不能在部署时随意互换。空邻域给零消息，掩码禁止泄漏 padding。固定 sum/mean 不是 QK 注意力；采用注意力必须明确归一范围与掩码。训练梯度沿归约分发，硬件实现要测重复索引与浮点顺序差异。

## D04 实体时序 SSM <a href="#section-004" id="section-004"></a>

**作用与数学。** 本版参考实例以有界衰减维护每实体、每层状态：

$$
u_i=W_u[h_i,\bar m_i,c],\ a=-\operatorname{softplus}(\alpha),\quad
\delta_i=\Delta t_i\operatorname{softplus}(W_\delta u_i+b_\delta),\quad
\bar a_i=\exp(\delta_i\odot a),
$$
$$
z_i^-=\bar a_i\odot z_i^{old}+(1-\bar a_i)\odot\phi_b(u_i).
$$

**构建与使用。** u、δ、a、z 均为 s 维，层间独立、事件间共享参数；新实体状态按固定初始化处理。用因果序列训练，可截断 BPTT 但须登记截断长度。该指数平滑实例只约束齐次对角项，不是完整 Mamba，也不是隐式求根。Δt=0 不应推进记忆，空间精化不得重复执行物理衰减。

## D05 观测创新校正 <a href="#section-005" id="section-005"></a>

**作用与数学。** 对观测前已保存的同一目标预测形成

$$
r_t=M_t\odot(y_t-\hat y_t^-),\qquad
z_t=z_t^-+\gamma(u_t,M_t)\odot W_r\operatorname{Scale}(r_t).
$$

**构建与使用。** W_r:R^{d_y}→R^s，γ 是 s 维门，并乘以本次存在有效创新的指示量，使全缺测时严格为零；多个观测头分别按单位和来源组合。标量 h/z 路径只接收不变量创新，向量创新需用等变投影进入向量状态，不能经任意 W_r 混入不变通道。每层可消费同一真实创新一次，不得先用 y 更新预测再把差值称为先验误差。缺少匹配预测时使用独立编码路径或标记创新未知。K/门控由任务损失训练，不自动成为 Kalman 最优增益或贝叶斯后验。

## D06 节点残差 FFN <a href="#section-006" id="section-006"></a>

**作用与数学。** 将当前表征、空间消息与记忆合成：

$$
h_i^+=h_i+W_o\phi_h([h_i,\bar m_i,z_i,c]).
$$

**构建与使用。** 无时序配置去掉 z 输入并另设对应权重形状，不能填充任意历史。FFN 可用两层 MLP，输出宽 d；参考 dropout=0，启用时记录训练/推理差异。残差保留旧信息路径但不保证稳定或永不退化。若加入新的残差专家，可零初始化输出投影以降低切换突变，仍需全模型验证。

## D07 EGNN 几何读出/更新 <a href="#section-007" id="section-007"></a>

**作用与数学。** 用不变消息产生标量系数，更新计算坐标 ξ：

$$
\xi_i^+=\xi_i+\eta\sum_jM_{ij}\phi_x(m_{ij})(\xi_i-\xi_j).
$$

向量输出可类似采用 Σ_jφ_v(m_ij)r_ij，每个向量通道有独立标量系数。

**构建与使用。** n 是配置维度，φ_x 输出标量，η 为声明的尺度；实测 x 保持原始记录，ξ 是推理变量。该结构在图、消息与群作用相容时满足 E(n) 等变，见第 26 节推导。训练坐标/向量目标需要对应类型损失；任意坐标更新既不是物理积分器，也不保证守恒力。E(n) 本身已含高维先例。[EGNN](https://proceedings.mlr.press/v139/satorras21a.html)

## D08 度量/流形插槽 <a href="#section-008" id="section-008"></a>

**作用与数学。** 在已定义黎曼流形 (M,g) 上，一种受约束更新为

$$
\xi^+=\operatorname{Retr}_\xi(-\eta\operatorname{grad}_g\mathcal J),\qquad
\operatorname{grad}_g\mathcal J=G(\xi)^{-1}\nabla_\xi\mathcal J
$$

（后式限于相应局部坐标且 G 正定）。

**构建与使用。** 必须提供坐标域、切空间、度量、retraction、奇点与梯度接口；不能把任意特征距离代入即称流形算法。Lorentz 型不定度量不适用上述正定梯度保证，应另定义几何和约束。默认关闭，只有任务有证据需要时与欧氏基线比较。



---

[← 上一页](section-19.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-21.md)
