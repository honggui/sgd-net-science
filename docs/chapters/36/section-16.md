# 14. 统一数学建模：任务、模型与适应过程

**本页目录**

- [14.1 模型是受条件约束的函数或算子族](#section-001)
- [14.2 统一符号与轴](#section-002)
- [14.3 三类持久对象与两个循环](#section-003)

---

## 14.1 模型是受条件约束的函数或算子族 <a href="#section-001" id="section-001"></a>

设任务条件为 c，可观测输入为 a，目标为 u。约束集合与解集合分别为

$$
\mathcal K(a,c)=\{u\in\mathcal Y:C(a,u,c)=0,\ G(a,u,c)\leq0\},\qquad
\mathcal S(a,c)=\{u\in\mathcal K(a,c):F(a,u,c)=0\}.
$$

也可用受约束风险最小化定义目标。解集合可能非线性、多解或为空；随机任务用条件分布表示，不能默认唯一真解。对场任务，观测算子 P_h 把函数采样为有限张量，重构 R_h 把网络输出恢复为场：

$$
\widehat{\mathcal S}_{\theta,\mathcal A}=R_h\circ E_\theta\circ
D_{\theta,\mathcal A}^{(L)}\circ\cdots\circ D_{\theta,\mathcal A}^{(1)}
\circ B_\theta\circ P_h.
$$

C 组构造各块使用的图，F 组按需调整计算，G 组更新参数，H 组负责执行，L 组提供适用的稳定性路径。R_h/P_h 不必是神经层。对于普通有限维回归可取恒等表示。隐式块则通过满足残差方程定义 D 的输出，不能直接解释为固定次数的显式复合。

## 14.2 统一符号与轴 <a href="#section-002" id="section-002"></a>

以下单元公式省略批量轴，节点为 i、邻居为 j、物理事件为 t、层为 ℓ、精化为 k、参数版本为 ν。N 为节点数，E 为边数，d 为标量宽，s 为时序宽，n 为欧氏坐标维，q 为向量通道数。h_i∈R^d，z_i∈R^s，x_i∈R^n，v_i∈R^{q×n}。物理动作记 a_t，任务函数输入在需要时记 a(·)，以避免混淆。Δt 与生成时间 τ、训练更新次数和求解迭代次数不是同一变量。

φ、ψ 表示带登记参数与激活的网络；⊙ 是逐元素乘法，concat 是拼接；M 是有效掩码；ε>0 是数值保护量，其物理单位应与被相加项一致。所有可训练映射均需登记输入/输出维数、共享方式、初始化及精度。下文中的概率只在显式定义分布后使用。

## 14.3 三类持久对象与两个循环 <a href="#section-003" id="section-003"></a>

参数 θ 包括编码器、消息函数、SSM、读出和选用的专家；运行记忆 m 包含各层 z、实体映射与候选状态；经验库 B 保存可追溯事件。三者不同：更新记忆不等于训练参数，写入经验不等于模型学会经验。

$$
(\hat y_t,m_t)=\mathcal F_{\theta_\nu,\mathcal A_\nu}(o_{\leq t},m_{t^-};c),
\qquad
(\theta_{\nu+1},\mathcal A_{\nu+1})=
\operatorname{Accept}\!\left(\operatorname{ProposeTrain}(\mathcal B,\theta_\nu,\mathcal A_\nu)\right).
$$

前式为事件级推理，后式为窗口级学习与版本接纳。Accept 可按预先定义的检验自动执行，也可拒绝并保持 ν；它不是可微层，更不是必然产生 ν+1。求解/反传期间冻结 ν，避免梯度对应一个不断变化的函数。


---

[← 上一页](section-15.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-17.md)
