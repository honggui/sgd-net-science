# 20. F 组详细构建：误差驱动的有界适应

**本页目录**

- [F01 误差证据提取](#section-001)
- [F02 校准与域外检测](#section-002)
- [F03 收益与预算选择](#section-003)
- [F04 动态树路由与生长候选](#section-004)
- [F05 Prolong 状态展开](#section-005)
- [F06 Restrict 状态汇聚](#section-006)
- [F07 局部精化执行](#section-007)
- [F08 候选状态版本](#section-008)

---

## F01 误差证据提取 <a href="#section-001" id="section-001"></a>

**作用与数学。** 输出分项而非混成一个“智能误差”：

$$
e_{diag}=(e_{obs},e_{constraint},e_{solver},e_{calibration},e_{reference}),
\quad e_{obs}=\|W_{unit}M(y-\hat y^-)\|.
$$

**构建与使用。** 每项绑定单位、掩码、来源、样本数与经验/有界/未知标签。缺真值时不生成假定的真实误差；不可辨识的观测、表示与学习误差不能凭一条残差完全分开。F01 为 EV02 提供原因证据和 F03 的动作依据，不直接决定生长。与 FN04 共用数值可靠性合同。

## F02 校准与域外检测 <a href="#section-002" id="section-002"></a>

**作用与数学。** 分类校准实例为

$$
p_T(k\mid x)=\operatorname{softmax}(l(x)/T)_k,\quad
T^*=\arg\min_{T>0}\sum_{(x,y)\in D_{cal}}-\log p_T(y\mid x).
$$

**构建与使用。** T 在独立校准集拟合；域外分数可以来自特征距离或模型分歧，但需专门验证其检出率与误报率。改变主干或数据域后重校准。置信度不是错误概率的自动保证，低熵模型也可能错误；任何不确定性触发规则须在部署近似分布上评估。

## F03 收益与预算选择 <a href="#section-003" id="section-003"></a>

**作用与数学。** 在有限候选动作 b 中选择

$$
b^*=\arg\max_{b\in\mathcal B_{allowed}}\left[\widehat{\Delta Q}(e,b)-\lambda_CCost(b)\right],
\quad Cost(b)+Reserve\leq Budget.
$$

**构建与使用。** 候选包括不变、更多迭代、补观测、局部细化和训练，不只生长。先以规则策略运行并记录真实收益，再训练收益预测器；其预测不得充当硬预算证明，H03 强制计量。若收益未知或均为负，允许不做动作。奖励/成本单位及 λ 来源须登记。

## F04 动态树路由与生长候选 <a href="#section-004" id="section-004"></a>

**作用与数学。** 对专家树的一种软路径为

$$
p_{left}=\sigma(g_\theta(s)),\quad
\pi_\ell(s)=\prod_{v\in path(\ell)}p_v^{b_v}(1-p_v)^{1-b_v},
\quad \hat y=\sum_\ell\pi_\ell f_\ell(s).
$$

**构建与使用。** s 在几何配置中采用不变量；硬路由用阈值时需处理不可微与边界跳变。representation 树细化图并调用 P/Q，expert 树改变局部函数，candidate 树管理搜索副本，三者不能互换。持久分裂通过 EV03—EV07 初始化、训练和竞争，树越大不自动更好；扩展上限和修剪规则必须存在。

## F05 Prolong 状态展开 <a href="#section-005" id="section-005"></a>

**作用与数学。** 对粗细计算表示，参考线性映射为

$$
z_f=Pz_c,\quad h_f=Ph_c+\delta_\theta(h_c,geometry),\qquad P\mathbf1=\mathbf1.
$$

**构建与使用。** P 行和为一可保持常量，但不足以保证守恒。每层 z/h 都迁移，细节头可零初始化；向量映射的系数需不变，物理位置、质量与标签使用各自单位规则。通过共同物理读出和稳定范数训练/检查，不强制细状态全部可由粗状态无损恢复。真实实体数不因 P 改变。

## F06 Restrict 状态汇聚 <a href="#section-006" id="section-006"></a>

**作用与数学。** 在 W_f 正定、P 满列秩时，可采用加权最小二乘限制：

$$
Q=(P^TW_fP)^{-1}P^TW_f,\quad QP=I,\qquad z_c=Qz_f.
$$

**构建与使用。** 上式通过线性求解实现，避免直接求逆；若秩不足/病态应改映射、正则并重新登记性质。QP=I 只保持粗子空间，不意味着 PQ=I 或无损压缩任意细节。学习 Q 时用一致性与实际任务损失联合训练，且检验向量类型、权重和物理量规则；节点平均不是普遍守恒限制。

## F07 局部精化执行 <a href="#section-007" id="section-007"></a>

**作用与数学。** 对选中区域 Ω、冻结外部边界 b：

$$
s_\Omega^{k+1}=\Phi_\theta(s_\Omega^k,b,c),\quad k<K_{max},
\qquad\text{或 }R_\theta(s_\Omega;b,c)=0.
$$

**构建与使用。** 显式重复块与 IM 隐式求解分别配置；halo 深度应覆盖消息感受野，边界可为耦合未知量时不能随意固定。精化 k 不推进真实时间，版本、单位与停止规则不变。训练共享块时梯度经过真实执行路径；边界截断误差和求解失败进入 F01，而不是只报告局部残差下降。

## F08 候选状态版本 <a href="#section-008" id="section-008"></a>

**作用与数学。** 快照显式包含

$$
S_\nu=(h^{1:L},z^{1:L},x,IDs,M,\mathcal G,\nu_\theta,\nu_{cal},\nu_{layout},t),
\quad S^{candidate}=\operatorname{Fork}(S_\nu).
$$

**构建与使用。** 实现可复制或共享只读对象，但候选写入不能污染活动轨迹；保留 parent 和候选时钟。缓存与图版本失配时不能直接加载。无可训练参数，训练中的候选模型也需独立优化器状态。丢弃计算候选仅回收资源，不能撤销已发生的设备动作。


---

[← 上一页](section-21.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-23.md)
