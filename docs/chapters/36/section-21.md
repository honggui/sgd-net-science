# 19. E 组详细构建：预测、生成与任务读出

**本页目录**

- [E01 状态转移头](#section-001)
- [E02 观测似然头](#section-002)
- [E03 接触/任务效果头](#section-003)
- [E04 本体动作头](#section-004)
- [E05 物理属性头](#section-005)
- [E06 生物/功能头](#section-006)
- [E07 条件生成头](#section-007)
- [E08 读出与池化](#section-008)

---

## E01 状态转移头 <a href="#section-001" id="section-001"></a>

**作用与数学。** 将隐状态与动作转换为条件未来分布，一个混合增量实例为

$$
p(x_{t+1}\mid h_t,a_t,c)=\sum_{k=1}^{K}\pi_k\mathcal N(x_{t+1};x_t+\mu_k,\Sigma_k),
\quad \pi=\operatorname{softmax}(l),\quad \Sigma_k=L_kL_k^T+\epsilon I.
$$

**构建与使用。** 独立输出模式权重、增量和合法协方差参数；确定性任务可只取均值头。动作和 Δt 必须作为条件或由已定义动力学承担，rollout 在隔离状态中递推。多步训练须登记 teacher forcing 与自由滚动比例；单步准确不能保证长时准确。旋转等变场景的均值和协方差应按相应表示构造，普通对角协方差在固定轴上不具任意旋转一致性。

## E02 观测似然头 <a href="#section-002" id="section-002"></a>

**作用与数学。** 将隐状态映到实际测量空间，连续观测可写为

$$
p(y\mid h)=\mathcal N(y;\mu_\theta(h),\operatorname{diag}(\sigma_\theta(h)^2)),\quad
\sigma=\operatorname{softplus}(s)+\epsilon.
$$

**构建与使用。** 按传感类型选择 Gaussian、计数或事件分布，通过负对数似然训练，并结合 A05 处理缺失。对角 Gaussian 是特定观测假设，不能代表所有不确定性；几何向量的分布需遵循群作用。为 D05 提供观测前预测快照，后到标签不得反向污染同一预测。

## E03 接触/任务效果头 <a href="#section-003" id="section-003"></a>

**作用与数学。** 对边接触和向量效果分别构造

$$
p_{ij}=\sigma(w^Tm_{ij}+b),\qquad
\hat d_i=\sum_j\phi_d(m_{ij})r_{ij}.
$$

**构建与使用。** 真实接触标签与预测边置信分开；类别失衡可采用登记的加权交叉熵，效果向量用尺度化回归。无接触标签时只能报告几何接近等代理目标。该头连接空间表示与任务条件，不能将高接触概率等同稳抓取或物理可行；摩擦、力矩及执行延迟需另验。

## E04 本体动作头 <a href="#section-004" id="section-004"></a>

**作用与数学。** 对有限上下界的标量执行器分量，可令

$$
a_{t:t+H_a-1}=M_a\odot\left[a_{mid}+a_{range}\odot\tanh g_a(h,c_{morph},goal)\right].
$$

**构建与使用。** 输出固定时间跨度动作块，按本体映射至原生关节/执行器；H05 仅提交允许前缀。tanh 仅提供箱约束，不保证速度、碰撞或接触可行。旋转动作需合法表示，不能逐维夹紧四元数当作姿态约束。训练可用模仿或明确 RL 目标，两者标签、探索与评价不能混称；未经检查的输出只为候选。

## E05 物理属性头 <a href="#section-005" id="section-005"></a>

**作用与数学。** 一个能量一致的分子/材料配置为

$$
\hat E=\sum_i e_\theta(h_i),\qquad \hat F_i=-\nabla_{x_i}\hat E,
\quad \hat\sigma=\frac1{\Omega}\left.\frac{\partial\hat E((I+\varepsilon)H)}{\partial\varepsilon}\right|_{\varepsilon=0}.
$$

**构建与使用。** 应力符号、应变约定及原子随晶胞变形方式必须明确；公式为一种约定，不能跨数据源直接比较。力监督涉及能量对坐标导数，训练参数时需要混合二阶导数支持。距离截断和邻居变化要平滑且验证。结构能量拟合好不保证力好；分别报告能量、力、应力和 rollout，非保守系统另设适当头。

## E06 生物/功能头 <a href="#section-006" id="section-006"></a>

**作用与数学。** 对计数输出可选负二项分布：

$$
\mu_{ig}=L_i\operatorname{softmax}_g(f_\theta(h_i,c)),\quad
Y_{ig}\sim NB(\mu_{ig},r_g),\quad
\operatorname{Var}(Y_{ig})=\mu_{ig}+\mu_{ig}^2/r_g.
$$

**构建与使用。** r_g>0 可用 softplus 参数化，L_i 为 exposure 或已知库大小；该形式假设所建模基因集合上的组成均值，若目标不适合固定总量则另配均值模型。表型/生存/终点使用独立似然与标签时域，不能把不同终点共享一个无单位 MSE。条件关联预测不等于干预效应或药效。

## E07 条件生成头 <a href="#section-007" id="section-007"></a>

**作用与数学。** 以条件流匹配为可选实例：

$$
x_\tau=(1-\tau)x_0+\tau x_1,\quad
\mathcal L_{FM}=\mathbb E\|v_\theta(x_\tau,\tau,c)-(x_1-x_0)\|^2,
\quad \frac{dx}{d\tau}=v_\theta(x,\tau,c).
$$

**构建与使用。** x_0 是噪声、x_1 是数据，耦合与噪声分布需声明；τ 为生成时钟，与物理 t 无关。推理积分得到候选，由领域检查过滤；此线性路径不一定在流形或约束域内，受约束生成需重新定义路径。离散类别不能直接沿此实数 ODE 生成，需明确离散模型；扩散模式另配噪声调度，不将一个噪声头称为完整算法。

## E08 读出与池化 <a href="#section-008" id="section-008"></a>

**作用与数学。** 选择与任务量一致的读出：

$$
y_{ext}=\sum_iw_i\phi(h_i),\quad
y_{int}=\frac{\sum_iw_i\phi(h_i)}{\sum_iw_i},\quad
\hat u(a,x_q)=g_\theta\!\left(x_q,\sum_iw_i\kappa_\theta(x_q,x_i)h_i\right).
$$

**构建与使用。** 前两式适用于相应广延/强度量，分母须正；第三式是查询位置场读出的候选，权重与离散测度匹配。普通摘要 attention 不保证质量守恒或跨网格一致。按标签或查询点监督训练，输出尺度由 B08/A02 恢复；空图需明确定义缺失而非除零。


---

[← 上一页](section-20.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-22.md)
