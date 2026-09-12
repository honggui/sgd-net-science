# 13. Layer 8：稳定性投影层

**本页目录**

- [13.1 功能](#section-001)
- [13.2 谱半径投影](#section-002)
- [13.3 奇异值裁剪](#section-003)
- [13.4 范数裁剪](#section-004)
- [13.5 Lyapunov 能量下降约束](#section-005)
- [13.6 工程接口](#section-006)

---

## 13.1 功能 <a href="#section-001" id="section-001"></a>

稳定性投影层实施指定数值约束，并检查所声明稳定性条件；它本身不证明动态图整体不发散。

它回答的问题是：

> 新增节点、更新权重、状态转移矩阵是否会破坏稳定性？如果会，如何投影回安全集合？

## 13.2 谱半径投影 <a href="#section-002" id="section-002"></a>

对方阵状态转移矩阵 $$A$$，且 $$0<\rho_{max}<1$$：

$$
A' = \frac{A}{\max(1,\rho(A)/\rho_{max})}
$$

该式只控制单矩阵谱半径，不等价于欧氏范数收缩，也不覆盖任意输入相关切换；若需要共同收缩条件，应检查完整更新映射在同一范数下的界。

## 13.3 奇异值裁剪 <a href="#section-003" id="section-003"></a>

$$
W=U\Sigma V^T
$$

$$
\Sigma'=\operatorname{clip}(\Sigma,0,\sigma_{max})
$$

$$
W'=U\Sigma'V^T
$$

## 13.4 范数裁剪 <a href="#section-004" id="section-004"></a>

对节点特征：

$$
x_i'=x_i\cdot\min\left(1,\frac{c}{\|x_i\|_2}\right)
$$

## 13.5 Lyapunov 能量下降约束 <a href="#section-005" id="section-005"></a>

定义候选 Lyapunov 函数（不是任意任务的真实物理能量）：

$$
V_t=\frac{1}{2}h_t^TPh_t+\Pi(X_t,\mathcal{G}_t)
$$

若：

$$
V_{t+1}>V_t+\delta
$$

则执行：

- 缩小更新步长；
- 回滚 topology refinement；
- 剪枝异常分支；
- 重新投影权重。

此处须声明 P 正定、附加项的下界、参考平衡/轨迹以及输入功或扰动供给项。正定不等于下降；固定 δ 反复容许上升不保证稳定。跨拓扑需共同 V 或受界的切换条件；对矢量通道还须检验 V 的群不变性。失败回滚仅指候选计算状态，不能撤销真实动作。详见 44 号。

## 13.6 工程接口 <a href="#section-006" id="section-006"></a>

```text
StabilityProjector.project(state, params=None) -> GraphState
```


---

[← 上一页](section-14.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-16.md)
