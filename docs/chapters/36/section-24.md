# 22. H 组详细构建：工具、执行与证据

H 组的公式描述协议不变量和资源约束，并非为外部系统强造可训练神经层。

**本页目录**

- [H01 领域工具适配](#section-001)
- [H02 ELN/LIMS/数据检索](#section-002)
- [H03 硬预算管理](#section-003)
- [H04 领域输出检查](#section-004)
- [H05 提交与回退](#section-005)
- [H06 设备与实验编排](#section-006)
- [H07 编译与数值配置](#section-007)
- [H08 审计与监控](#section-008)

---

## H01 领域工具适配 <a href="#section-001" id="section-001"></a>

**作用与数学。** 外部求解调用返回结构化结果：

$$
Tool(q;version,budget)\to(value,fidelity,error,status,provenance).
$$

**构建与使用。** 为 DFT、MD、仿真或外部模型分别定义输入单位、边界、随机性与失败原因；返回近似解不自动成为真值。不可微工具可用于离线标签/候选评价，不默认参与梯度；代理模型若替代工具须有独立误差验证。工具超时、网络等待与数据搬运计入总任务成本。

## H02 ELN/LIMS/数据检索 <a href="#section-002" id="section-002"></a>

**作用与数学。** 检索是在版本与权限允许的集合上选择记录：

$$
\mathcal R_q=\operatorname{TopK}_{r\in D_{allowed}}score(q,r),\quad
result=(records,versions,sources).
$$

**构建与使用。** 支持精确 ID 查找和语义检索，但二者保证不同；近似向量命中不能替代任意地址查询。模型状态保存摘要，原始证据仍由记录系统提供。检索结果是上下文候选，不自动是事实或可执行指令。无参数也能实现精确检索，学习排序器另登记数据与评价。

## H03 硬预算管理 <a href="#section-003" id="section-003"></a>

**作用与数学。** 准入需满足保守资源约束，例如

$$
T_{remain}\geq T_{candidate}^{bound}+T_{check}^{bound}+T_{fallback}^{bound},
\quad M_{active}+M_{candidate}+M_{reserve}\leq M_{limit}.
$$

**构建与使用。** 没有可靠最坏时延上界时不能声称硬实时，只能用时限中止与隔离提供规定退路。计量覆盖队列、搬运、不可抢占段和日志争用；F03 的收益预测不能修改预算事实。训练/复盘在后台资源域执行，超预算暂停候选，不影响活动版本及最低处置。

## H04 领域输出检查 <a href="#section-004" id="section-004"></a>

**作用与数学。** 对各项命题产生三值结果：

$$
check_j(cand)\in\{pass,violation,unknown\}.
$$

**构建与使用。** 检查单位、形状、几何、有效期、约束及数值证据；仅当所有必需命题满足时允许相应用途。缺证据返回 unknown，不能将“未发现问题”自动当证明。静态研究报告可保留未知供分析，设备动作是否允许由具体合同决定；H04 不把数值通过升级为科学因果结论。

## H05 提交与回退 <a href="#section-005" id="section-005"></a>

**作用与数学。** 将决定绑定精确候选版本：

$$
Commit=(requestID,candidateID,digest,epoch,decision),\qquad
Committed\not\Rightarrow Executed.
$$

**构建与使用。** 提交、设备接受与执行反馈分开，同请求身份的重试按协议处理；通信不确定不能盲目重复物理动作。修改候选后需重新检查。报告输出与设备动作使用不同适用合同；回退可换计算候选或采取预定动作，不能撤销已发生的实验效应。

## H06 设备与实验编排 <a href="#section-006" id="section-006"></a>

**作用与数学。** 一个协议状态机可表示为

$$
s_{dev}^{+}=\delta(s_{dev},command,feedback),\qquad command\in Allowed(s_{dev},protocol).
$$

**构建与使用。** 设备 adapter 负责原生单位、ACK 意义、互锁与结果归属，保留样本/会话/协议版本。模型可提供候选，但实际执行须满足既定授权与状态条件。生物效应可能延迟且有噪声，不能用设备 ACK 当实验成功标签。该单元属于系统编排，不是新增推理能力。

## H07 编译与数值配置 <a href="#section-007" id="section-007"></a>

**作用与数学。** 将逻辑图降为执行计划：

$$
Plan=Compile(\mathcal A,shapes,dtypes,layout,capabilities),\quad
\|F_{Plan}(x)-F_{ref}(x)\|\leq\epsilon_{profile}
$$

（后式是需要验证的目标，不是编译自动保证）。

**构建与使用。** 显式列出 gather/scatter、GEMM、归约、scan、自动微分及求解器原语。动态结构可用形状桶、预编译库或后台编译；不支持时报告设计缺口。Aegis-X 功能范围按 SGD-Net 需求修订，但不能伪造实际 capability。降精度必须按任务/导数误差验证，单看权重误差不足。

## H08 审计与监控 <a href="#section-008" id="section-008"></a>

**作用与数学。** 证据事件可组织为带版本依赖链：

$$
e_k=(id_k,parent_k,time_k,objectVersions_k,decision_k,reason_k,source_k).
$$

**构建与使用。** 保存预测先于结果、候选训练、检查与发布的可追溯关系；可用摘要校验完整性，但哈希不证明科学真实性。溢出/丢失显式记录，关键执行不能被大日志写入无限阻塞。审计支撑复盘与模型回退，不替代运行时授权或任务评估。


---

[← 上一页](section-23.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-25.md)
