# `gravityCU` 技术路线与开发排期文档

> **项目更名完成**：本项目已由 **`grav-gpu`** 全面更名为 **`gravityCU`**（CU = CUDA + Cosmology + CUHK），GitHub 仓库（[`HainaHuang729/gravityCU`](https://github.com/HainaHuang729/gravityCU)）、Pages 展示页（[`hainahuang729.github.io/gravityCU/`](https://hainahuang729.github.io/gravityCU/)）、论文手稿、基准图表与本地工作目录已全部统一。

---

## 1. 项目定位与路线原则

`gravityCU` 是一个**全 GPU 显存常驻（100% VRAM-resident）**的 CUDA TreePM + FMM 宇宙学引力求解器与全 GPU FoF / Catalog / SO 暗晕分析引擎，核心特性包括：

- **零主机粒子换入换出**：在 `1–4` 张 GPU 上将 `64^3` 到 `1024^3`（`1,073,741,824` 粒子）全部物理状态、Lazy Hilbert 八叉树、分布式 cuFFT PM 缓冲及在线 FoF 全部常驻显存。
- **高能效与高吞吐**：在同节点（`a9`：`2× EPYC 9754 + RTX A6000`）受控 `256^3` 跨代码对比中，达到 **`49.62 M` 活跃粒子更新/秒**，较 256 核 CPU `SWIFT` 实现 **`28.9×` 总耗时加速 / `41.9×` 纯演化加速**，整机能耗降低 **`22.0–25.8×`**（`5.27–6.19 Wh` vs. `136.16 Wh`），实测 GPU 利用率达 `42.7%–73.6%`。
- **端到端科学等价性**：与 `SWIFT` / `GADGET-4` / `PKDGRAV3` 完成多红移功率谱（`1024^3` $z=0$ 处 $P_\delta(k)$ 偏差 `0.00453%`）、暗晕质量函数与独立解析守恒门禁（Zel'dovich、周期性 Ewald $\theta$-收敛、Plummer 维里平衡、Layzer–Irvine `0.0307%`）验证。

### 战略优先级：先「精度 + 物理广度」，延后「超算级万卡基础设施」

结合实验室单机/小集群 GPU 算力（`1–4× RTX 4090 / A6000`）与天体物理科研产出效率，现行路线明确如下优先级：

1. **优先推进 — 主线一：数值精度跃迁（P1–P3）**：将当前对标 `SWIFT` 2 阶精度水平（Ewald 全力中位误差 `1.03e-3` @ $\theta=0.3$）进一步拉升至超高精度宇宙学基准线（`< 1e-4` 乃至 `~1–3e-5`，看齐 `Abacus` 与高阶 FMM），并在工作站级 `1–2×` 卡即可完成全部精度回归门禁。
2. **优先推进 — 主线二：物理广度扩展（B1–B4）**：将求解器从纯冷暗物质（CDM）拓展为支持**有质量中微子（Massive Neutrinos）**、**自相互作用暗物质（SIDM）与超越 $\Lambda$CDM 修改引力**、**多质量 Zoom-in 极高分辨率暗晕**，以及**全 GPU 流体力学（SPH/MFM + 辐射冷却）**的多物理平台。
3. **远期延后 — 远期路线：超算级 / 万卡集群基础设施（S1–S4）**：固定 `N_TOP=12` 顶层网格、2D Pencil FFT（`cuFFTMp`）、稀疏邻域通信图、LET 树剪枝与百万亿粒子并行 HDF5/ADIOS2 等万卡工程重构，延后至确有大规模超算机时与 `2048^3–8192^3` 全盒巡天需求时启动。

---

## 2. 当前已验证基线与 Phase 2 MPI 进度

### 2.1 物理与跨代码基线（已完成）

- **$256^3$ 与 $1024^3$ 单节点多卡演化**：`reports/opt12-to-opt5-bench.json`、`reports/code-energy-comparison.json`（已收录于 `paper/main.pdf` 与交互展示页）。
- **独立解析与守恒验证门禁**（[reports/analytical-physics-gates.json](reports/analytical-physics-gates.json)）：
  - **1D Zel'dovich 平面波塌缩**（$64^3$，$a=0.02\to0.50$）：$\Delta x_{\rm RMS}/L = 4.41\times 10^{-4}$，$\Delta v_{\rm RMS}/v_{\rm pk} = 0.592\%$。
  - **精确周期性 Ewald 力收敛**（$4{,}096$ 粒子，$\theta \in [0.9, 0.3]$）：短程中位相对误差从 $8.48\times 10^{-3}$ 单调收敛至 $1.14\times 10^{-4}$；总力中位误差受 $128^3$ CIC PM 网格下限限制，在 $\theta=0.3$ 处为 $1.03\times 10^{-3}$。
  - **32K 孤立 Plummer 平衡球**：$t=0\to4.0$ 总能量漂移 $|\Delta E/E_0| = 1.15\times 10^{-5}$，动量漂移 $6.38\times 10^{-6}$，维里比 $2T/|U| \in [0.9956, 0.9988]$。
  - **$256^3$ 全宇宙史 ($z=200\to0$)**：`9,257` 步、`526.65 s`（`4× A6000`），Layzer–Irvine 残差 `3.07e-4`，动量漂移 `3.72e-5`，$z=0$ 功率谱偏差 `0.123%`，大质量暗晕 `89/89`（$N\ge 10^4$）与 `18/18`（$N\ge 10^5$）全匹配。

### 2.2 Phase 2 多节点 MPI 扩展现状（[reports/mpi-multinode-bench.json](reports/mpi-multinode-bench.json)）

| 里程碑 | 内容与目标 | 状态 | 已验证指标 / 备注 |
| --- | --- | --- | --- |
| **Milestone 2A** | MPI 拓扑规划与 `MultiNodeCommStats`（`src/grav_mpi_comm.h`, `src/grav_domain_3d.h`） | **已通过** | `scripts/test_domain3d_smoke.cpp` 本地与集群均输出 `DOMAIN3D_SMOKE_PASSED`；连续 Hilbert rank 分配显著降低跨节点 ghost 网格与粒子数。 |
| **Milestone 2B** | 分布式 1D Slab `cuFFT` + `MPI_Alltoallv` PM（`src/grav_cuda_pm_mpi.cu/h`, `scripts/test_mpi_pm.cu`） | **部分通过（`np=2`）** | `phy-gpus` `b4`（RTX 4090, OpenMPI 4.1.6 + CUDA 12.6），`np=2, side=128, N=64^3`：`max_rel_field=1.012e-14`，`max_rel_pot=3.177e-16`，`pm_comm_ms=90.875`，输出 `MPI_PM_PASSED`；`1N×4R / 2N×1G / 2N×2G` 待空闲卡补跑。 |
| **Milestone 2C** | 3D 域分解 MPI 运行时引擎（`src/grav_domain_3d_mpi.cuh`, `src/multi_gpu_slab.cu`） | **编译通过** | `test_mpi_pm`、`grav_mpi`、`multi_gpu_slab` 在 `phy-gpus`（`sm_89`）全部零错误编出，包含 lazy ghost 基址修正、per-rank `compute_nvel` 与 `ns==0` 集合通信安全处理。 |
| **Gate M1 / M2 / M3** | `np=4` / 跨节点 PM 收尾、`64^3` 4-rank/2N 等价性门禁、`256^3` 多节点扩展性测试（`slurm/mpi-multinode-test.sbatch`） | **待集群空闲卡** | `phy-gpus` 全部 51 张在线 GPU（`a` 15/15、`b` 32/32、`c` 4/4）均处于满占状态，待释放后立即执行。 |

---

## 3. 近期与中期开发计划（第 1–18 周）

> **第 1 周收尾**：一旦 `phy-gpus` 集群释放空闲 GPU，立即提交 [slurm/mpi-multinode-test.sbatch](slurm/mpi-multinode-test.sbatch) 完成 **Gate M1**（`1N×4R`, `2N×1G`, `2N×2G`）、**Gate M2**（$64^3$ 4-rank / 2-node vs 1G 力/位移/能量一致性）与 **Gate M3**（$256^3$ 多节点扩展曲线）。

### 主线一：数值精度跃迁（Track 1: P1–P3）

| 阶段 | 排期 | 目标硬件 | 核心改造内容与对应模块 | 定量验收门禁（Gate） |
| --- | --- | --- | --- | --- |
| **P1：高阶 PM 网格与去混叠** | 第 2 周（**已完成 Gate P1**） | `1× RTX 4090` | 已在 [src/grav_cuda_pm.cu](src/grav_cuda_pm.cu)、[src/grav_cuda_pm_dist.cu](src/grav_cuda_pm_dist.cu) 与 [src/grav_cuda_pm_mpi.cu](src/grav_cuda_pm_mpi.cu) 实现 `CIC/TSC/PCS`（`GRAV_PM_ORDER=2|3|4`）、半网格对称 `Interlacing`（`GRAV_PM_INTERLACE=1`）、精确 $-i\mathbf{k}$ 谱梯度（`GRAV_PM_GRAD=ik`）及 `GRAV_PM_ASMTH`；同步完成 `monofonIC`/`MUSIC` 统一 IC 适配层。 | **Gate P1 已通过**（[reports/p1-pm-precision-gate.json](reports/p1-pm-precision-gate.json)）：12 组单测零自力 $\le 1.43\times 10^{-14}$、牛顿第三定律 `0.0`；`Ewald` 全力中位误差（$\theta=0.3$）由 `1.034e-3` 压降至 **`5.58e-5`**（`PCS+Interlace+ik`，**18.5× 提升**）与 **`6.68e-5`**（`TSC+Interlace+ik`，**15.5× 提升**）。 |
| **P2：高阶多极展开 + 高精度模式** | 第 5–6 周 | `1–2× RTX 4090` | 在 [src/grouped_force.cuh](src/grouped_force.cuh) 与 `src/m2p/` 引入高阶（5 阶多极 / FMM `L2P`）核函数，并新增 `GRAV_PRECISION_MODE=high` 编译/运行模式（FP64 力/多极累加与可选高阶树）。 | **Gate P2**：`Ewald` 全力中位误差达到 **`~1–3e-5`**（看齐 `Abacus` / 高阶 `PKDGRAV3` 梯队）；`Zel'dovich` 位移 RMS `< 0.1%`；$256^3$ `Layzer–Irvine` 残差 **`< 1e-4`**。 |
| **P3：自适应时间步与辛守恒精细化** | 第 7 周 | `1× RTX 4090` | 在时间步进器中增加潮汐张量 / 动力学时间步判据，优化多层级跨 rung 对称同步规则，降低高密度晕心积分相位漂移。 | **Gate P3**：孤立 Plummer 球能量漂移 **`< 1e-6`**；大暗晕内区密度剖面（$r < 3\epsilon$）与超细固定步长参考解偏差 **`< 1%`**。 |

### 主线二：物理广度扩展（Track 2: B1–B4）

| 阶段 | 排期 | 目标硬件 | 核心改造内容与对应模块 | 定量验收门禁（Gate） |
| --- | --- | --- | --- | --- |
| **B1：有质量中微子（Massive Neutrinos）** | 第 3–4 周 | `1–2× RTX 4090` | 实现 **$k$-空间线性响应中微子（$\delta_\nu(k)$）**：每步 PM FFT 在傅里叶空间直接按线性传递函数叠加中微子密度响应，零额外显存开销；并支持可选离散中微子粒子物种接口。 | **Gate B1**：全物质功率谱 $P_m(k)$ 相对阻尼与 `CLASS` / `CAMB` 线性预言及 `GADGET-4` / `SWIFT` 基准吻合至 **`< 0.5%`**。 |
| **B2：超越 $\Lambda$CDM 暗区物理（SIDM + 修改引力）** | 第 8–9 周 | `1–2× RTX 4090` | 在已有 $w_0, w_a$ 暗能量基础上：(1) 直接利用常驻显存八叉树叶节点（`packed_sources`）实现 **GPU 原生蒙特卡洛 SIDM 自相互作用散射**；(2) 增加 GPU 多重网格 **$f(R)$ 修改引力** 标量场求解器。 | **Gate B2**：精确复现孤立 Hernquist / NFW 暗晕 SIDM 核形成（core formation）与晚期核塌缩（core collapse）解析/基准曲线。 |
| **B3：多质量粒子 Zoom-in 极高分辨率** | 第 10 周 | `1–4× RTX 4090` | 发挥代码原生具备的非均匀 `d_mass` / `d_soft` / `GRAV_SOFT_IN` 与非均匀 GPU SO 分段排序能力，补齐树构建/PM 多质量网格边界与多级加密初始条件管线。 | **Gate B3**：单卡/四卡运行局部等效 $4096^3$–$8192^3$ 嵌套暗晕，$R_{200}$ 内**零低分辨率粒子污染**，次暗晕质量函数稳定收敛。 |
| **B4：全 GPU 流体力学（SPH / MFM + 辐射冷却）** | 第 11–18 周 | `1–4× RTX 4090` | 为 `PartType0` 构建全 GPU 流水线：(1) 阶段一（第 11–14 周）：GPU 紧邻搜索、平滑长度 $h_i$ 迭代、密度步与无粘/有粘理想气体流体双树遍历；(2) 阶段二（第 15–18 周）：查表式 GPU 辐射冷却与简化恒星形成/反馈。 | **Gate B4**：通过标准流体力学检验集：**Sod 激波管**、**Sedov–Taylor 爆炸波**、**Kelvin–Helmholtz 不稳定性**，以及 **Santa Barbara 绝热星系团** 基准。 |

---

## 4. 远期路线：超算级 / 万卡集群基础设施（Track 3: S1–S4，延后）

当前代码架构面向 **单节点多卡至小规模多节点（`1–16` GPU）** 最优。未来若需向百卡至万卡级（`10^3–10^4` GPUs / $8192^3+$ 粒子）超算扩展，需按以下四个模块重构，目前已列入远期储备：

1. **S1 动态顶层网格**：替换当前静态 `N_TOP=12`（固定 `1,728` 个顶层网格），实现 $N_{\rm top} \propto G^{1/3}$ 或纯树层级 ORB/Hilbert 动态切分，消除超多卡场景下的顶层域粒度瓶颈。
2. **S2 2D Pencil FFT 与 64 位集合通信**：将 1D Slab 分布 FFT（受限 `side >= 2*G`）替换为基于 `cuFFTMp` 或 `heFFTe` 的 2D Pencil 分解，并将 MPI `int` 偏移全面升级为 `MPI_Alltoallv_c` / 64 位计数。
3. **S3 稀疏邻域通信与极限显存压缩**：以 `MPI_Dist_graph_create_adjacent` 替代全连接状态收集，引入 Locally Essential Tree（LET）远程树剪枝与节点内 NVLink + 节点间 GPUDirect RDMA 分层聚合；压缩显存结构（`~200 B/粒子` $\to$ `~80–100 B/粒子`）。
4. **S4 64 位粒子 ID 与并行异步 I/O**：将粒子索引扩展至 `uint64_t`（突破 $2^{32} \approx 4.29\times 10^9$ 上限），并用分片 MPI-IO / Parallel HDF5 / ADIOS2 及异步增量 Checkpoint 取代 `.bin` 单路汇总转换。

---

## 5. 18 周滚动执行排期与算力/存储需求

### 5.1 滚动甘特总览

| 周次 | 重点交付 | 依赖与验证目标 |
| --- | --- | --- |
| **W1** | MPI Gate M1/M2/M3（待集群空闲卡） | 完成 `1N×4R`、`2N` PM 与 $64^3$/$256^3$ 3D 引擎多节点闭合报告 |
| **W2** | **P1**：TSC/PCS + Interlacing + 谱梯度 | Ewald 全力误差 `< 1e-4` |
| **W3–W4** | **B1**：$k$-空间线性响应有质量中微子 | $P_m(k)$ 相对阻尼 `< 0.5%` |
| **W5–W6** | **P2**：5 阶多极 / L2P + FP64 高精度模式 | Ewald 全力 `~1–3e-5`，`256^3` L-I `< 1e-4` |
| **W7** | **P3**：潮汐/动力学步长与跨 rung 同步 | Plummer 漂移 `< 1e-6`，内区剖面 `< 1%` |
| **W8–W9** | **B2**：GPU MC-SIDM + $f(R)$ 多重网格 | 孤立 Hernquist/NFW 核形成与核塌缩曲线 |
| **W10** | **B3**：多质量 Zoom-in 全链路 | 等效 $4096^3$–$8192^3$ 暗晕 $R_{200}$ 零污染 |
| **W11–W14** | **B4-a**：全 GPU SPH/MFM 绝热流体核心 | Sod / Sedov–Taylor / K-H 不稳定性通过 |
| **W15–W18** | **B4-b**：GPU 辐射冷却 + Santa Barbara 基准 | Santa Barbara 绝热星系团温度/熵剖面通过 |

### 5.2 各阶段算力与存储配置

| 科研目标等级 | 推荐 GPU 规模 | 单卡显存需求 | 存储配置 | 典型场景 |
| --- | --- | --- | --- | --- |
| **精度开发 + 物理回归（P1–P3, B1–B2）** | `1–2× RTX 4090` / `A6000` | `24–48 GB` | `1–2 TB NVMe SSD` | $64^3$–$256^3$ Ewald / Zel'dovich / SIDM / 中微子开发验证 |
| **中等生产 + Zoom-in + 流体（B3–B4）** | `4× RTX 4090` 或 `4× A6000` | `24–48 GB/卡`（4 卡聚合 `96–192 GB`） | `4–8 TB NVMe + 20–50 TB NAS` | $512^3$–$1024^3$ 生产演化、多质量放大暗晕、Santa Barbara 流体星系团 |
| **远期 $2048^3+$ 超算巡天（S1–S4）** | `64–1,024+` H100/A100（RDMA 互联） | `80 GB HBM3/卡` | 并行文件系统（Lustre/GPFS `100 TB–PB`） | 万亿自由度超算巡天（当前已延后） |
