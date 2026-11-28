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
| **Gate M1 / M2 / M3** | `2N×1G` 跨节点 PM + `64^3` 跨节点等价性 + `256^3` 跨节点扩展实测（`b[4,6]`） | **已通过（`2N×1G`）** | `Slurm 492167`（`2N×1G` PM `3.89e-16`）+ `Slurm 492195`（`b[4,6]` 真双节点 3D Hilbert `grav_mpi`）：`64^3` 与单卡步进完全一致，`pos_rms/L=2.59e-6`；`256^3`（`16.78M` 粒子）零错完跑（`54.27 s`，TCP ghost 通信占 `49.5 s`，验证了远期 S3 RDMA/LET 剪枝的必要性）。 |

---

## 3. 近期与中期开发计划（第 1–18 周）

> **第 1 周收尾**：一旦 `phy-gpus` 集群释放空闲 GPU，立即提交 [slurm/mpi-multinode-test.sbatch](slurm/mpi-multinode-test.sbatch) 完成 **Gate M1**（`1N×4R`, `2N×1G`, `2N×2G`）、**Gate M2**（$64^3$ 4-rank / 2-node vs 1G 力/位移/能量一致性）与 **Gate M3**（$256^3$ 多节点扩展曲线）。

### 主线一：数值精度跃迁（Track 1: P1–P3）

| 阶段 | 排期 | 目标硬件 | 核心改造内容与对应模块 | 定量验收门禁（Gate） |
| --- | --- | --- | --- | --- |
| **P1：高阶 PM 网格与去混叠** | 第 2 周（**已完成 Gate P1**） | `1× RTX 4090` | 已在 [src/grav_cuda_pm.cu](src/grav_cuda_pm.cu)、[src/grav_cuda_pm_dist.cu](src/grav_cuda_pm_dist.cu) 与 [src/grav_cuda_pm_mpi.cu](src/grav_cuda_pm_mpi.cu) 实现 `CIC/TSC/PCS`（`GRAV_PM_ORDER=2|3|4`）、半网格对称 `Interlacing`（`GRAV_PM_INTERLACE=1`）、精确 $-i\mathbf{k}$ 谱梯度（`GRAV_PM_GRAD=ik`）及 `GRAV_PM_ASMTH`；同步完成 `monofonIC`/`MUSIC` 统一 IC 适配层。 | **Gate P1 已通过**（[reports/p1-pm-precision-gate.json](reports/p1-pm-precision-gate.json)）：12 组单测零自力 $\le 1.43\times 10^{-14}$、牛顿第三定律 `0.0`；`Ewald` 全力中位误差（$\theta=0.3$）由 `1.034e-3` 压降至 **`5.58e-5`**（`PCS+Interlace+ik`，**18.5× 提升**）与 **`6.68e-5`**（`TSC+Interlace+ik`，**15.5× 提升**）。 |
| **P2：高阶多极展开 + 高精度模式** | 第 5–6 周（**已完成 Gate P2**） | `1–2× RTX 4090` | 已在 [src/grav_pm_config.cuh](src/grav_pm_config.cuh)、[src/multi_gpu_slab.cu](src/multi_gpu_slab.cu)、[src/grav_domain_3d_engine.cuh](src/grav_domain_3d_engine.cuh) 与 [src/grav_domain_3d_mpi.cuh](src/grav_domain_3d_mpi.cuh) 实现 **`GRAV_PRECISION_MODE=high`**（自动启用 `PCS + Interlacing + -ik`、扩展截断 `GRAV_PM_RCUT=5.5`、精确 `erfc/exp` 短程核、`mac=1e-4`、`theta=0.25` 与每步完整树重建）。 | **Gate P2 已通过**（[reports/b1-p2-gate.json](reports/b1-p2-gate.json)）：`Ewald` 全力中位误差压降至 **`1.014e-5`**（`p95=1.036e-4`，较基线 `1.034e-3` 提升 **`102×`**，跨入 `10^-5` 精度梯队）；`Zel'dovich` $64^3$ 位移 RMS **`0.0382%`**（提升 **`14.5×`**，远优于 `< 0.1%`）、速度 RMS **`0.0740%`**（提升 **`8.0×`**）。 |
| **P3：自适应时间步与辛守恒精细化** | 第 7 周（**已完成 Gate P3**） | `1× RTX 4090` | 已在 [src/grav_timestep_p3.cuh](src/grav_timestep_p3.cuh) 与 [src/multi_gpu_slab.cu](src/multi_gpu_slab.cu) 实现局部动力学时标 $\tau_{\rm dyn}=\sqrt{|\Phi|}/|\mathbf{a}|$、软化穿越判据、升阶滞回阻尼（`GRAV_TIDAL_TIMESTEP=1`）及严格辛全局同步步长模式（`GRAV_SYNC_STEP=1`）。 | **Gate P3 已通过**（[reports/p3-b2-gate.json](reports/p3-b2-gate.json)）：32K 孤立 Plummer 球 $t=0\to 4.0$ 总能量漂移由基线 `1.15e-5` 降至 **`7.64e-7`**（**提升 15.1×**，突破 `< 1e-6` 门禁），动量漂移 `2.44e-6`，维里比 `0.9956–0.9989`。 |

### 主线二：物理广度扩展（Track 2: B1–B4）

| 阶段 | 排期 | 目标硬件 | 核心改造内容与对应模块 | 定量验收门禁（Gate） |
| --- | --- | --- | --- | --- |
| **B1：有质量中微子（Massive Neutrinos）** | 第 3–4 周（**已完成 Gate B1**） | `1–2× RTX 4090` | 已在 [src/grav_neutrino.cuh](src/grav_neutrino.cuh) 及全部三套 PM 后端（单卡 / 单节点分布式 / 多节点 MPI）与演化驱动实现 **$k$-空间线性响应中微子（$W_\nu(k,a)$）** 与 Komatsu+2011 费米–狄拉克背景 $H(a)$，支持 `GRAV_NU_MASS_EV`、`GRAV_NU_NSPECIES` 及外部 `CLASS`/`CAMB` 2D 传递表 `GRAV_NU_TABLE`，零额外显存开销。 | **Gate B1 已通过**（[reports/b1-p2-gate.json](reports/b1-p2-gate.json)）：FD 背景拟合误差 `0.216%`；RTX 4090 三尺度实测（$\sum m_\nu=0.30\text{ eV}$，$f_\nu=0.02251$，覆盖 $k\ll k_{\rm fs}$ / $k\sim 0.25k_{\rm fs}$ / $k\gg k_{\rm fs}$）功率谱阻尼与精确线性微扰 ODE 偏差 **`0.009%–0.028%`**（远优于 `< 0.5%` 门禁）。 |
| **B2：超越 $\Lambda$CDM 暗区物理（SIDM + 修改引力）** | 第 8–9 周（**已完成 Gate B2**） | `1–2× RTX 4090` | 已在 [src/grav_sidm.cuh](src/grav_sidm.cuh) 实现基于常驻显存八叉树叶节点的 **Warp 循环赛无竞争 GPU 原生 MC-SIDM**（支持常截面与速度依赖截面 $\sigma(v)/m$）；在 [src/grav_fr_gravity.cuh](src/grav_fr_gravity.cuh) 与三套 PM 后端实现 **Hu–Sawicki $n=1$ $f(R)$ 修改引力**（含 $G_{\rm eff}(k,a)/G$ 与薄壳变色龙屏蔽）。 | **Gate B2 已通过**（[reports/p3-b2-gate.json](reports/p3-b2-gate.json)）：32K Hernquist 暗晕 GPU MC-SIDM 实测 `107,365` 次弹性散射（`25 s`），动能守恒残差 `9.35e-12`，成功复现引力热核塌缩（晕心密度 `+29.0%`、速度弥散 `+7.5%`）；$f(R)$ F4（$|f_{R0}|=10^{-4}$）双尺度功率谱增强与精确线性 ODE 吻合至 **`0.0001%` / `0.1794%`**。 |
| **B3：多质量粒子 Zoom-in 极高分辨率** | 第 10 周（**已完成 Gate B3**） | `1–4× RTX 4090` | 已在 [scripts/zoomin_pipeline.py](scripts/zoomin_pipeline.py) 与 [src/multi_gpu_slab.cu](src/multi_gpu_slab.cu) 打通多级嵌套 Zoom-in 初始条件生成、逐粒子软化 `GRAV_SOFT_IN`、`ZOOMIN_CONFIG` 质量动态范围诊断、`GRAV_FOF_LINK_MPC` 及低分辨率污染/GPU FoF 次暗晕审计管线。 | **Gate B3 已通过**（[reports/b3-zoomin-gate.json](reports/b3-zoomin-gate.json)）：单卡 RTX 4090 完成 9 层嵌套 **$4096^3$ 等效** Zoom-in 演化（`1.16M` 粒子、`16,777,216×` 质量比、$0.61\text{ kpc}$ 软化），$R_{200c}$ 内 `49,300` 颗高分辨率粒子、**`R200c`/`2×R200c` 内零低分辨率污染**（最近低分辨粒子在 `9.82×R200c` 外），GPU FoF 解析出 `35` 个次暗晕，$R_{200c}/M_{200c}$ 相对 $2048^3$ 收敛至 `1.20% / 3.59%`。 |
| **B4：全 GPU 流体力学（SPH / MFM + 辐射冷却）** | 第 11–18 周（**已完成 Gate B4**） | `1–4× RTX 4090` | 已在 [src/grav_cuda_hydro.h](src/grav_cuda_hydro.h)、[src/grav_cuda_hydro.cu](src/grav_cuda_hydro.cu) 与 [src/grav_hydro_driver.cu](src/grav_hydro_driver.cu) 构建全 GPU 显存常驻 3D Wendland C2 SPH 求解器（GPU 空间哈希紧邻搜索、自适应 $h_i$ 与 $\Omega_i$、Monaghan 粘性 + Balsara 开关、Price 人工热传导、隐式子步 Sutherland–Dopita 辐射冷却及耦合 DM+Gas 自引力），零上游 SWIFT 依赖（`make hydro`）。 | **Gate B4 5/5 全部通过**（[reports/b4-hydro-gate.json](reports/b4-hydro-gate.json)，RTX 4090）：**3D Sod 激波管**（$\rho/P/v_x$ L1 相对误差 `2.28% / 2.56% / 2.54%`，能量漂移 `0.089%`）、**3D Sedov–Taylor $48^3$**（激波半径误差 **`0.67%`**）、**Kelvin–Helmholtz**（横向动能增长 **`516.1×`**）、**GPU 辐射冷却**（$10^6\to 10^4\text{ K}$，能量账本误差 **`2.44e-15`**）、**Santa Barbara 绝热星系团**（能量守恒 `0.98%`，熵剖面 $K(r)$ 单调增长 `13.96×`）。 |

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

---

## 6. v3.x / v4.0 演进规划：插件化扩展、化学/重子演化与工程瓶颈清单

> **状态基线**：截至 `v2.0.0`（commit `0140ff9`），`P1–P3`、`B1–B4`、宇宙学耦合 PM+SPH（`grav_cosmo_hydro`）、真双节点 MPI（`b[4,6]` Gates M1/M2/M3）以及 v3.0 初步重构（PM/流体零 SWIFT 依赖 + `--config` INI 参数文件 + `GravPhysics` 抽离）已全部通过实测验真。下文记录下一阶段（v3.x / v4.0）待推进的**功能扩展**与**工程瓶颈**，供后续排期与合作者按需认领。

### 6.1 可插拔物理插件架构（Plugin / Stage Registry）

- **统一 `PhysicsStage` 插件接口**：将当前硬编码在主循环中的各物理模块抽象为标准生命周期钩子（`on_init` / `on_pre_pm` / `on_post_force` / `on_post_kick` / `on_snapshot`），允许第三方以单个 `.cuh` 头文件或 `.so` 动态库注册新物理，无需改动主引擎循环。
- **插件目录与配置绑定**：在 `plugins/` 下提供模板插件（如自定义暗物质截面、自定义暗能量 $w(z)$、自定义外场势、自定义在线诊断），在 `--config run.ini` 中通过 `[plugins] enable = ...` 动态启用。
- **TreePM + SPH 彻底合流**：将 [src/grav_cosmo_hydro.cu](src/grav_cosmo_hydro.cu) 的 `PartType0` 气体粒子直接合入 [src/multi_gpu_slab.cu](src/multi_gpu_slab.cu) 的短程八叉树 + 3D 域分解主循环，使流体也能享受短程树分辨率、多质量 Zoom-in 与在线 FoF。

### 6.2 天体化学网络与辐射冷却（Chemistry & Cooling）

- **非平衡原初化学网络（GPU 求解器）**：在现有 Sutherland–Dopita 查表冷却基础上，引入 6–9 物种原初化学网络（$\text{H}, \text{H}^+, \text{He}, \text{He}^+, \text{He}^{++}, e^-, \text{H}_2, \text{H}^-, \text{H}_2^+$），采用 GPU 隐式 BDF / Rosenbrock 子步积分，支持第一代恒星（Pop III）与低金属度矮星系冷却。
- **UV 背景光致电离与自屏蔽**：接入 Haardt & Madau (2012) / Faucher-Giguère (2020) 随红移演化的光致电离加热率 $\Gamma(z)$ 与 Rahmati+2013 氢自屏蔽修正。
- **金属丰度追踪与金属线冷却**：为气体与恒星粒子增加多元素金属质量分数数组（总金属度 $Z$ 及 $\text{C}, \text{N}, \text{O}, \text{Mg}, \text{Si}, \text{Fe}$），对接 `Grackle` / `CHIMES` / `Ploeckinger & Schaye` 查表式金属线冷却与尘埃冷却。

### 6.3 重子演化与星系形成次网格物理（Galaxy & Baryon Evolution）

- **恒星演化与化学增丰（Stellar Evolution）**：按 IMF（Chabrier / Salpeter）与恒星寿命表追踪星族年龄、质量损失及 **II 型超新星（SNII）**、**Ia 型超新星（SNIa）**、**AGB 星风** 的延迟金属返还（化学增丰注入邻近 SPH 气体）。
- **恒星反馈（Stellar Feedback）**：在现有 Schmidt 恒星形成律基础上增加随机热反馈（Dalla Vecchia & Schaye 2012，固定 $\Delta T_{\rm heat}$ 避免数值过冷却）与动能星风反馈。
- **超大质量黑洞与 AGN 反馈（SMBH & AGN）**：
  - 暗晕在线 FoF 触发黑洞种子植入（$M_{\rm halo} > M_{\rm seed}$）；
  - GPU 邻域 Bondi–Hoyle 吸积率计算与 Eddington 上限截断；
  - 双模式 AGN 反馈（高吸积率热/类星体模式 + 低吸积率喷流/射电模式）。
- **非线性 $f(R)$ GPU 多重网格松弛**：在现有线性 Yukawa + 薄壳屏蔽基础上，补充全非线性 Red–Black Gauss–Seidel GPU 多重网格标量场求解器，对标 `ECOSMOG` / `MG-GADGET` 强非线性小尺度。
- **在线 GPU 科学分析**：6D 相空间 GPU 子暗晕识别（类 `ROCKSTAR` / `SUBFIND`）、在线光锥弱引力透镜（HEALPix 汇聚图 $\kappa$）与热 SZ / X-ray 射线投影。

### 6.4 已知工程与性能瓶颈清单（v3.x / v4.0 优化目标）

| 编号 | 瓶颈模块 | 当前实测瓶颈现象 | 目标优化方案与预期收益 |
| --- | --- | --- | --- |
| **E1** | **跨节点通信**（`grav_domain_3d_mpi.cuh`） | `b[4,6]` `256^3` 跨节点实测 `54.27 s` 中有 `49.46 s`（`91%`）花在 TCP 整片 ghost 粒子拷贝 | 引入 **Locally Essential Tree（LET）远程树剪枝**（仅传远端会展开的多极节点，数据量降 `10–50×`）+ **GPU-Aware MPI / NCCL / GPUDirect RDMA** + 通信与本地力计算异步重叠。 |
| **E2** | **显存占用**（`multi_gpu_slab.cu`） | 当前约 `~200 B/粒子`（双精度坐标 + 树/打包副本），单张 24 GB 卡粒子上限偏紧 | 粒子坐标改相对 cell 中心的紧凑 FP32 / 32-bit 定点编码 + 紧凑树节点，压降至 **`~90–110 B/粒子`**（同显存可装 `2×` 粒子规模）。 |
| **E3** | **主驱动单体臃肿** | `multi_gpu_slab.cu`（6,800 行）、`grav_domain_3d_engine.cuh`（3,500 行）、`grav_domain_3d_mpi.cuh`（2,100 行）存在大量重复演化主循环 | 抽离统一的 `SimulationEngine` + 可插拔 `DomainBackend`（`1D` / `3D-NVLink` / `3D-MPI`），消除三份重复循环，将单文件控制在 `< 1,000` 行。 |
| **E4** | **最后残留外部头文件依赖** | `pm / pm-mpi / hydro / cosmo-hydro / fof` 已 100% 零依赖，仅 `multi_gpu_slab` 的 SWIFT FMM 多极核仍需 `KERNEL_BUILD_DIR` | 将 SWIFT 软化/M2P 多项式原生内置为独立头文件，实现全仓 `git clone && make` 真正零外部依赖。 |
| **E5** | **高精度模式短程力下限** | `GRAV_PRECISION_MODE=high` 当前 Ewald 误差 `1.01e-5`，受短程树力 FP32 累加下限制约 | 短程 P2P / M2P 力累加升级为 FP64（`double4`）混合精度，目标将 Ewald 全力误差进一步压至 **`~1–3e-6`**。 |
| **E6** | **I/O 与 Checkpoint** | 目前需通过 Python 脚本在 HDF5 与 `.bin` 之间中转 | 接入原生 Parallel HDF5 / ADIOS2 直接读写与异步增量 Checkpoint，省去中间 `.bin` 落盘与转换时间。 |
