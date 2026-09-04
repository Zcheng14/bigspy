# bigspy 性能优化 benchmark 报告（2026-09）

本轮对 6 个热点做**纯加速**优化（不改逻辑）。所有优化均以"位级一致（bit-exact）"为验收标准——
每个 benchmark 的 parity 列全部为 `0.0e+00`，即新旧实现输出**逐位相同**；且 parity 不是仅打印，
而是**内建断言**（脚本超差即报错退出），容差对齐项目现有测试容差。唯一在理论上允许偏差的
P3（dgemv→dgemm 求和顺序）实测也为 0。回归防护：优化前先补齐测试并在旧代码上跑绿，优化后全套通过。

**基线快照：`archive/v0_6d70457/`（目录名含 commit 短哈希，即 `src/bigspy` 在 commit `6d70457`
处的逐字节拷贝）。benchmark 与 archive **长期保留**，任何时候可复现。**

## 1. 环境

| 项目 | 值 |
|---|---|
| CPU | Intel Xeon Gold 6248 @ 2.50GHz（80 核） |
| 内存 | 754 GB |
| Python | 3.10.20（conda env `bigspy`） |
| numpy / scipy | 2.2.5 / 1.15.3 |
| astropy / lmfit / ultranest | 6.1.7 / 1.3.4 / 4.5.0 |
| 测试基线 | 优化前 138 passed + 2 skipped（40.96 s）；优化后 **156 passed + 2 skipped（~19 s）** |

## 2. 方法

- **old**：`archive/v0_6d70457/bigspy`（优化前全包逐字节快照，相对导入自洽）；**new**：安装的 `bigspy`。
- 计时：`time.perf_counter`，n_repeat=5（端到端 2–3），报均值；种子固定的 `numpy.random.RandomState`
  （输入无随机性：old/new 拿到完全相同的输入）；均含预热调用（排除 BLAS/惰性初始化的一次性开销）。
- **parity 断言**：同一输入下 old/new 输出的 `max|Δ|`（或 `rel`）必须 ≤ 容差，否则 `AssertionError`。
  容差取项目测试中已有的等价性容差：SFH/interp/calz 1e-15（位级），CSP/卷积矩阵/mode2 1e-12，
  call_batch/chi2 链路 rtol 1e-10。本机实测全部为 0。
- benchmark 脚本：`benchmarks/benchmark_*.py`，从仓库根 `python benchmarks/benchmark_<name>.py` 运行
  （不匹配 pytest 的 `test_*` 收集规则）。合成数据来自 `tests/_synth.py`，真实数据为 LFS 的
  MaNGA 测试谱 + BC03 SSP/PCA 模板（缺失时脚本打印说明并跳过对应节）。

## 3. P1 — 卷积矩阵向量化构建 + 缓存（mcmc/kinematics.py）

原实现为纯 Python 双重循环（每个 MCMC likelihood batch 重建 n_pix×kernel 次），且类内缓存
`_get_conv_matrix` 是死代码。新实现：一次性索引散射构建（K 逐位不变）+ 模块级
`lru_cache(maxsize=4)`（冻结只读），`gauss_convolve_batch` 与 `VelocityBroadening` 均走缓存。
MCMC 期间 ve/vd 固定 → 矩阵每次采样只建一次。`K @ X` 保留（换直接卷积会改求和顺序，不做）。

```
== A) _build_convolution_matrix: old double loop vs vectorized ==
 n_pix |  sigma |   x0 |   old (s) |  new (s) |  speedup |  max|dK|
----------------------------------------------------------------------
   150 |   1.74 |  0.0 |    0.0012 |    0.0001 |     9.9x |  0.0e+00
   150 |   8.00 |  1.3 |    0.0036 |    0.0002 |    24.1x |  0.0e+00
  3129 |   1.74 |  0.0 |    0.0325 |    0.0269 |     1.2x |  0.0e+00
  3129 |   4.30 |  1.3 |    0.0626 |    0.0247 |     2.5x |  0.0e+00
  3129 |   8.00 |  1.3 |    0.0975 |    0.0205 |     4.8x |  0.0e+00

== B) gauss_convolve_batch (N, 3129): old rebuild-every-call vs new cached ==
     N |   old (s) | new 1st (s) | new cached (s) |  speedup | max|dout|
------------------------------------------------------------------------------
    10 |    0.0917 |      0.0179 |         0.0010 |    91.9x |   0.0e+00
    50 |    0.0899 |      0.0199 |         0.0021 |    42.7x |   0.0e+00
   200 |    0.0930 |      0.0239 |         0.0066 |    14.1x |   0.0e+00
   400 |    0.1003 |      0.0292 |         0.0110 |     9.1x |   0.0e+00
```

**结论：构建 1.2–24×；稳态批量卷积 9–92×（旧代码每 batch 重建，新代码仅矩阵乘）。parity 断言 tol=1e-12，实测 0。**

## 4. P2 — call_batch 插值/掩码预计算（mcmc/likelihood.py）

原实现每次调用重建 `scipy.interpolate.interp1d`（构造+查找开销）并重算 5500 Å 掩码。新实现：
`__init__` 预计算 `_LinearInterpPlan`（逐算子复刻 scipy `_call_linear` 公式，含严格越界置 0）、
掩码索引与 dust 曲线引用；每调用只剩 gather + lerp。

```
== A) batch interpolation: interp1d (rebuilt per call) vs plan.apply ==
     N |    old (s) |    new (s) |  speedup |   max|d|
------------------------------------------------------------
    10 |     0.0007 |     0.0002 |     2.8x |  0.0e+00
   100 |     0.0050 |     0.0020 |     2.4x |  0.0e+00
   400 |     0.0248 |     0.0187 |     1.3x |  0.0e+00
  1000 |     0.0734 |     0.0617 |     1.2x |  0.0e+00

== B) Likelihood.call_batch, vd=0, offset obs grid (synthetic SSP) ==
     N |    old (s) |    new (s) |  speedup | max|dchi2| |      rel
----------------------------------------------------------------------
    10 |     0.0049 |     0.0015 |     3.3x |    0.0e+00 |  0.0e+00
   100 |     0.0079 |     0.0020 |     3.9x |    0.0e+00 |  0.0e+00
   400 |     0.0229 |     0.0058 |     3.9x |    0.0e+00 |  0.0e+00
  1000 |     0.0571 |     0.0158 |     3.6x |    0.0e+00 |  0.0e+00
```

**结论：孤立插值 1.2–2.8×；call_batch（vd=0，隔离 P1）3.3–3.9×。与 scipy interp1d 逐位相等
（测试用 `assert_array_equal` 钉死；benchmark 断言 tol：A 1e-15 / B rel 1e-10）。**

## 5. P3 — CSP build_batch 分组逐金属 dgemm（mcmc/csp.py）

原实现逐样本 Python 循环（每样本 searchsorted + 2 次小 dgemv，N=400 时 800 次小 BLAS 调用），
且 `logZ_grid` 每次调用重算。新实现：`__init__` 预计算 logZ_grid；按金属分组的 6 次 dgemm
（循环次数与 N 无关），边界语义（钳制/精确格点）逐位复现。

```
== Synthetic SSP (3 metal x 6 age x 761 px) ==
     N |    old (s) |    new (s) |  speedup |  max|dCSP| |      rel
----------------------------------------------------------------------
    10 |     0.0003 |     0.0002 |     2.0x |    0.0e+00 |  0.0e+00
   200 |     0.0104 |     0.0020 |     5.3x |    0.0e+00 |  0.0e+00
  1000 |     0.0523 |     0.0142 |     3.7x |    0.0e+00 |  0.0e+00

== Real SSP (6 x 196 x 3129 px) ==
     N |    old (s) |    new (s) |  speedup |  max|dCSP| |      rel
----------------------------------------------------------------------
    10 |     0.1104 |     0.0457 |     2.4x |    0.0e+00 |  0.0e+00
    50 |     0.5231 |     0.0646 |     8.1x |    0.0e+00 |  0.0e+00
   200 |     2.1271 |     0.0977 |    21.8x |    0.0e+00 |  0.0e+00
   400 |     4.2191 |    0.1332 |    31.7x |    0.0e+00 |  0.0e+00
```

**结论：合成 1.4–5.3×；真实 SSP 2.4–31.7×（N=400：4.22 s → 0.13 s）。实测 dgemv/dgemm 在本环境逐位一致（断言 tol=1e-12，实测 0）。**

## 6. P4 — DelayedExponentialSFH.evaluate_batch 向量化（mcmc/sfh.py）

原通用实现逐行构造对象再 evaluate。新实现：`DelayedExponentialSFH` 上加向量化覆盖（broadcast），
`SFHBase` 通用循环保留（自定义 SFH 子类不受影响）。含 `age_universe=14.0` 截断（与 `evaluate` 一致）。

```
     N |    old (s) |    new (s) |  speedup |  max|dSFR|
------------------------------------------------------------
    10 |     0.0002 |     0.0000 |     4.7x |    0.0e+00
   100 |     0.0014 |     0.0001 |    10.9x |    0.0e+00
  1000 |     0.0140 |     0.0019 |     7.3x |    0.0e+00
 10000 |     0.1432 |     0.0294 |     4.9x |    0.0e+00
```

**结论：4.7–10.9×，逐位一致（测试 `assert_array_equal` 钉死 batch == 逐对象循环；benchmark 断言 tol=1e-15，实测 0）。**

## 7. P5 — mean_filter 滑动窗口向量化（specfit.py）

原实现逐像素 Python 循环（每条光谱 mode2 拟合约 4.3 万次迭代、12 次调用）。新实现：
`sliding_window_view` + 沿末轴归约；`eff==0→1` 与 `mean==0→flux[i]` 回退、首尾 k 像素不动、
`n<=2k` 空循环等语义全部保留。

```
              case |  win |   old (s) |   new (s) |  speedup |   max|d|
------------------------------------------------------------------------
          unmasked |  200 |    0.0148 |    0.0003 |    49.3x |  0.0e+00
        mask=all-1 |  200 |    0.0171 |    0.0025 |     6.9x |  0.0e+00
       mask=random |  200 |    0.0171 |    0.0017 |     9.9x |  0.0e+00
          unmasked |  400 |    0.0140 |    0.0005 |    30.1x |  0.0e+00
        mask=all-1 |  400 |    0.0165 |    0.0042 |     4.0x |  0.0e+00
       mask=random |  400 |    0.0163 |    0.0031 |     5.3x |  0.0e+00
```

**结论：4.0–49.3×，逐位一致（pairwise 求和在连续末轴上与逐窗口 `np.mean`/`.sum` 完全相同——
PRE 测试用 `assert_array_equal` 对 2 种窗宽 × 3 种掩码验证；benchmark 断言 tol=1e-13，实测 0）。**

## 8. P6 — specfit mode1/mode2 热路径预计算

四处改动，全部为"提取 + 不变量外提"：①提取 `calz_klam(wave)`，`run_mode1` 循环外算一次，
`_m1_residual` 只剩 `10**(0.4·klam·(−ebv))`；②提取 `_gauss_kernel`，`run_mode2` 模板循环
kernel 只建 1 次（原 10 次），保留 `vd0<=0` 恒等分支；③ebv 扫描循环外提 `g`/`flux[g]/error[g]`/
误差列/`es`/`klam_c`（已核实 mask 循环内不变）；④`wave_temp` 切片外提。

```
== A) dust curve per lmfit iteration (synthetic wave_fit, ~3400 px) ==
  old calz_unred    :     87.1 us/call
  klam build (once) :     69.3 us
  new 10**(0.4*klam*(-ebv)):     21.2 us/call -> speedup   4.1x, max|d| = 0.0e+00

== B) run_mode1 end-to-end (synthetic, n_repeat=3) ==
  old: 1.394 s | new: 0.977 s | speedup: 1.43x
  max|dparam| = 0.000e+00 (0 = identical lmfit trajectory)

== C) run_mode2 end-to-end (synthetic, n_repeat=3) ==
  old: 0.292 s | new: 0.039 s | speedup: 7.53x
  max|dp1,p2,ebv,chi2r| = 0.000e+00 | max|dslr_flux| = 0.000e+00 | max|ddust_A| = 0.000e+00

== D) fit_spectrum(mode='both') total (synthetic, n_repeat=3) ==
  old: 1.553 s | new: 1.000 s | speedup: 1.55x
  max|dparam| (mode1) = 0.000e+00 | max|d| (mode2 scalars) = 0.000e+00

== E) SpecFit.fit(mode='mode2') on real MaNGA spectrum (n_repeat=2) ==
  old: 0.40 s | new: 0.14 s | speedup: 2.98x
  identical ve/vd: True | d_ebv = 0.000e+00
```

**结论：calz 逐迭代 4.1×；run_mode1 1.43×；run_mode2 7.53×；完整 fit 1.55×；真实 MaNGA 谱 SpecFit.fit 2.98×。lmfit 轨迹逐位一致（新旧参数解完全相同；断言 tol=1e-12，实测 0）。**

## 9. 组合结果 — MCMC 热路径（headline）

真实 MaNGA 测试谱 → SpecFit(mode2) → 双方各建 Likelihood（同一 SSP 文件、同一 SpecFit 输出），
在 UltraNest 向量化批量尺寸下计时。P1+P2+P3+P4 叠加：

```
== Real MaNGA spectrum via SpecFit(mode2) + real SSP ==
  vd=67.7 km/s (sigma_pix=0.98), n_wave_ssp=13160, n_obs=3129
     N |    old (s) |    new (s) |  speedup | max|dchi2| |      rel
----------------------------------------------------------------------
    10 |     0.3903 |     0.0664 |     5.9x |    0.0e+00 |    0.0e+00
    50 |     0.8375 |     0.1009 |     8.3x |    0.0e+00 |    0.0e+00
   200 |     2.6165 |     0.2858 |     9.2x |    0.0e+00 |    0.0e+00
   400 |     4.8881 |     0.3745 |    13.1x |    0.0e+00 |    0.0e+00
  1000 |    11.5821 |    0.8416 |    13.8x |    0.0e+00 |    0.0e+00
```

**结论：MCMC likelihood 热路径整体 5.9–13.8×（对数百万次调用的嵌套采样，一次典型 MCMC 运行
的 likelihood 总耗时同比例下降），chi2 逐位一致（断言 rtol=1e-10，实测 0）。**

### 内存注意事项（P1 缓存）

真实 SSP 全网格为 13160 px → 卷积矩阵约 1.38 GB。旧代码**每个 batch** 分配再释放这块内存；
新代码缓存持有（每次 MCMC 运行 1 个 key）。峰值内存相当，但常驻时间变长；`lru_cache(maxsize=4)`
最坏 4 个 key（进程内拟合多条不同 vd 的光谱时），可用 `clear_convolution_cache()` 主动释放。

## 10. 测试增量（优化前先对旧代码跑绿）

| 文件 | 新增测试 | 断言 |
|---|---|---|
| test_mcmc.py | `test_evaluate_batch_matches_loop_delayed`, `test_evaluate_batch_age_cutoff` | batch == 逐对象循环，`assert_array_equal` |
| test_mcmc.py | `test_matrix_matches_naive_loop` | K == 测试内复刻旧双循环，`assert_array_equal` |
| test_mcmc.py | `test_convolve_batch_offset_matches_rows` | x0=1.7 批量 == 逐行，atol=1e-12 |
| test_mcmc.py | `test_matrix_cache_reused`, `test_cached_matrix_readonly` | 缓存命中/重建计数、只读 |
| test_csp.py | `test_build_batch_grid_exact`, `test_build_batch_large_n_parity` | batch == 逐样本 build，atol=1e-12 |
| test_likelihood.py | `test_call_batch_offset_grid` | 偏移网格+越界点+加宽启用，vs 测试内旧链，rtol=1e-10 |
| test_likelihood.py | `test_interp_plan_matches_interp1d` | plan == scipy interp1d，`assert_array_equal` |
| test_specfit_synthetic.py | `test_matches_naive_reference`, `test_edge_pixels_untouched`, `test_window_wider_than_array` | mean_filter == 测试内旧循环，`assert_array_equal` |
| test_specfit_synthetic.py | `test_klam_reproduces_calz_unred`, `test_klam_branch_edges`, `test_module_klam_matches_reference` | klam 公式/分支逐位一致 |
| test_specfit_synthetic.py | `test_recovers_known_ebv` | mode1 恢复 ebv=0.15（旧代码实测恢复 0.1328，带宽 ±0.05） |
| test_specfit_synthetic.py | `test_deterministic_repeat` | run_mode2 两次调用完全一致 |

另有 `_synth.make_synthetic_obs` 新增 `ebv` 参数（默认 0 时表达式逐字节不变）。

## 11. 复现说明

所有 benchmark 位于 `benchmarks/`，长期保留（含 `archive/v0_6d70457` 快照），随时可复现：

```bash
python benchmarks/benchmark_sfh.py
python benchmarks/benchmark_csp.py
python benchmarks/benchmark_conv_matrix.py
python benchmarks/benchmark_likelihood_interp.py
python benchmarks/benchmark_mean_filter.py
python benchmarks/benchmark_specfit_hotpath.py
python benchmarks/benchmark_mcmc_pipeline.py
```

（另 `benchmarks/benchmark_jax.py` 为更早的 NumPy-vs-JAX 基准，需安装 jax，与本轮无关。）
回归验证：`python -m pytest` → 156 passed, 2 skipped（2 个 skip 为本机未装 jax，属预期）。

## 12. 未做的候选（未来可做，均改变浮点求和顺序）

- `run_mode2` 十分量卷积合并为一次矩阵乘；`gauss_convolve_batch` 的 `K@X` 换直接卷积
- priors `norm.ppf → scipy.special.ndtri`、sampler 参数重组索引预计算（收益很小）
- 死代码 `_fit_residual`（specfit.py，无调用方）删除
