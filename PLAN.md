# bigspy — Bayesian Inference of Galaxy Spectra (Python)

> 计划书 · 基于 `bigs_myself` 重构为可安装 Python 包

---

## 一、项目概述

将 `bigs_myself/code/` 下的独立脚本重构为可发布、可安装的 Python 包 `bigspy`。
核心功能：对星系光谱做两阶段拟合——SpecFit（视向速度、速度弥散、消光）→ MCMC（星族金属丰度、SFH 参数贝叶斯推断）。

**不包含**：SSP 模板构建、PCA 模板生成（以已有 FITS 作为标准输入）。

---

## 二、包结构

```
bigs_v2/
├── PLAN.md                          # 本文件
├── pyproject.toml
├── src/bigspy/
│   ├── __init__.py                  # 顶层 API 暴露
│   ├── specfit.py                   # SpecFit 类 + PCA 拟合引擎
│   ├── mcmc/
│   │   ├── __init__.py
│   │   ├── ssp.py                   # SSPLibrary（从 FITS 加载 SSP 模板）
│   │   ├── csp.py                   # CSPBuilder（SSP × SFH 卷积，含 build_batch）
│   │   ├── sfh.py                   # SFHBase 抽象 + DelayedExponentialSFH
│   │   ├── priors.py                # UniformPrior, LogUniformPrior, GaussianPrior, FixedPrior
│   │   ├── dust.py                  # DustAttenuation（Calzetti + 多项式，合并两套实现）
│   │   ├── kinematics.py            # VelocityBroadening + gauss_convolve_batch
│   │   ├── likelihood.py            # Likelihood（vectorized χ² loglike）
│   │   └── sampler.py               # UltraNestSampler 封装
│   ├── mask.py                      # 发射线掩码 + 11 个默认区间
│   ├── constants.py                 # C_LIGHT, DLOGW, WAVE_NORM 等物理常数
│   ├── utils.py                     # log_rebin, air_to_vacuum 等工具
│   └── io.py                        # FITS 读写、结果序列化
└── tests/
    ├── conftest.py
    ├── test_specfit.py
    ├── test_mcmc_sfh.py
    ├── test_mcmc_vectorized.py
    └── test_end2end.py
```

---

## 三、顶层 API 设计

### 3.1 SpecFit

输入为**裸数组**（通用性最强），FITS 读取作为可选便捷方法。

```python
from bigspy import SpecFit

sf = SpecFit(pca_fits="BC03_PCA.fits")

result = sf.fit(
    wave=wave_obs,          # ndarray, 观测波长 (Angstrom)
    flux=flux_obs,          # ndarray, 流量
    error=error_obs,        # ndarray, 误差
    mask=mask_obs,          # ndarray, bool 或 0/1
    z_sys=0.04,             # float, 星系系统红移
    mode="mode2",           # 默认 "mode2"（S/L 非参数消光），可选 "mode1"（仅 Calzetti）
    emission_mask=None,     # None=默认 11 区间，可传 list[(lo, hi)]
    neig=10,                # 使用 PCA 主成分数
)

# 输出：SpecFitResult
result.ve          # (value, error)   km/s
result.vd          # (value, error)   km/s
result.ebv         # (value, error)   Mode1 Calzetti
result.p1          # float            Mode2 消光参数
result.p2          # float
result.chi2        # float            拟合 χ²
result.bestfit     # ndarray          PCA 重建最佳拟合光谱
result.dust_curve  # callable(wave)   消光曲线函数
```

**FITS 快捷入口**（内部提取数组后走同一路径）：
```python
result = sf.fit(observed_fits="manga-7443.fits", z_sys=0.04)
```

### 3.2 MCMC

```python
from bigspy import MCMCFitter, UniformPrior, LogUniformPrior

mc = MCMCFitter(
    ssp_fits="SSP_BC03.fits",
    specfit_result=result,
    wave=wave_obs,
    flux=flux_obs,
    error=error_obs,
    mask=mask_obs,              # 可选，与 SpecFit mask 合并
    sfh_model="delayed",        # 默认，也可传自定义 SFHBase 子类
    wave_range=(3600, 7400),
    emission_mask=None,         # MCMC 独立掩码
)

# 方式 A：使用默认先验（各参数有预定义合理范围）
mc.run(
    n_live=400,
    chain_dir="out/chains_manga-7443",
)

# 方式 B：自定义先验
mc.run(
    n_live=400,
    priors={
        "logZsun": UniformPrior(-2.5, 0.5),     # 金属丰度
        "t0":      UniformPrior(0.1, 13.5),     # 形成开始时间 (Gyr)
        "tau":     LogUniformPrior(0.1, 10.0),  # 衰减时标 (Gyr)
    },
    chain_dir="out/chains_manga-7443",
    # UltraNest 参数透传
    frac_remain=0.5,            # 后验采样比例
    max_ncalls=200000,          # 最大似然调用次数
    dlogz=0.5,                  # 证据容差
    min_ess=200,                # 最小有效样本数
)

# 输出：内存
mc.bestfit           # {"logZsun": ..., "t0": ..., "tau": ...}
mc.posterior         # (N, 3) ndarray
mc.log_evidence      # float, log(Z)
mc.result            # UltraNest 原始结果字典（用于高级分析）
```

### 3.3 先验（Prior）系统

#### 内置先验类型

| 先验 | 用法 | 说明 |
|------|------|------|
| `UniformPrior(lo, hi)` | 均匀分布 | `p(x) ∝ 1`，用于无强先验信息的参数 |
| `LogUniformPrior(lo, hi)` | 对数均匀分布 | `p(log₁₀x) ∝ 1`，用于跨数量级的正参数（如 τ） |
| `GaussianPrior(mu, sigma)` | 高斯分布 | 有外部测量约束时使用 |
| `FixedPrior(value)` | 固定值 | 冻结参数，不参与采样 |

#### 默认先验

不传 `priors` 时，每个 SFH 模型自带预定义默认先验：

```python
# DelayedExponentialSFH 默认先验：
{
    "logZsun": UniformPrior(-2.5, 0.5),     # Z ∈ [0.0001, 0.05] 对应 log(Z/Z⊙) ∈ [-2.3, 0.4]
    "t0":      UniformPrior(0.1, 13.5),     # 形成开始时间 (Gyr)
    "tau":     LogUniformPrior(0.1, 10.0),  # 衰减时标 (Gyr)
}
```

#### 自定义先验

```python
from bigspy import MCMCFitter, UniformPrior, GaussianPrior, FixedPrior

mc = MCMCFitter(...)

mc.run(
    priors={
        "logZsun": GaussianPrior(mu=-0.5, sigma=0.3),  # 外部金属丰度约束
        "t0":      UniformPrior(0.5, 12.0),
        "tau":     FixedPrior(5.0),                    # 冻结 τ=5，只采样 logZ 和 t0
    },
    ...
)
```

`FixedPrior` 效果：参数不参与采样，UltraNest 维度 -1，似然函数中始终使用固定值。

### 3.4 UltraNest 参数透传

`mc.run()` 接受以下参数，其余以 `**ultranest_kwargs` 透传给 `ReactiveNestedSampler.run()`：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_live` | int | 400 | `min_num_live_points`，活跃采样点数 |
| `priors` | dict | None | 先验字典（None 用默认） |
| `chain_dir` | str | 必需 | UltraNest 输出目录 |
| `frac_remain` | float | 0.5 | 后验采样阶段保留的似然调用比例 |
| `max_ncalls` | int | None | 最大似然函数调用次数 |
| `dlogz` | float | 0.5 | 证据积分容差（越小越精确但越慢） |
| `min_ess` | int | 400 | 后验采样阶段最小有效样本数 |
| `Lepsilon` | float | 0.001 | 似然等值面精度 |
| `max_iters` | int | None | 最大迭代次数 |

透传参数：
```python
mc.run(
    n_live=600,
    chain_dir="out/chains_galaxy",
    dlogz=0.1,               # 更严格的证据精度
    max_ncalls=500000,
    # 其他 UltraNest 参数直接透传
    resume=True,               # 断点续跑
    show_status=True,          # 显示进度条
    num_test_samples=4,        # 收敛测试样本数
)
```

### 3.5 SFH 自定义扩展

```python
from bigspy.sfh import SFHBase
from bigspy import UniformPrior, LogUniformPrior

class MySFH(SFHBase):
    n_params = 3
    param_names = ["tau", "beta", "t_trunc"]

    # 默认先验（可选覆盖）
    default_priors = {
        "tau":      LogUniformPrior(0.1, 10.0),
        "beta":     UniformPrior(0.0, 5.0),
        "t_trunc":  UniformPrior(0.5, 13.0),
    }

    def evaluate(self, timegrid):
        """timegrid: SSP ages (Gyr), 0=early → max=present.
           Returns: SFR weight array, same shape as timegrid."""
        t = np.max(timegrid) - timegrid
        sfr = np.where(t < self._params["t_trunc"],
                       t**self._params["beta"] * np.exp(-t/self._params["tau"]), 0.0)
        sfr[timegrid > 13.8] = 0.0
        return sfr

mc = MCMCFitter(..., sfh_model=MySFH, ...)
```

`SFHBase` 抽象接口：
- `n_params: int` — 参数个数
- `param_names: list[str]` — 参数标签（用于 UltraNest）
- `default_priors: dict[str, Prior]` — 默认先验（用户不传 `priors` 时使用）
- `__init__(**params)` — 接收参数字典
- `evaluate(timegrid) -> ndarray` — 核心计算
- `evaluate_batch(timegrid, params_2d) -> ndarray` — vectorized 版本（基类提供 fallback）

---

## 四、输入/输出设计

### 4.1 输入约定

| 输入 | SpecFit | MCMC | 格式 |
|------|---------|------|------|
| PCA 模板 | ✅ 必需 | — | FITS 路径 |
| SSP 模板 | — | ✅ 必需 | FITS 路径 |
| 观测波长 | ✅ 必需 | ✅ 必需 | ndarray |
| 观测流量 | ✅ 必需 | ✅ 必需 | ndarray |
| 观测误差 | ✅ 必需 | ✅ 必需 | ndarray |
| 像素掩码 | ✅ 必需 | ✅ 可选 | ndarray(bool) |
| 系统红移 | ✅ 必需 | — | float |
| SpecFit 结果 | — | ✅ 必需 | SpecFitResult |

### 4.2 结果保存策略

**原则**：库不主动写盘，所有 I/O 由用户显式调用。

**SpecFit 阶段**：
| 结果 | 必须？ | 保存方式 |
|------|--------|----------|
| 拟合参数 + 误差 | ✅ | `result.save("out/specfit.fits")` |
| PCA 最佳拟合谱 | ⭕ | 同上 FITS（扩展 HDU） |
| 消光曲线 | ⭕ | 同上 FITS |
| 诊断图 | ❌ | `result.plot_fit(path)` / `result.plot_dust(path)` |

**MCMC 阶段**：
| 结果 | 必须？ | 保存方式 |
|------|--------|----------|
| 采样链 | ✅ | `chain_dir`（`run()` 时指定，UltraNest 管理） |
| 最佳拟合参数 | ⭕ | `mc.save_result("out/mcmc_result.fits")` |
| 后验样本 | ⭕ | `mc.posterior.tofile(...)` |
| CSP 最佳拟合谱 | ⭕ | 同上 FITS |
| 诊断图 | ❌ | `mc.plot_corner()` / `mc.plot_bestfit()` / `mc.plot_sfh()` |

### 4.3 典型输出目录

```
out/
├── specfit_manga-7443.fits          # SpecFit 全量结果
├── chains_manga-7443/               # UltraNest 链（内部结构由 UltraNest 管理）
│   ├── chains/
│   │   └── equal_weighted_post.txt
│   ├── results/
│   │   └── points.hdf5
│   └── ...
├── mcmc_result_manga-7443.fits      # 最佳拟合 + CSP 谱
└── figs/
    ├── fit_spectrum_manga-7443.png
    ├── dust_curve_manga-7443.png
    ├── corner_manga-7443.png
    ├── bestfit_manga-7443.png
    └── sfh_manga-7443.png
```

---

## 五、核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 包名 | `bigspy` | Bayesian Inference of Galaxy Spectra (Python)，与 BIGS 关联 |
| SFH 模型 | `DelayedExponential` 默认，移除 `GammaSFH` | 更物理、更常用 |
| SFH 扩展 | `SFHBase` 抽象基类 | 用户可自定义，开闭原则 |
| 消光模式 | Mode2（S/L 非参数）默认，保留 Mode1（Calzetti） | S/L 方法更精准，Calzetti 留作备选 |
| SpecFit 输入 | 裸数组（wave, flux, error, mask） | 通用性最强，不绑定 FITS |
| 发射线掩码 | 默认 11 区间 + 用户可覆盖/追加 | 兼顾便捷和灵活 |
| 先验系统 | 内置 Uniform / LogUniform / Gaussian / Fixed + 可自定义 | 覆盖常见需求，同时保持扩展性 |
| 先验默认值 | 各 SFH 模型自带 `default_priors` | 用户零配置可用，高级用户可覆盖 |
| UltraNest 参数 | `run()` 显式参数 + `**kwargs` 透传 | 常用参数有文档，其余不阻挡 |
| UltraNest 向量化 | `vectorized=True` + 真正批量计算 | 当前代码只是标量 loop |
| 文件保存 | 用户显式调用 `.save()` / `.plot_*()` | 库不应偷偷写文件 |

---

## 六、UltraNest vectorized 改造

### 6.1 现状

```python
class UltraNestSampler:
    def loglike(self, params):
        # params: (N, active_params)   ← 不含 FixedPrior 参数
        results = np.zeros(len(params))
        for i in range(len(params)):
            results[i] = self._single(params[i])   # 逐点算，vectorized 名存实亡
        return results

    # prior_transform 也需适配：
    # FixedPrior → 维度降低（不参与采样），似然中插入固定值
```


### 6.2 目标

| 组件 | 当前 | 改造后 |
|------|------|--------|
| `CSPBuilder.build` | 标量 → 1D | `build_batch` → `(N, n_wave)` |
| `SFH.evaluate` | 标量 SFH → 1D | `evaluate_batch` → `(N, n_age)` |
| `gauss_convolve` | `np.convolve` 逐条 | 预计算卷积矩阵 → `np.dot` 批量 |
| `DustAttenuation.apply` | 逐条 | NumPy 广播级向量化 |
| `np.interp` | 标量插值 | `scipy.interpolate.interp1d` 批量 |
| `Likelihood.loglike` | for loop | 纯矩阵运算 |

### 6.3 关键优化：高斯卷积矩阵

`σ_pix = vd / velscale` 在整个 MCMC 中不变 → 在 `VelocityBroadening.__init__` 时预计算一次卷积矩阵 `K (n_wave, n_wave)`，后续 `apply_batch(spectra_2d)` = `np.dot(spectra_2d, K.T)`。

### 6.4 验证

```python
# test_mcmc_vectorized.py
N, n_params = 1000, 3
params = np.random.uniform(...)
result_batch = like.loglike(params)          # (N,)
result_loop = np.array([like._single(p) for p in params])
assert np.allclose(result_batch, result_loop, rtol=1e-6)
```

---

## 七、实施计划

### Phase 1 — 拆分物理组件为独立模块（不改行为）

| # | 步骤 | 源文件 | 目标文件 |
|---|------|--------|----------|
| 1.1 | 创建 `src/bigspy/` 目录结构 + 空 `__init__.py` | — | — |
| 1.2 | 提取 `SSPLibrary` | `MCMC_fit.py` | `mcmc/ssp.py` |
| 1.3 | 提取 `CSPBuilder` | `MCMC_fit.py` | `mcmc/csp.py` |
| 1.4 | 提取 `DelayedExponentialSFH` + 定义 `SFHBase` | `MCMC_fit.py` | `mcmc/sfh.py` |
| 1.5 | 实现 Prior 类体系（Uniform, LogUniform, Gaussian, Fixed） | — | `mcmc/priors.py` |
| 1.6 | 合并两套 `calz_unred`/`DustAttenuation` | `fit_spec.py` + `MCMC_fit.py` | `mcmc/dust.py` |
| 1.7 | 合并两套 `gauss_convolve` + `VelocityBroadening` | `fit_spec.py` + `MCMC_fit.py` | `mcmc/kinematics.py` |
| 1.8 | 提取 `Likelihood` | `MCMC_fit.py` | `mcmc/likelihood.py` |
| 1.9 | 提取 `UltraNestSampler` | `MCMC_fit.py` | `mcmc/sampler.py` |
| 1.10 | 提取 SpecFit 核心 | `fit_spec.py` | `specfit.py` |
| 1.11 | 抽取常量/工具/掩码 | `fit_spec.py` + `MCMC_fit.py` | `constants.py`、`utils.py`、`mask.py` |
| 1.12 | **验证**：确保原 `pipeline.py` import 新模块行为不变 | — | — |

### Phase 2 — 封装公共 API

| # | 步骤 |
|---|------|
| 2.1 | `SpecFit` 类：`__init__` 加载 PCA，`fit()` 返回 `SpecFitResult` |
| 2.2 | `MCMCFitter` 类：`__init__` 组装 SSP+Likelihood，`run()` 触发采样 |
| 2.3 | `MCMCFitter.run()` 集成先验系统：读取 `SFH.default_priors`，合并用户 `priors` 参数 |
| 2.4 | `UltraNestSampler` 重构：`prior_transform` 由 Prior 对象驱动，支持 `FixedPrior` 降维 |
| 2.5 | `io.py`：FITS 读写 + 结果序列化 |
| 2.6 | `__init__.py` 暴露顶层 API（含所有 Prior 类） |
| 2.7 | `pyproject.toml` 声明依赖 |
| 2.8 | 输入类型适配：`str | Path | HDUList | ndarray` |

### Phase 3 — UltraNest `vectorized=True` 真向量化

| # | 步骤 |
|---|------|
| 3.1 | `CSPBuilder.build_batch(logZsun_1d, sfh_params_2d)` → `(N, n_wave)` |
| 3.2 | `SFHBase.evaluate_batch(timegrid, params_2d)` → `(N, n_age)` |
| 3.3 | `gauss_convolve_batch(spectra_2d, sigma_pix)` — 预计算卷积矩阵 + `np.dot` |
| 3.4 | `DustAttenuation.apply_batch` — NumPy 广播 |
| 3.5 | `np.interp` → `scipy.interpolate.interp1d` 批量插值 |
| 3.6 | `Likelihood.loglike(params_Nx3)` → 纯批量计算 |
| 3.7 | 验证：逐点结果 = 批量结果（`rtol=1e-6`） |

### Phase 4 — 集成测试 + 文档

| # | 步骤 |
|---|------|
| 4.1 | 端到端测试（`manga-7443`、`manga-8131`） |
| 4.2 | 与原 `pipeline.py` 输出对比，数值一致性验证 |
| 4.3 | `README.md` + API 文档 |

---

## 八、依赖清单

| 包 | 用途 |
|----|------|
| `numpy` | 数值计算 |
| `scipy` | 插值、信号处理 |
| `astropy` | FITS I/O |
| `lmfit` | SpecFit 非线性拟合 |
| `ultranest` | MCMC 嵌套采样 |
| `matplotlib` | 可视化（可选） |
| `corner` | Corner plot（可选） |

---

## 九、风险与缓解

| 风险 | 缓解 |
|------|------|
| `np.interp` → `interp1d` 性能退化 | 批量 `interp1d` 远快于逐点 loop |
| 卷积矩阵内存 `(n_wave, n_wave)` | ~4000² × 8B ≈ 128 MB，可接受 |
| 与原 `pipeline.py` 数值差异 | Phase 1.11 + Phase 4.2 回归测试，`rtol=1e-6` |
| UltraNest `vectorized=True` 行为约定 | Phase 3 先做小规模验证 |
| `GammaSFH` 移除后原测试数据 | Phase 4 仅测 `DelayedExponentialSFH` |

---

## 十、数据依赖

所有原始数据位于 `bigs_myself/` 目录，不拷贝进 `bigs_v2/`：

| 数据 | 路径 |
|------|------|
| PCA 模板 | `../bigs_myself/out_data/BC03_Padova1994_chab_PCA_extend_new.fits` |
| SSP 模板 | `../bigs_myself/out_data/SSP_BC03_Padova1994_chab.fits` |
| 测试光谱 | `../bigs_myself/test_data/` 或原 `bigs-V202309/` |
| SpecFit 参考结果 | `../bigs_myself/out_data/fit_result_*.fits` |
| MCMC 参考链 | `../bigs_myself/out_data/chains_*/` |
