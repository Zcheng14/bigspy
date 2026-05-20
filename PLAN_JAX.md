# bigspy JAX 加速计划书

> 分支: `jax-experiment` | 状态: 待实施

---

## 零、当前状态

| 项目 | 状态 |
|------|------|
| `jax` 0.9.1 (CPU) | ✅ |
| `numpyro` 0.20.1 | ✅ |
| `jax-experiment` 分支 | ✅ |
| 32 tests pass (NumPy) | ✅ |

---

## 一、改造目标

将 `MCMCFitter` 的 likelihood 计算改为 JAX 实现，目标：

1. **JIT 编译** — CPU 上 CSP 构建提速 5-10×
2. **自动微分** — `jax.grad` 提供 `∂χ²/∂θ`，为 NUTS 做准备
3. **GPU-ready** — 代码不改直接跑 GPU（如果有）
4. **可选 NUTS** — 用 NumPyro 替代 UltraNest，高维参数更快

**不改的**：SpecFit（lmfit 依赖 NumPy，JAX 化收益小，不动）

---

## 二、核心代码：`call_batch` → JAX

### 2.1 当前计算图（NumPy, `likelihood.py`）

```
la:12:likelihood.py
SFH weights    →  CSP_.build_batch  →  VelocityBroadening  →  5500 归一化  →  Dust  →  interp1d  →  χ²
(Python loop)     (np.dot × N)         (矩阵乘法，已批量化)      (np.median)    (mul)    (scipy)      (sum)
________________________________________________________________
瓶颈: Python for-loop 逐条金属丰度插值
```

### 2.2 JAX 改造方案

**新建文件**: `src/bigspy/mcmc/likelihood_jax.py`

**核心函数**: `compute_chi2_jax` — 纯函数，无副作用，可 `@jax.jit`

```python
@jax.jit
def compute_chi2_batch_jax(
    logZ_arr,          # (N,)
    sfh_params,        # (N, n_sfh)
    spec_3d,           # (n_metal, n_age, n_wave_ssp) — SSP 光谱立方
    time_grid, dt,     # SSP age grid
    metal_grid,        # metallicity grid
    conv_matrix,       # (n_wave_ssp, n_wave_ssp) — 预计算高斯卷积矩阵
    dust_curve,        # (n_wave_ssp,) — 消光曲线
    obs_wave,          # (n_obs,) — 观测波长
    obs_flux, obs_err, obs_mask,  # 观测数据
) -> chi2:             # (N,)
    ...
```

### 2.3 NumPy → JAX 函数映射

| NumPy / SciPy | JAX 等价 | 备注 |
|---------------|----------|------|
| `np.dot(a, b)` | `jnp.dot(a, b)` | 直接替换 |
| `np.median(x, axis=1)` | `jnp.median(x, axis=1)` | 直接替换 |
| `np.percentile(...)` | 不需要 | likelihood 不用 |
| `scipy.interpolate.interp1d` | `jnp.interp` | `jnp.interp(x_new, x_old, y_old)` |
| `np.where(cond, a, b)` | `jnp.where(cond, a, b)` | 直接替换 |
| `np.sum(x**2, axis=1)` | `jnp.sum(x**2, axis=1)` | 直接替换 |
| Python for-loop (金属插值) | `jax.lax.switch` 或向量化 | **关键改造点** |

### 2.4 关键挑战：金属丰度插值向量化

当前 `CSPBuilder.build_batch`:

```python
for j in range(N):                      # Python loop — 无法 JIT
    logM = logZsun_arr[j]
    if logM <= logZ_grid[0]:
        result[j] = np.dot(w[j], spec[0])
    elif logM >= logZ_grid[-1]:
        result[j] = np.dot(w[j], spec[-1])
    else:
        i = np.searchsorted(logZ_grid, logM)
        f = (logM - logZ_grid[i-1]) / (...)
        result[j] = (1-f)*np.dot(...) + f*np.dot(...)
```

JAX 改造 — 用 `jax.vmap` + `jax.lax.switch` 消除循环：

```python
@jax.jit
def build_csp_batch_jax(logZ_arr, sfr_weights, spec_3d, metal_grid):
    # logZ_arr: (N,)
    # sfr_weights: (N, n_age)  — SFH × dt, 已归一化

    # 找到每个 logZ 对应的金属丰度插值区间
    idx = jnp.searchsorted(metal_grid, logZ_arr)  # (N,)
    idx = jnp.clip(idx, 1, len(metal_grid) - 1)

    f = (logZ_arr - metal_grid[idx-1]) / (metal_grid[idx] - metal_grid[idx-1])  # (N,)

    # 批量 dot: (N, n_age) @ (n_age, n_wave) → (N, n_wave)
    # 先对两个端点分别做 dot，再线性插值
    spec_lo = jnp.einsum('na,anw->nw', sfr_weights, spec_3d[idx-1])  # (N, n_wave)
    spec_hi = jnp.einsum('na,anw->nw', sfr_weights, spec_3d[idx])    # (N, n_wave)

    return (1 - f[:, None]) * spec_lo + f[:, None] * spec_hi
```

---

## 三、实施步骤

### Step 1: `likelihood_jax.py`（核心）

**新建**: `src/bigspy/mcmc/likelihood_jax.py`

```python
import jax.numpy as jnp
from jax import jit

@jit
def compute_chi2_batch_jax(
    logZ_arr, sfh_weights,         # SFH 权重 (已由用户侧计算好)
    spec_3d, metal_log_grid,       # SSP 数据 (预加载为 JAX array)
    conv_matrix, dust_curve,       # 速度展宽 + 消光 (预计算)
    obs_wave, obs_flux, obs_err, obs_mask,
):
    # 1. CSP 构建
    idx = jnp.clip(jnp.searchsorted(metal_log_grid, logZ_arr), 1, len(metal_log_grid)-1)
    f = (logZ_arr - metal_log_grid[idx-1]) / (metal_log_grid[idx] - metal_log_grid[idx-1])
    csp_lo = jnp.einsum('na,anw->nw', sfh_weights, spec_3d[idx-1])
    csp_hi = jnp.einsum('na,anw->nw', sfh_weights, spec_3d[idx])
    csp = (1 - f[:, None]) * csp_lo + f[:, None] * csp_hi

    # 2. 速度展宽
    csp = jnp.dot(csp, conv_matrix.T)

    # 3. 5500 归一化
    norms = jnp.median(csp, axis=1)
    norms = jnp.where(norms == 0, 1.0, norms)
    csp = csp / norms[:, None]

    # 4. 消光
    csp = csp * dust_curve[None, :]

    # 5. 插值到观测网格
    model = jnp.array([jnp.interp(obs_wave, ssp_wave, csp[i]) for i in range(N)])
    # TODO: 用 vmap 替代上面的 for-loop

    # 6. χ²
    residuals = (model - obs_flux) / obs_err
    chi2 = jnp.sum(residuals[:, obs_mask] ** 2, axis=1)
    return chi2
```

### Step 2: 包装类 `JAXLikelihood`

```python
class JAXLikelihood:
    """JAX-accelerated likelihood, API 兼容 NumPy Likelihood."""

    def __init__(self, ssp, ow, oflux, oerr, omask, ve, vd, dust, nr=(5450, 5550)):
        # 预计算所有不随参数变化的部分
        self._conv_matrix = _build_conv_matrix_jax(...)
        self._dust_curve = jnp.asarray(dust._curve)
        self._obs_wave = jnp.asarray(ow)
        self._obs_flux = jnp.asarray(oflux)
        self._obs_err  = jnp.asarray(oerr)
        self._obs_mask = jnp.asarray(omask)
        self._spec_3d = jnp.asarray(ssp._spec)
        self._ssp_wave = jnp.asarray(ssp.wave)
        self._metal_log_grid = jnp.log10(jnp.asarray(ssp.metal) / 0.02)
        self._time_grid = jnp.asarray(ssp.time)
        self._dt = jnp.asarray(ssp.dt)

        # 归一化观测
        n = self._med5500(ow, oflux, omask, nr)
        self._obs_flux = self._obs_flux / n
        self._obs_err  = self._obs_err / n

    def call_batch(self, logZsun_arr, sfh_class, sfh_params_2d):
        """Vectorized chi2, NumPy interface → JAX backend."""
        sfh_weights = _compute_sfh_weights_jax(sfh_class, sfh_params_2d,
                                                self._time_grid, self._dt)
        return np.asarray(compute_chi2_batch_jax(
            jnp.asarray(logZsun_arr), sfh_weights,
            self._spec_3d, self._metal_log_grid,
            self._conv_matrix, self._dust_curve,
            self._obs_wave, self._obs_flux, self._obs_err, self._obs_mask,
        ))
```

### Step 3: Benchmark 脚本

**新建**: `tests/benchmark_jax.py`

```python
"""Compare NumPy vs JAX likelihood performance."""
import time, numpy as np
from bigspy.mcmc.likelihood import Likelihood      # NumPy
from bigspy.mcmc.likelihood_jax import JAXLikelihood  # JAX

# 相同数据，分别跑 N=100, 500, 2000, 5000 组参数
# 输出: 加速比 vs 参数数量
```

**预期**:
| 批量大小 N | NumPy (s) | JAX JIT (s) | 加速比 |
|------------|-----------|-------------|--------|
| 100 | ~0.5 | ~0.05 (含编译) | ~10× |
| 500 | ~2.5 | ~0.15 | ~17× |
| 2000 | ~10 | ~0.4 | ~25× |

### Step 4（可选）: NumPyro NUTS 替代 UltraNest

```python
import numpyro
from numpyro.infer import MCMC, NUTS

def model():
    # 先验
    tau = numpyro.sample("tau", LogUniform(0.1, 10.0))
    logZ = numpyro.sample("logZsun", Uniform(-2.5, 0.5))
    # 似然
    chi2 = compute_chi2_batch_jax(...)
    numpyro.factor("obs", -0.5 * chi2)

nuts = NUTS(model)
mcmc = MCMC(nuts, num_warmup=200, num_samples=1000)
mcmc.run(rng_key)
posterior = mcmc.get_samples()
```

---

## 四、文件规划

```
src/bigspy/mcmc/
├── likelihood.py          # NumPy 版（不动）
├── likelihood_jax.py      # JAX 版（新建）  ← Step 1-2
├── sampler.py             # UltraNest（不动）
└── sampler_numpyro.py     # NumPyro NUTS（可选）← Step 4

tests/
└── benchmark_jax.py       # 性能对比（新建）  ← Step 3
```

---

## 五、风险与缓解

| 风险 | 缓解 |
|------|------|
| JAX 函数非纯 → JIT 编译失败 | 所有输入通过参数传入，不依赖 self.xxx 副作用 |
| `jnp.interp` 性能不如 `scipy.interp1d` | JIT 后仍快于逐条 Python loop |
| JAX CPU 第一次编译慢 | 用 `jax.block_until_ready()` 预热；后续调用即时 |
| NumPy ↔ JAX 数组频繁转换 | 尽可能一次性转换，likelihood 内部全 JAX |
| SFH 的 `evaluate` 非 JAX | 直接传 precomputed SFH weights（`sfr·dt/∑sfr·dt`） |
