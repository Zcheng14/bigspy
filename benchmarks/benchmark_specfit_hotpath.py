"""Benchmark P6: specfit mode1/mode2 hot-path precomputes — old vs new.

A) calz_unred per-iteration cost vs precomputed klam (micro)
B) run_mode1 end-to-end (synthetic + real spectrum)
C) run_mode2 end-to-end (synthetic)
D) fit_spectrum(mode="both") total (synthetic, context)

Run from repo root:  python benchmarks/benchmark_specfit_hotpath.py

Parity between old and new outputs is ASSERTED (tolerances 1e-15 (A) /
1e-12 (B-E); all measured exactly 0 on the reference machine — the new
code reproduces the old lmfit trajectories bit-for-bit).
"""
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)                         # repo root (for archive.v0_6d70457)
sys.path.insert(0, os.path.join(REPO, "tests"))  # tests/ for _synth

import numpy as np

if not os.path.isdir(os.path.join(REPO, "archive")):
    print("archive/ not found — old-vs-new benchmarks require it (pre-cleanup state).")
    sys.exit(0)

import _synth
import bigspy.specfit as new_sf
import archive.v0_6d70457.bigspy.specfit as old_sf

PCA_REAL = os.path.join(REPO, "template", "BC03_Padova1994_chab_PCA_extend_new.fits")
PKL_REAL = os.path.join(REPO, "tests", "manga-7443-12703-28-28.pkl")


def params_vec(res, ncomp):
    return np.array([res.params[f"a{i}"].value for i in range(ncomp)]
                    + [res.params["ve"].value, res.params["vd"].value,
                       res.params["ebv"].value, res.redchi])


def bench_mode1(mod, prep, velscale, n_repeat):
    times, res = [], None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        res = mod.run_mode1(prep["flux"], prep["error"], prep["mask"],
                            prep["temp_pca"], prep["wave"], prep["vsys"],
                            velscale, prep["sigma_dap"])
        times.append(time.perf_counter() - t0)
    return res, np.mean(times)


def bench_mode2(mod, prep, pca, wave_temp, velscale, res_m1, n_repeat):
    times, out = [], None
    for _ in range(n_repeat):
        mask = prep["mask"].copy()  # run_mode2 mutates mask
        t0 = time.perf_counter()
        out = mod.run_mode2(prep["flux"], prep["error"], mask, prep["temp_pca"],
                            prep["wave"], prep["vsys"], velscale,
                            0.0, 0.0, res_m1, pca, prep["it1"], wave_temp)
        times.append(time.perf_counter() - t0)
    return out, np.mean(times)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        pca_path = os.path.join(tmp, "pca.fits")
        _synth.make_pca_fits(pca_path)
        data, _ = _synth.make_synthetic_obs()
        # preprocess is unchanged by P6 — either package gives identical prep
        pca, wave_temp, velscale = new_sf.load_pca_templates(pca_path)
        prep = new_sf.preprocess_spectrum(data, wave_temp, pca)
        ncomp = prep["temp_pca"].shape[1]

        # ── A) calz micro: per-iteration dust-curve evaluation ──
        print("== A) dust curve per lmfit iteration (synthetic wave_fit, ~3400 px) ==")
        wave_fit = prep["wave"]
        ebv = 0.13
        n_iter = 2000
        t0 = time.perf_counter()
        for _ in range(n_iter):
            c_old = old_sf.calz_unred(wave_fit, -ebv)
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        klam = new_sf.calz_klam(wave_fit)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n_iter):
            c_new = 10.0 ** (0.4 * klam * (-ebv))
        t_new = time.perf_counter() - t0
        max_diff = float(np.max(np.abs(c_old - c_new)))
        assert max_diff <= 1e-15, f"parity violated (calz): {max_diff:.3e} > 1e-15"
        print(f"  old calz_unred    : {t_old / n_iter * 1e6:>8.1f} us/call")
        print(f"  klam build (once) : {t_build * 1e6:>8.1f} us")
        print(f"  new 10**(0.4*klam*(-ebv)): {t_new / n_iter * 1e6:>8.1f} us/call "
              f"-> speedup {t_old / t_new:>5.1f}x, max|d| = {max_diff:.1e}")

        # ── B) run_mode1 end-to-end ──
        print("\n== B) run_mode1 end-to-end (synthetic, n_repeat=3) ==")
        res_old, t_old = bench_mode1(old_sf, prep, velscale, 3)
        res_new, t_new = bench_mode1(new_sf, prep, velscale, 3)
        v_old, v_new = params_vec(res_old, ncomp), params_vec(res_new, ncomp)
        assert float(np.max(np.abs(v_old - v_new))) <= 1e-12, \
            f"parity violated (run_mode1 params): {float(np.max(np.abs(v_old - v_new))):.3e}"
        print(f"  old: {t_old:.3f} s | new: {t_new:.3f} s | speedup: {t_old / t_new:.2f}x")
        print(f"  max|dparam| = {float(np.max(np.abs(v_old - v_new))):.3e} "
              f"(0 = identical lmfit trajectory)")

        # ── C) run_mode2 end-to-end (same result_m1 fed to both) ──
        print("\n== C) run_mode2 end-to-end (synthetic, n_repeat=3) ==")
        out_old, t_old = bench_mode2(old_sf, prep, pca, wave_temp, velscale, res_old, 3)
        out_new, t_new = bench_mode2(new_sf, prep, pca, wave_temp, velscale, res_old, 3)
        print(f"  old: {t_old:.3f} s | new: {t_new:.3f} s | speedup: {t_old / t_new:.2f}x")
        diffs = [abs(out_old[k] - out_new[k]) for k in ("p1", "p2", "ebv", "chi2r")]
        d_slr = float(np.max(np.abs(out_old["slr_flux"] - out_new["slr_flux"])))
        d_A = float(np.max(np.abs(out_old["dust_A"] - out_new["dust_A"])))
        assert max(diffs) <= 1e-12 and d_slr <= 1e-12 and d_A <= 1e-12, \
            f"parity violated (run_mode2): {max(diffs):.3e}/{d_slr:.3e}/{d_A:.3e}"
        print(f"  max|dp1,p2,ebv,chi2r| = {max(diffs):.3e} | max|dslr_flux| = {d_slr:.3e} "
              f"| max|ddust_A| = {d_A:.3e}")

        # ── D) fit_spectrum(mode="both") total (context) ──
        print("\n== D) fit_spectrum(mode='both') total (synthetic, n_repeat=3) ==")
        times_old, times_new = [], []
        for _ in range(3):
            t0 = time.perf_counter()
            fit_old = old_sf.fit_spectrum(prep, pca, wave_temp, mode="both")
            times_old.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            fit_new = new_sf.fit_spectrum(prep, pca, wave_temp, mode="both")
            times_new.append(time.perf_counter() - t0)
        print(f"  old: {np.mean(times_old):.3f} s | new: {np.mean(times_new):.3f} s "
              f"| speedup: {np.mean(times_old) / np.mean(times_new):.2f}x")
        v_old, v_new = params_vec(fit_old["mode1_result"], ncomp), \
            params_vec(fit_new["mode1_result"], ncomp)
        d_m2 = [abs(fit_old["mode2_result"][k] - fit_new["mode2_result"][k])
                for k in ("p1", "p2", "ebv", "chi2r")]
        assert float(np.max(np.abs(v_old - v_new))) <= 1e-12 and max(d_m2) <= 1e-12, \
            f"parity violated (fit_spectrum): {float(np.max(np.abs(v_old - v_new))):.3e}"
        print(f"  max|dparam| (mode1) = {float(np.max(np.abs(v_old - v_new))):.3e} | "
              f"max|d| (mode2 scalars) = {max(d_m2):.3e}")

    # ── E) real MaNGA spectrum end-to-end (if LFS data present) ──
    if os.path.isfile(PCA_REAL) and os.path.isfile(PKL_REAL):
        print("\n== E) SpecFit.fit(mode='mode2') on real MaNGA spectrum (n_repeat=2) ==")
        data = new_sf.load_test_spectrum(PKL_REAL)
        kw = dict(wave=data["wave_obs"], flux=data["flux_obs"],
                  error=data["error_obs"], mask=data["mask_obs"],
                  z_sys=data["z"], mode="mode2")
        times_old, times_new = [], []
        res_old = res_new = None
        for _ in range(2):
            t0 = time.perf_counter()
            res_old = old_sf.SpecFit(PCA_REAL).fit(**kw)
            times_old.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            res_new = new_sf.SpecFit(PCA_REAL).fit(**kw)
            times_new.append(time.perf_counter() - t0)
        print(f"  old: {np.mean(times_old):.2f} s | new: {np.mean(times_new):.2f} s "
              f"| speedup: {np.mean(times_old) / np.mean(times_new):.2f}x")
        assert res_old.ve == res_new.ve and res_old.vd == res_new.vd, \
            "parity violated (real spectrum ve/vd)"
        assert abs(res_old.ebv[0] - res_new.ebv[0]) <= 1e-12, \
            f"parity violated (real spectrum ebv): {abs(res_old.ebv[0] - res_new.ebv[0]):.3e}"
        print(f"  identical ve/vd: {res_old.ve == res_new.ve and res_old.vd == res_new.vd} "
              f"| d_ebv = {abs(res_old.ebv[0] - res_new.ebv[0]):.3e}")
    else:
        print("\nReal MaNGA spectrum not found — skipping section E.")


if __name__ == "__main__":
    main()
