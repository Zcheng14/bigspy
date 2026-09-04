"""Tests for bigspy.mcmc.ssp.SSPLibrary on a synthetic FITS library (no LFS data)."""

import numpy as np

import _synth
from bigspy.mcmc.ssp import SSPLibrary


class TestSSPLibrarySynthetic:
    def test_load_shapes_and_values(self, synth_ssp_file):
        ssp = SSPLibrary(synth_ssp_file)
        assert ssp.n_metal == 3
        assert ssp.n_age == 6
        assert ssp.n_wave == 761
        np.testing.assert_array_equal(ssp.wave, _synth.SSP_WAVE)
        np.testing.assert_array_equal(ssp.metal, _synth.SSP_METAL)
        np.testing.assert_array_equal(ssp.time, _synth.SSP_TIME)
        np.testing.assert_array_equal(ssp.dt, _synth.SSP_DT)

    def test_wave_range_slice(self, synth_ssp_file):
        ssp = SSPLibrary(synth_ssp_file, wave_range=(4500, 6000))
        assert ssp.wave[0] >= 4500
        assert ssp.wave[-1] <= 6000
        assert ssp._spec.shape == (3, 6, ssp.n_wave)
        # Slice is a pure restriction of the full library
        full = SSPLibrary(synth_ssp_file)
        k = np.searchsorted(full.wave, ssp.wave[0])
        np.testing.assert_allclose(ssp.get_spectrum(1, 2),
                                   full.get_spectrum(1, 2)[k:k + ssp.n_wave],
                                   atol=1e-12)

    def test_get_spectrum_values_and_copy(self, synth_ssp_file):
        ssp = SSPLibrary(synth_ssp_file)
        for mi in range(ssp.n_metal):
            for ai in range(ssp.n_age):
                spec = ssp.get_spectrum(mi, ai)
                expected = (mi + 1) * (ai + 1) * _synth.SSP_WAVE / 5500.0
                np.testing.assert_allclose(spec, expected, atol=1e-12)
        # get_spectrum returns a copy: mutating it must not touch the library
        spec = ssp.get_spectrum(1, 2)
        spec[:] = -1.0
        assert not np.any(ssp._spec[1, 2, :] == -1.0)

    def test_get_mass(self, synth_ssp_file):
        ssp = SSPLibrary(synth_ssp_file)
        np.testing.assert_allclose(ssp.get_mass(0, 0), 1.0)
        np.testing.assert_allclose(ssp.get_mass(2, 5), 1.0)

    def test_repr(self, synth_ssp_file):
        ssp = SSPLibrary(synth_ssp_file)
        r = repr(ssp)
        assert "SSPLibrary" in r
        assert "n_metal=3" in r
        assert "n_age=6" in r
