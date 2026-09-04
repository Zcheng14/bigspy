"""Tests for bigspy.mask (emission-line table and masking utilities)."""

import numpy as np

from bigspy.mask import EMISSION_LINES, build_emission_mask, mask_emlines_detailed


class TestEmissionLines:
    def test_count_and_bounds(self):
        assert len(EMISSION_LINES) == 58
        for lo, hi in EMISSION_LINES.values():
            assert lo < hi
            assert 3000 < lo < hi < 10000


class TestBuildEmissionMask:
    def test_custom_regions_exact(self):
        wave = np.linspace(4000, 6000, 2001)  # 1 A pixels
        regions = [(5000, 5010), (5500, 5510)]
        mask = build_emission_mask(wave, regions)
        expected = np.ones(len(wave), dtype=bool)
        for lo, hi in regions:
            expected[(wave >= lo) & (wave <= hi)] = False
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, expected)

    def test_default_list(self):
        wave = np.array([4000.0, 5400.0, 6560.0, 5450.0])
        mask = build_emission_mask(wave)
        assert mask.dtype == bool
        assert mask[1] is np.True_          # 5400 A not covered by any line
        assert not mask[2]                   # 6560 A inside l6555/l6575
        assert mask[3] is np.True_           # 5450 A not covered

    def test_no_overlap_all_true(self):
        wave = np.linspace(2000, 2500, 100)
        mask = build_emission_mask(wave)
        assert np.all(mask)


class TestMaskEmLinesDetailed:
    def test_masks_default_lines(self):
        wave = np.linspace(4000, 7000, 3001)
        mask_out, lines = mask_emlines_detailed(wave, np.ones(len(wave), dtype=bool))
        np.testing.assert_array_equal(mask_out, build_emission_mask(wave))
        assert lines == EMISSION_LINES

    def test_mask_add_extends(self):
        wave = np.linspace(4000, 6000, 2001)
        _, lines = mask_emlines_detailed(
            wave, np.ones(len(wave), dtype=bool), mask_add={"custom": [4999, 5001]})
        assert "custom" in lines
        assert len(lines) == len(EMISSION_LINES) + 1
        mask_out, _ = mask_emlines_detailed(
            wave, np.ones(len(wave), dtype=bool), mask_add={"custom": [4999, 5001]})
        assert not np.all(mask_out[(wave >= 4999) & (wave <= 5001)])

    def test_input_false_preserved(self):
        wave = np.linspace(4000, 6000, 2001)
        mask_in = np.ones(len(wave), dtype=bool)
        mask_in[[10, 20, 1500]] = False      # outside any emission region
        mask_out, _ = mask_emlines_detailed(wave, mask_in)
        assert not mask_out[10]
        assert not mask_out[20]
        assert not mask_out[1500]
