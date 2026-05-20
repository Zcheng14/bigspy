"""Shared test fixtures for bigspy."""

import os
import pytest
import numpy as np

# Paths to reference data — skip tests if unavailable
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCA_FILE = os.path.join(_REPO, "template", "BC03_Padova1994_chab_PCA_extend_new.fits")
SSP_FILE = os.path.join(_REPO, "template", "SSP_BC03_Padova1994_chab.fits")
TEST_PKL = os.path.join(_REPO, "tests", "manga-7443-12703-28-28.pkl")

_HAS_DATA = os.path.exists(PCA_FILE) and os.path.exists(SSP_FILE) and os.path.exists(TEST_PKL)

requires_data = pytest.mark.skipif(not _HAS_DATA, reason="Reference data not found")


@pytest.fixture(scope="session")
def pca_file():
    if not _HAS_DATA:
        pytest.skip("No reference data")
    return PCA_FILE


@pytest.fixture(scope="session")
def ssp_file():
    if not _HAS_DATA:
        pytest.skip("No reference data")
    return SSP_FILE


@pytest.fixture(scope="session")
def test_data():
    """Load the manga-7443 test spectrum."""
    from bigspy.specfit import load_test_spectrum
    if not _HAS_DATA:
        pytest.skip("No reference data")
    return load_test_spectrum(TEST_PKL)


@pytest.fixture(scope="session")
def specfit_result(pca_file, test_data):
    """Run SpecFit once for reuse across MCMC tests."""
    from bigspy import SpecFit
    sf = SpecFit(pca_file)
    return sf.fit(
        wave=test_data["wave_obs"],
        flux=test_data["flux_obs"],
        error=test_data["error_obs"],
        mask=test_data["mask_obs"],
        z_sys=test_data["z"],
        mode="mode2",
    )
