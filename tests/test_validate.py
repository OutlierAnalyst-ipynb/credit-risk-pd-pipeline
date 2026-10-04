import numpy as np
import pytest

from credit_risk.data.validate import DataValidationError, validate_applications


def test_clean_data_passes(apps):
    validate_applications(apps)


def test_out_of_range_credit_score(apps):
    bad = apps.copy()
    bad.loc[0, "credit_score"] = 990
    with pytest.raises(DataValidationError, match="credit_score"):
        validate_applications(bad)


def test_null_in_required_field(apps):
    bad = apps.copy()
    bad.loc[0, "annual_income"] = np.nan
    with pytest.raises(DataValidationError, match="annual_income"):
        validate_applications(bad)


def test_missingness_spike_is_caught(apps):
    bad = apps.copy()
    bad.loc[: len(bad) // 2, "revolving_utilization"] = np.nan
    with pytest.raises(DataValidationError, match="null share"):
        validate_applications(bad)
