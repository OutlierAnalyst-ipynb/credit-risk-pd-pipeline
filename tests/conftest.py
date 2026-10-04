import pytest

from credit_risk.data.generate import generate_applications


@pytest.fixture(scope="session")
def apps():
    return generate_applications(6000, "2023-01-01", 24, seed=3)
