import numpy as np

from credit_risk.models.metrics import decile_table, ks_statistic
from credit_risk.models.scorecard import decide, pd_to_score


def test_scorecard_anchor_and_pdo():
    # PD at base odds (1/(1+20)) -> base score; doubling good:bad odds adds PDO points
    s0 = pd_to_score([1 / 21], 600, 20, 20)[0]
    s1 = pd_to_score([1 / 41], 600, 20, 20)[0]
    assert s0 == 600 and s1 == 620


def test_higher_pd_means_lower_score():
    s = pd_to_score([0.01, 0.1, 0.5], 600, 20, 20)
    assert s[0] > s[1] > s[2]


def test_decisions():
    assert decide([700, 575, 500], 590, 560).tolist() == ["APPROVE", "REVIEW", "DECLINE"]


def test_ks_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert ks_statistic(y, y.astype(float)) == 1.0
    rng = np.random.default_rng(0)
    assert ks_statistic(y, rng.random(100)) < 0.4


def test_decile_table_rank_orders(apps):
    t = decile_table(apps["default_12m"], -apps["credit_score"])
    assert t["actual_dr"].iloc[0] > t["actual_dr"].iloc[-1]
