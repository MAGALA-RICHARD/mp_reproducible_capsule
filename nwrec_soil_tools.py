import numpy as np

import numpy as np
import pandas as pd


def exp_scale(x, *, steps=1, rate=None, half_life=None, doubling_time=None, mode="decay"):
    """
    Exponentially scale a value (or 1D array) over `steps`.

    Returns:
      - if x is scalar: shape (steps,)
      - if x is 1D array: shape (steps, len(x)) with each column scaled over time
    """
    # choose k
    if half_life is not None:
        k = -np.log(2.0) / float(half_life)
    elif doubling_time is not None:
        k = np.log(2.0) / float(doubling_time)
    elif rate is not None:
        k = float(rate) if mode == "growth" else -abs(float(rate)) if mode == "decay" else float(rate)
    else:
        raise ValueError("Provide one of: rate, half_life, or doubling_time.")

    t = np.arange(steps, dtype=float)
    factor = np.exp(k * t)  # (steps,)

    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return x * factor  # (steps,)
    if x.ndim == 1:
        return factor[:, None] * x[None, :]  # (steps, n)
    raise ValueError("x must be scalar or 1D array.")


def exp_bridge(start, end, n):
    """
    Exponential path from `start` to `end` in exactly `n` points.

    Requires start > 0 and end > 0 (multiplicative model).

    y[0]   = start
    y[-1]  = end
    y[t]   = start * exp(k * t), where k = ln(end/start) / (n-1)

    Parameters
    ----------
    start, end : float
        Positive start and end values.
    n : int
        Number of points (>=1).

    Returns
    -------
    np.ndarray
        Exponential sequence of length n.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([float(end)])
    if not (start > 0 and end > 0):
        raise ValueError("start and end must be positive for exponential bridging.")

    k = np.log(float(end) / float(start)) / (n - 1)
    t = np.arange(n, dtype=float)
    return float(start) * np.exp(k * t)


def exp_scale_df(df: pd.DataFrame, *, steps=10, rate=None, half_life=None, doubling_time=None, name='step') -> pd.DataFrame:
    """
    Create an exponential scaling over time for all numeric columns in a DataFrame.

    This is useful e.g. for generating decay curves or growth curves for soil
    properties (SAT, DUL, LL15, etc.) to simulate N-dilution, soil degradation,
    or scenario scaling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric columns to scale.
    steps : int
        Number of discrete exponential points to produce.
    rate : float, optional
        per-step exponent.
        - positive  → growth
        - negative  → decay
        Only needed if half_life and doubling_time are None.
    half_life : float, optional
        number of steps required to halve the value.
    doubling_time : float, optional
        number of steps required to double the value.
    name : str
        name for the added MultiIndex level (usually 'step').

    Returns
    -------
    scaled_df : pd.DataFrame
        MultiIndexed DataFrame:
            index = (step, original_index...)
            columns = same as `df`
        Each 'step' row is the exponential scaled value of the original df.

    Notes
    -----
    - The original df is not modified.
    - `df` can have non-numeric columns — they will be auto-ignored if selected before calling.

    Example
    -------
    >>> dg = water.groupby([...])[['SAT','DUL','LL15']].mean()
    >>> scaled = exp_scale_df(dg, steps=10, rate=-0.03)   # exponential decay
    """
    # choose exponent k
    if half_life is not None:
        k = -np.log(2.0) / float(half_life)
    elif doubling_time is not None:
        k =  np.log(2.0) / float(doubling_time)
    elif rate is not None:
        k = float(rate)
    else:
        raise ValueError("Provide one of: rate, half_life, or doubling_time.")

    # exponential factors for each discrete step
    factors = np.exp(k * np.arange(steps))     # shape (steps,)

    # apply factor to df per step, stack them into a MultiIndex
    return pd.concat({t: df.mul(f) for t, f in enumerate(factors)}, names=[name])


if __name__ == "__main__":
    # 1) Exponential decay: start at 100, half-life = 3 steps, 10 steps long
    exp_scale(100, steps=10, half_life=3, mode="decay")
    # -> [100., 79.37, 63.10, 50., 39.68, 31.50, 25., 19.84, 15.75, 12.50]

    # 2) Exponential growth: start at 5, doubling every 4 steps, 9 steps long
    exp_scale(5, steps=9, doubling_time=4, mode="growth")

    # 3) Use a raw per-step rate: y_t = x * exp(0.1*t) (growth)
    exp_scale(2.0, steps=6, rate=0.1, mode="growth")

    # 4) Bridge from 0.04 to 0.018 (e.g., critical N at low→high biomass) in 5 points
    exp_bridge(0.04, 0.018, 5)
    # unit tests
    # tests/test_exp_scale_df.py
    import numpy as np
    import pandas as pd
    import pytest


    def make_df():
        # simple numeric DataFrame with a MultiIndex-like index shape in mind
        idx = pd.Index([("10-20", "A", 1), ("10-20", "B", 2), ("20-30", "A", 3)], name="key")
        # but we'll keep it a plain Index to avoid coupling; content matters, not index type
        df = pd.DataFrame(
            {
                "SAT": [0.40, 0.38, 0.36],
                "DUL": [0.30, 0.28, 0.27],
                "LL15": [0.15, 0.14, 0.13],
            },
            index=idx,
        )
        return df


    def test_raises_when_no_time_param():
        df = make_df()
        with pytest.raises(ValueError):
            exp_scale_df(df, steps=5)  # no rate/half_life/doubling_time


    @pytest.mark.parametrize("rate", [0.0, 0.05, -0.05])
    def test_shape_and_level_name(rate):
        df = make_df()
        out = exp_scale_df(df, steps=7, rate=rate, name="step")
        # shape: (steps * len(df), ncols)
        assert out.shape == (7 * len(df), df.shape[1])
        # first level must be named as provided
        assert out.index.names[0] == "step"
        # values at step 0 equal original df (exp(0) == 1)
        pd.testing.assert_frame_equal(out.xs(0, level=0), df, check_dtype=False)


    def test_growth_matches_theory_with_rate():
        df = make_df()
        steps = 5
        rate = 0.03
        out = exp_scale_df(df, steps=steps, rate=rate)
        # last step t = steps-1
        t_last = steps - 1
        factor_last = np.exp(rate * t_last)
        expected_last = df * factor_last
        pd.testing.assert_frame_equal(out.xs(t_last, level=0), expected_last, check_dtype=False)


    def test_decay_matches_half_life():
        df = make_df()
        half_life = 3  # every 3 steps halves
        steps = 7
        out = exp_scale_df(df, steps=steps, half_life=half_life)
        # After exactly 'half_life' steps, factor should be 0.5
        expected = df * 0.5
        pd.testing.assert_frame_equal(out.xs(half_life, level=0), expected, check_dtype=False)


    def test_growth_matches_doubling_time():
        df = make_df()
        doubling_time = 4
        steps = 5
        out = exp_scale_df(df, steps=steps, doubling_time=doubling_time)
        # After exactly 'doubling_time' steps, factor should be 2.0
        expected = df * 2.0
        pd.testing.assert_frame_equal(out.xs(doubling_time, level=0), expected, check_dtype=False)


    def test_multiple_columns_scaled_identically():
        df = make_df()
        steps = 3
        rate = 0.1
        out = exp_scale_df(df, steps=steps, rate=rate)
        # ratio of step 2 to step 1 equals exp(rate) for all cells
        step1 = out.xs(1, level=0)
        step2 = out.xs(2, level=0)
        ratio = (step2 / step1).to_numpy()
        assert np.allclose(ratio, np.exp(rate))


    def test_preserves_column_order_and_names():
        df = make_df()
        out = exp_scale_df(df, steps=2, rate=-0.02)
        assert list(out.columns) == ["SAT", "DUL", "LL15"]
