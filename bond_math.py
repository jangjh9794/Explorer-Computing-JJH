"""Bond pricing and risk calculations for the Streamlit simulator."""

import numpy as np
import pandas as pd


def _to_decimal(rate):
    """Convert a percent-like rate to decimal if needed."""
    if rate is None:
        return 0.0
    rate = float(rate)
    return rate / 100 if abs(rate) > 1 else rate


def _coupon_to_decimal(rate):
    """Convert coupon rates entered in percent units to decimal."""
    if rate is None:
        return 0.0
    return float(rate) / 100


def _valid_positive(*values):
    """Return True when all values are positive numbers."""
    try:
        return all(float(value) > 0 for value in values)
    except (TypeError, ValueError):
        return False


def calculate_cashflows(face_value, coupon_rate, years_to_maturity, payments_per_year):
    """Return a cash flow table for a coupon bond."""
    if not _valid_positive(face_value, years_to_maturity, payments_per_year):
        return pd.DataFrame(columns=["period", "year", "cashflow"])

    face_value = float(face_value)
    coupon_rate = max(_coupon_to_decimal(coupon_rate), 0.0)
    years_to_maturity = float(years_to_maturity)
    payments_per_year = int(payments_per_year)
    total_periods = max(int(round(years_to_maturity * payments_per_year)), 1)
    coupon_payment = face_value * coupon_rate / payments_per_year

    rows = []
    for period in range(1, total_periods + 1):
        cashflow = coupon_payment
        if period == total_periods:
            cashflow += face_value
        rows.append(
            {
                "period": period,
                "year": period / payments_per_year,
                "cashflow": cashflow,
            }
        )
    return pd.DataFrame(rows)


def calculate_bond_price(face_value, coupon_rate, ytm, years_to_maturity, payments_per_year):
    """Calculate bond price from discounted cash flows using one market yield."""
    if not _valid_positive(face_value, years_to_maturity, payments_per_year):
        return 0.0

    ytm = _to_decimal(ytm)
    payments_per_year = int(payments_per_year)
    cashflows = calculate_cashflows(face_value, coupon_rate, years_to_maturity, payments_per_year)
    if cashflows.empty:
        return 0.0

    periodic_rate = ytm / payments_per_year
    if periodic_rate <= -1:
        return 0.0
    discounts = (1 + periodic_rate) ** cashflows["period"]
    return float((cashflows["cashflow"] / discounts).sum())


def calculate_macaulay_duration(face_value, coupon_rate, ytm, years_to_maturity, payments_per_year):
    """Calculate Macaulay duration in years."""
    if not _valid_positive(face_value, years_to_maturity, payments_per_year):
        return 0.0

    price = calculate_bond_price(face_value, coupon_rate, ytm, years_to_maturity, payments_per_year)
    if price <= 0:
        return 0.0

    ytm = _to_decimal(ytm)
    payments_per_year = int(payments_per_year)
    cashflows = calculate_cashflows(face_value, coupon_rate, years_to_maturity, payments_per_year)
    periodic_rate = ytm / payments_per_year
    if periodic_rate <= -1:
        return 0.0
    pv = cashflows["cashflow"] / ((1 + periodic_rate) ** cashflows["period"])
    weighted_time = (cashflows["year"] * pv).sum()
    return float(weighted_time / price)


def calculate_modified_duration(face_value, coupon_rate, ytm, years_to_maturity, payments_per_year):
    """Calculate modified duration in years."""
    if not _valid_positive(face_value, years_to_maturity, payments_per_year):
        return 0.0

    ytm = _to_decimal(ytm)
    payments_per_year = int(payments_per_year)
    denominator = 1 + ytm / payments_per_year
    if denominator <= 0:
        return 0.0
    macaulay = calculate_macaulay_duration(
        face_value, coupon_rate, ytm, years_to_maturity, payments_per_year
    )
    return float(macaulay / denominator)


def calculate_convexity(face_value, coupon_rate, ytm, years_to_maturity, payments_per_year):
    """Calculate discrete-compounding bond convexity."""
    if not _valid_positive(face_value, years_to_maturity, payments_per_year):
        return 0.0

    price = calculate_bond_price(face_value, coupon_rate, ytm, years_to_maturity, payments_per_year)
    if price <= 0:
        return 0.0

    ytm = _to_decimal(ytm)
    payments_per_year = int(payments_per_year)
    cashflows = calculate_cashflows(face_value, coupon_rate, years_to_maturity, payments_per_year)
    periodic_rate = ytm / payments_per_year
    if periodic_rate <= -1:
        return 0.0
    periods = cashflows["period"]
    numerator = (cashflows["cashflow"] * periods * (periods + 1) / ((1 + periodic_rate) ** (periods + 2))).sum()
    return float(numerator / (price * payments_per_year**2))


def calculate_total_value(price, quantity):
    """Calculate total position value."""
    try:
        return float(price) * max(float(quantity), 0.0)
    except (TypeError, ValueError):
        return 0.0


def calculate_expected_coupon_income(face_value, coupon_rate, years_to_maturity, quantity):
    """Calculate expected coupon income until maturity."""
    if not _valid_positive(face_value, years_to_maturity, quantity):
        return 0.0
    return float(face_value) * max(_coupon_to_decimal(coupon_rate), 0.0) * float(years_to_maturity) * float(quantity)


def simulate_rate_change(
    face_value,
    coupon_rate,
    current_ytm,
    rate_change_bp,
    years_to_maturity,
    payments_per_year,
    quantity,
):
    """Reprice a bond after a basis-point yield change."""
    current_ytm = _to_decimal(current_ytm)
    changed_ytm = current_ytm + float(rate_change_bp) / 10000
    current_price = calculate_bond_price(
        face_value, coupon_rate, current_ytm, years_to_maturity, payments_per_year
    )
    changed_price = calculate_bond_price(
        face_value, coupon_rate, changed_ytm, years_to_maturity, payments_per_year
    )
    price_change = changed_price - current_price
    price_change_pct = price_change / current_price * 100 if current_price else 0.0
    pnl = price_change * max(float(quantity), 0.0)
    return {
        "current_ytm": current_ytm,
        "changed_ytm": changed_ytm,
        "current_price": current_price,
        "changed_price": changed_price,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "pnl": pnl,
    }


def calculate_zero_coupon_price(face_value, ytm, years_to_maturity):
    """Calculate zero-coupon bond price."""
    if not _valid_positive(face_value, years_to_maturity):
        return 0.0
    ytm = _to_decimal(ytm)
    if 1 + ytm <= 0:
        return 0.0
    return float(face_value) / ((1 + ytm) ** float(years_to_maturity))


def calculate_discount_factor(rate, period):
    """Calculate a discount factor for one spot rate and period."""
    rate = _to_decimal(rate)
    if period < 0 or 1 + rate <= 0:
        return 0.0
    return 1 / ((1 + rate) ** float(period))


def calculate_forward_rate(spot_rate_n, spot_rate_nt, n, t):
    """Calculate forward rate from two spot rates."""
    spot_rate_n = _to_decimal(spot_rate_n)
    spot_rate_nt = _to_decimal(spot_rate_nt)
    if n < 0 or t <= 0 or 1 + spot_rate_n <= 0 or 1 + spot_rate_nt <= 0:
        return np.nan
    return float(((1 + spot_rate_nt) ** (n + t) / (1 + spot_rate_n) ** n) ** (1 / t) - 1)


def calculate_forward_rates_from_curve(yield_curve_df):
    """Return major forward rates from a yield curve DataFrame."""
    if yield_curve_df is None or yield_curve_df.empty:
        return pd.DataFrame(columns=["start_year", "forward_period", "end_year", "forward_rate"])

    curve = yield_curve_df[["maturity_years", "yield"]].dropna().copy()
    if curve.empty:
        return pd.DataFrame(columns=["start_year", "forward_period", "end_year", "forward_rate"])
    curve["yield"] = curve["yield"].apply(_to_decimal)
    curve = curve.sort_values("maturity_years")

    desired = [(0.5, 0.5), (1, 1), (3, 2), (5, 5), (10, 10)]
    maturities = curve["maturity_years"].to_numpy(dtype=float)
    rates = curve["yield"].to_numpy(dtype=float)
    rows = []
    for start, period in desired:
        end = start + period
        if start < maturities.min() or end > maturities.max():
            continue
        start_rate = float(np.interp(start, maturities, rates))
        end_rate = float(np.interp(end, maturities, rates))
        forward = calculate_forward_rate(start_rate, end_rate, start, period)
        if not np.isnan(forward):
            rows.append(
                {
                    "start_year": start,
                    "forward_period": period,
                    "end_year": end,
                    "forward_rate": forward * 100,
                }
            )
    return pd.DataFrame(rows)


def price_bond_with_yield_curve(face_value, coupon_rate, years_to_maturity, payments_per_year, yield_curve_df):
    """Price a bond by discounting each cash flow with interpolated spot rates."""
    if not _valid_positive(face_value, years_to_maturity, payments_per_year):
        return 0.0
    cashflows = calculate_cashflows(face_value, coupon_rate, years_to_maturity, payments_per_year)
    if cashflows.empty or yield_curve_df is None or yield_curve_df.empty:
        return 0.0

    curve = yield_curve_df[["maturity_years", "yield"]].dropna().copy()
    if curve.empty:
        return 0.0
    curve["yield"] = curve["yield"].apply(_to_decimal)
    curve = curve.sort_values("maturity_years")
    maturities = curve["maturity_years"].to_numpy(dtype=float)
    rates = curve["yield"].to_numpy(dtype=float)

    price = 0.0
    for _, row in cashflows.iterrows():
        year = float(row["year"])
        if maturities.min() <= year <= maturities.max():
            spot_rate = float(np.interp(year, maturities, rates))
        else:
            nearest_index = int(np.argmin(np.abs(maturities - year)))
            spot_rate = float(rates[nearest_index])
        price += float(row["cashflow"]) * calculate_discount_factor(spot_rate, year)
    return float(price)
