"""Data loading helpers with API fallbacks for the bond simulator."""

import os
import re
from datetime import date, datetime, timedelta
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def _get_secret(name):
    """Read a value from Streamlit secrets or environment variables."""
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def get_fred_api_key():
    """Return the FRED API key if configured."""
    return _get_secret("FRED_API_KEY")


def get_bok_api_key():
    """Return the Bank of Korea ECOS API key if configured."""
    return _get_secret("BOK_API_KEY")


def _sample_macro_rates():
    """Return sample Korean macro rate indicators."""
    return pd.DataFrame(
        [
            {"indicator": "한국 기준금리", "rate": 3.50, "date": "2026-05-26", "source": "sample"},
            {"indicator": "국고채 1년", "rate": 3.24, "date": "2026-05-26", "source": "sample"},
            {"indicator": "국고채 3년", "rate": 3.28, "date": "2026-05-26", "source": "sample"},
            {"indicator": "국고채 5년", "rate": 3.35, "date": "2026-05-26", "source": "sample"},
            {"indicator": "국고채 10년", "rate": 3.48, "date": "2026-05-26", "source": "sample"},
            {"indicator": "회사채 AA-", "rate": 4.05, "date": "2026-05-26", "source": "sample"},
            {"indicator": "CD 91일물", "rate": 3.62, "date": "2026-05-26", "source": "sample"},
        ]
    )


def _sample_kr_yield_history():
    """Return sample Korean 3Y and 10Y government bond yield history."""
    end = pd.Timestamp(date.today())
    dates = pd.date_range(end=end, periods=180, freq="B")
    x = np.linspace(0, 1, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "국고채 3년": 3.20 + 0.16 * np.sin(x * 7) + 0.08 * x,
            "국고채 10년": 3.38 + 0.14 * np.cos(x * 6) + 0.06 * x,
            "source": "sample",
        }
    )


def _sample_kr_yield_curve():
    """Return sample Korean government bond yield curve data."""
    return pd.DataFrame(
        [
            (0.5, "6개월", 3.18, "sample", pd.Timestamp("2026-05-26"), None),
            (1, "1년", 3.24, "sample", pd.Timestamp("2026-05-26"), None),
            (2, "2년", 3.26, "sample", pd.Timestamp("2026-05-26"), None),
            (3, "3년", 3.28, "sample", pd.Timestamp("2026-05-26"), None),
            (5, "5년", 3.35, "sample", pd.Timestamp("2026-05-26"), None),
            (10, "10년", 3.48, "sample", pd.Timestamp("2026-05-26"), None),
            (20, "20년", 3.55, "sample", pd.Timestamp("2026-05-26"), None),
            (30, "30년", 3.50, "sample", pd.Timestamp("2026-05-26"), None),
        ],
        columns=["maturity_years", "label", "yield", "source", "date", "item_code"],
    )


def _sample_exchange_rate():
    """Return sample USD/KRW exchange-rate data."""
    return pd.DataFrame(
        [
            {
                "currency": "USD/KRW",
                "rate": 1365.0,
                "date": pd.Timestamp("2026-05-26"),
                "source": "sample",
                "stat_code": "731Y001",
                "item_code": "0000001",
            }
        ]
    )


def _fetch_bok_json(service, start_count=1, end_count=100, extra_path=""):
    """Fetch one ECOS API JSON response and return the payload dict."""
    api_key = get_bok_api_key()
    if not api_key:
        return {}
    url = f"https://ecos.bok.or.kr/api/{service}/{api_key}/json/kr/{start_count}/{end_count}/{extra_path}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if service in data and "row" in data[service]:
            return data
        return {}
    except Exception:
        return {}


def fetch_bok_key_statistics():
    """Fetch ECOS key statistics and return a DataFrame."""
    data = _fetch_bok_json("KeyStatisticList", 1, 100)
    rows = data.get("KeyStatisticList", {}).get("row", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "DATA_VALUE" in df.columns:
        df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    if "CYCLE" in df.columns:
        df["date"] = pd.to_datetime(df["CYCLE"], errors="coerce")
    return df


def fetch_bok_series(stat_code, cycle, start, end, item_code, limit=1000):
    """Fetch one ECOS StatisticSearch series."""
    extra_path = f"{stat_code}/{cycle}/{start}/{end}/{item_code}/"
    data = _fetch_bok_json("StatisticSearch", 1, limit, extra_path)
    rows = data.get("StatisticSearch", {}).get("row", [])
    if not rows:
        return pd.DataFrame(columns=["date", "value", "item_name", "item_code", "source"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["TIME"], errors="coerce")
    df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    df["item_name"] = df.get("ITEM_NAME1", "")
    df["item_code"] = df.get("ITEM_CODE1", item_code)
    df["source"] = "BOK ECOS"
    return df[["date", "value", "item_name", "item_code", "source"]].dropna(subset=["date", "value"])


def _latest_bok_series_value(stat_code, cycle, item_code, days_back=370):
    """Return the latest value from one ECOS series."""
    end = date.today()
    start = end - timedelta(days=days_back)
    df = fetch_bok_series(stat_code, cycle, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), item_code)
    if df.empty:
        return None
    row = df.sort_values("date").iloc[-1]
    return {"rate": float(row["value"]), "date": row["date"], "source": row["source"]}


def fetch_fred_series(series_id, observation_start=None, observation_end=None, limit=None):
    """Fetch one FRED series with requests and return date/value columns."""
    api_key = get_fred_api_key()
    if not api_key:
        return pd.DataFrame(columns=["date", "value"])

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc" if limit else "asc",
    }
    if observation_start:
        params["observation_start"] = observation_start
    if observation_end:
        params["observation_end"] = observation_end
    if limit:
        params["limit"] = limit

    try:
        response = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params=params,
            timeout=8,
        )
        response.raise_for_status()
        observations = response.json().get("observations", [])
        df = pd.DataFrame(observations)
        if df.empty:
            return pd.DataFrame(columns=["date", "value"])
        df = df[["date", "value"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"].replace({".": np.nan, "": np.nan}), errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date")
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date", "value"])


def load_macro_rates():
    """Load Korean macro rate indicators from ECOS or return sample data."""
    indicator_map = {
        "한국은행 기준금리": "한국 기준금리",
        "CD수익률(91일)": "CD 91일물",
        "국고채수익률(3년)": "국고채 3년",
        "국고채수익률(5년)": "국고채 5년",
        "회사채수익률(3년,AA-)": "회사채 AA-",
    }
    market_codes = {
        "국고채 1년": ("817Y002", "D", "010190000"),
        "국고채 10년": ("817Y002", "D", "010210000"),
    }

    rows = []
    key_stats = fetch_bok_key_statistics()
    if not key_stats.empty:
        for key_name, indicator in indicator_map.items():
            matched = key_stats[key_stats["KEYSTAT_NAME"] == key_name]
            if not matched.empty:
                item = matched.iloc[-1]
                rows.append(
                    {
                        "indicator": indicator,
                        "rate": float(item["DATA_VALUE"]),
                        "date": item.get("CYCLE"),
                        "source": "BOK ECOS",
                    }
                )

    for indicator, (stat_code, cycle, item_code) in market_codes.items():
        latest = _latest_bok_series_value(stat_code, cycle, item_code)
        if latest:
            rows.append({"indicator": indicator, **latest})

    result = pd.DataFrame(rows)
    required_order = ["한국 기준금리", "국고채 1년", "국고채 3년", "국고채 5년", "국고채 10년", "회사채 AA-", "CD 91일물"]
    if result.empty or len(set(result["indicator"])) < 4:
        return _sample_macro_rates()

    sample = _sample_macro_rates().set_index("indicator")
    result = result.drop_duplicates("indicator", keep="last").set_index("indicator")
    for indicator in required_order:
        if indicator not in result.index:
            result.loc[indicator] = sample.loc[indicator]
    return result.loc[required_order].reset_index()


def load_policy_rates():
    """Load Korean and US policy rates, using sample data when APIs fail."""
    bok_latest = _latest_bok_series_value("722Y001", "D", "0101000")
    rows = [
        {
            "country": "한국",
            "policy_rate": bok_latest["rate"] if bok_latest else 3.50,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "date": bok_latest["date"] if bok_latest else pd.Timestamp("2026-05-26"),
            "source": bok_latest["source"] if bok_latest else "sample",
        }
    ]

    upper = fetch_fred_series("DFEDTARU", limit=1)
    lower = fetch_fred_series("DFEDTARL", limit=1)
    effective = fetch_fred_series("FEDFUNDS", limit=1)

    if not upper.empty and not lower.empty:
        lower_value = float(lower.iloc[-1]["value"])
        upper_value = float(upper.iloc[-1]["value"])
        rows.append(
            {
                "country": "미국",
                "policy_rate": (lower_value + upper_value) / 2,
                "lower_bound": lower_value,
                "upper_bound": upper_value,
                "date": max(lower.iloc[-1]["date"], upper.iloc[-1]["date"]),
                "source": "FRED",
            }
        )
    elif not effective.empty:
        rows.append(
            {
                "country": "미국",
                "policy_rate": float(effective.iloc[-1]["value"]),
                "lower_bound": np.nan,
                "upper_bound": np.nan,
                "date": effective.iloc[-1]["date"],
                "source": "FRED",
            }
        )
    else:
        rows.append(
            {
                "country": "미국",
                "policy_rate": 4.50,
                "lower_bound": 4.25,
                "upper_bound": 4.50,
                "date": pd.Timestamp("2026-05-26"),
                "source": "sample",
            }
        )
    return pd.DataFrame(rows)


def _sample_policy_calendar():
    """Return editable fallback policy meeting schedules."""
    today = pd.Timestamp(date.today()).normalize()
    bok_dates = [
        "2026-05-28",
        "2026-07-09",
        "2026-08-27",
        "2026-10-15",
        "2026-11-26",
    ]
    fomc_dates = [
        "2026-06-17",
        "2026-07-29",
        "2026-09-16",
        "2026-11-04",
        "2026-12-16",
    ]

    rows = []
    for event, dates in [("한국은행 금통위", bok_dates), ("FOMC", fomc_dates)]:
        parsed = [pd.Timestamp(item) for item in dates]
        future_dates = [item for item in parsed if item >= today]
        next_date = min(future_dates) if future_dates else max(parsed)
        rows.append(
            {
                "event": event,
                "next_date": next_date.date(),
                "d_day": int((next_date - today).days),
                "source": "sample",
            }
        )
    return pd.DataFrame(rows)


def _parse_bok_meeting_date(text, year):
    """Parse BOK meeting text such as '01월 15일(목)' into a date."""
    match = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", str(text))
    if not match:
        return None
    try:
        return pd.Timestamp(year=int(year), month=int(match.group(1)), day=int(match.group(2)))
    except Exception:
        return None


def fetch_bok_mpc_calendar(year=None):
    """Scrape the BOK official MPC meeting schedule for one year."""
    target_year = int(year or date.today().year)
    url = (
        "https://www.bok.or.kr/portal/singl/crncyPolicyDrcMtg/listYear.do"
        f"?mtgSe=A&menuNo=200755&pYear={target_year}"
    )
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        table = soup.select_one("table#tableId")
        if table is None:
            return pd.DataFrame(columns=["event", "meeting_date", "source", "url"])

        rows = []
        for cell in table.select('tbody tr th[scope="row"]'):
            meeting_date = _parse_bok_meeting_date(cell.get_text(" ", strip=True), target_year)
            if meeting_date is not None:
                rows.append(
                    {
                        "event": "한국은행 금통위",
                        "meeting_date": meeting_date,
                        "source": "BOK official",
                        "url": url,
                    }
                )
        return pd.DataFrame(rows).drop_duplicates("meeting_date").sort_values("meeting_date")
    except Exception:
        return pd.DataFrame(columns=["event", "meeting_date", "source", "url"])


_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _extract_fomc_dates_from_text(text, year):
    """Extract likely FOMC meeting dates from a text block."""
    dates = []
    month_names = "|".join(_MONTH_MAP.keys())
    pattern = re.compile(
        rf"\b({month_names})\s+(\d{{1,2}})(?:\s*[-–]\s*(\d{{1,2}}))?(?:,\s*(\d{{4}}))?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(str(text)):
        parsed_year = int(match.group(4) or year)
        if parsed_year != int(year):
            continue
        month = _MONTH_MAP[match.group(1).lower()]
        day = int(match.group(3) or match.group(2))
        try:
            dates.append(pd.Timestamp(year=parsed_year, month=month, day=day))
        except Exception:
            continue
    return dates


def _parse_fomc_meeting_date(month_text, day_text, year):
    """Parse one Fed calendar meeting month/date pair."""
    clean_month = str(month_text).strip().split("/")[-1].lower()
    month = _MONTH_MAP.get(clean_month)
    if not month:
        return None
    numbers = [int(item) for item in re.findall(r"\d{1,2}", str(day_text))]
    if not numbers:
        return None
    day = numbers[-1]
    try:
        return pd.Timestamp(year=int(year), month=month, day=day)
    except Exception:
        return None


def fetch_fomc_calendar(year=None):
    """Scrape the Federal Reserve FOMC calendar for one year."""
    target_year = int(year or date.today().year)
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        dates = []
        for panel in soup.select(".panel.panel-default"):
            heading = panel.find(["h2", "h3", "h4"])
            if heading is None or str(target_year) not in heading.get_text(" ", strip=True):
                continue
            for meeting in panel.select(".fomc-meeting"):
                month_node = meeting.select_one(".fomc-meeting__month")
                date_node = meeting.select_one(".fomc-meeting__date")
                meeting_date = _parse_fomc_meeting_date(
                    month_node.get_text(" ", strip=True) if month_node else "",
                    date_node.get_text(" ", strip=True) if date_node else "",
                    target_year,
                )
                if meeting_date is not None:
                    dates.append(meeting_date)

        if not dates:
            dates = sorted(set(_extract_fomc_dates_from_text(soup.get_text(" ", strip=True), target_year)))
        rows = [
            {
                "event": "FOMC",
                "meeting_date": item,
                "source": "Federal Reserve official",
                "url": url,
            }
            for item in dates
        ]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["event", "meeting_date", "source", "url"])


def _next_meeting_from_calendar(calendar_df, event_name, today):
    """Return a next-meeting row from a scraped calendar DataFrame."""
    if calendar_df is None or calendar_df.empty:
        return None
    dates = pd.to_datetime(calendar_df["meeting_date"], errors="coerce").dropna()
    future_dates = dates[dates >= today]
    if future_dates.empty:
        return None
    next_date = future_dates.min()
    source = calendar_df.iloc[0].get("source", "official")
    return {
        "event": event_name,
        "next_date": next_date.date(),
        "d_day": int((next_date.normalize() - today).days),
        "source": source,
    }


def load_policy_calendar():
    """Return the next BOK MPC and FOMC dates from official pages or fallback schedules."""
    today = pd.Timestamp(date.today()).normalize()
    fallback = _sample_policy_calendar()
    current_year = today.year
    next_year = current_year + 1

    bok_calendar = fetch_bok_mpc_calendar(current_year)
    bok_row = _next_meeting_from_calendar(bok_calendar, "한국은행 금통위", today)
    if bok_row is None:
        bok_calendar = fetch_bok_mpc_calendar(next_year)
        bok_row = _next_meeting_from_calendar(bok_calendar, "한국은행 금통위", today)

    fomc_calendar = fetch_fomc_calendar(current_year)
    fomc_row = _next_meeting_from_calendar(fomc_calendar, "FOMC", today)
    if fomc_row is None:
        fomc_calendar = fetch_fomc_calendar(next_year)
        fomc_row = _next_meeting_from_calendar(fomc_calendar, "FOMC", today)

    rows = []
    for event_name, scraped_row in [("한국은행 금통위", bok_row), ("FOMC", fomc_row)]:
        if scraped_row:
            rows.append(scraped_row)
        else:
            rows.append(fallback[fallback["event"] == event_name].iloc[0].to_dict())
    return pd.DataFrame(rows)


def _sample_central_bank_materials():
    """Return fallback official central bank material links."""
    return pd.DataFrame(
        [
            ("한국은행", "최근 금통위 결정문", "한국은행 통화정책방향", "https://www.bok.or.kr/portal/bbs/P0000559/list.do?menuNo=200690", "fallback"),
            ("한국은행", "총재 기자간담회", "한국은행 총재 기자간담회", "https://www.bok.or.kr/portal/bbs/P0002559/list.do?menuNo=201156", "fallback"),
            ("한국은행", "의사록", "한국은행 금융통화위원회 의사록", "https://www.bok.or.kr/portal/bbs/P0000245/list.do?menuNo=200761", "fallback"),
            ("한국은행", "금융·경제 이슈", "한국은행 이슈노트", "https://www.bok.or.kr/portal/bbs/P0002353/list.do?menuNo=200433", "fallback"),
            ("연준", "FOMC Statement", "FOMC Statements", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", "fallback"),
            ("연준", "FOMC Minutes", "FOMC Minutes", "https://www.federalreserve.gov/monetarypolicy/fomcminutes.htm", "fallback"),
            ("연준", "Press Conference", "FOMC Press Conferences", "https://www.federalreserve.gov/monetarypolicy/fomcpresconf.htm", "fallback"),
            ("연준", "Projection Materials", "FOMC Projection Materials", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", "fallback"),
        ],
        columns=["bank", "material", "title", "url", "source"],
    ).assign(status="fallback")


def _fetch_fed_material_links():
    """Fetch recent official FOMC material links from the calendar page."""
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    patterns = {
        "FOMC Statement": (r'href="([^"]*?/newsevents/pressreleases/monetary(\d{8})a\.htm)"', "FOMC Statement"),
        "FOMC Minutes": (r'href="([^"]*?/monetarypolicy/fomcminutes(\d{8})\.htm)"', "FOMC Minutes"),
        "Press Conference": (r'href="([^"]*?/monetarypolicy/fomcpresconf(\d{8})\.htm)"', "Press Conference"),
        "Projection Materials": (r'href="([^"]*?/monetarypolicy/fomcprojtabl(\d{8})\.htm)"', "Projection Materials"),
    }
    today = pd.Timestamp(date.today()).normalize()
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        html_text = response.text
        rows = []
        for material, (pattern, title_prefix) in patterns.items():
            candidates = []
            for href, yyyymmdd in re.findall(pattern, html_text):
                meeting_date = pd.to_datetime(yyyymmdd, format="%Y%m%d", errors="coerce")
                if pd.isna(meeting_date) or meeting_date > today:
                    continue
                candidates.append((meeting_date, urljoin(url, href)))
            if candidates:
                meeting_date, link = sorted(candidates, key=lambda item: item[0])[-1]
                rows.append(
                    {
                        "bank": "연준",
                        "material": material,
                        "title": f"{title_prefix} ({meeting_date.date()})",
                        "url": link,
                        "source": "Federal Reserve official",
                        "status": "official",
                    }
                )
    except Exception:
        return pd.DataFrame(columns=["bank", "material", "title", "url", "source", "status"])
    return pd.DataFrame(rows)


def _pick_bok_material_link(row, include_terms, exclude_terms=None):
    """Pick one meaningful BOK link from a meeting row."""
    exclude_terms = exclude_terms or []
    for anchor in row.select("a[href]"):
        text = anchor.get_text(" ", strip=True)
        href = anchor.get("href", "")
        if not text or "javascript" in href:
            continue
        if not all(term in text for term in include_terms):
            continue
        if any(term in text for term in exclude_terms):
            continue
        return text, href
    return None, None


def _fetch_bok_material_links():
    """Fetch latest official BOK MPC material links from the yearly meeting table."""
    today = pd.Timestamp(date.today()).normalize()
    target_year = today.year
    url = (
        "https://www.bok.or.kr/portal/singl/crncyPolicyDrcMtg/listYear.do"
        f"?mtgSe=A&menuNo=200755&pYear={target_year}"
    )
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        completed_rows = []
        for row in soup.select("table#tableId tbody tr"):
            date_cell = row.select_one('th[scope="row"]')
            meeting_date = _parse_bok_meeting_date(date_cell.get_text(" ", strip=True) if date_cell else "", target_year)
            if meeting_date is not None and meeting_date <= today:
                completed_rows.append((meeting_date, row))
        if not completed_rows:
            return pd.DataFrame(columns=["bank", "material", "title", "url", "source", "status"])

        pick_rules = [
            ("최근 금통위 결정문", ["국문보도자료", ".pdf"], []),
            ("총재 기자간담회", ["총재 기자간담회"], []),
            ("의사록", ["의사록", ".pdf"], []),
            ("금융·경제 이슈", ["금융·경제 이슈", ".pdf"], []),
        ]
        rows = []
        for material, include_terms, exclude_terms in pick_rules:
            for meeting_date, meeting_row in sorted(completed_rows, key=lambda item: item[0], reverse=True):
                title, href = _pick_bok_material_link(meeting_row, include_terms, exclude_terms)
                if title and href:
                    rows.append(
                        {
                            "bank": "한국은행",
                            "material": material,
                            "title": f"{title} ({meeting_date.date()})",
                            "url": urljoin(url, href),
                            "source": "BOK official",
                            "status": "official",
                        }
                    )
                    break
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["bank", "material", "title", "url", "source", "status"])


def load_central_bank_materials():
    """Load official BOK/Fed material links with stable fallbacks."""
    fallback = _sample_central_bank_materials()
    bok_links = _fetch_bok_material_links()
    fed_links = _fetch_fed_material_links()
    official_links = pd.concat([bok_links, fed_links], ignore_index=True)
    if official_links.empty:
        return fallback

    result = fallback.copy()
    for _, row in official_links.iterrows():
        result = result[~((result["bank"] == row["bank"]) & (result["material"] == row["material"]))]
    result = pd.concat([result, official_links], ignore_index=True)
    order = ["최근 금통위 결정문", "총재 기자간담회", "의사록", "금융·경제 이슈", "FOMC Statement", "FOMC Minutes", "Press Conference", "Projection Materials"]
    result["order"] = result["material"].apply(lambda value: order.index(value) if value in order else 99)
    return result.sort_values(["bank", "order"]).drop(columns="order").reset_index(drop=True)


def load_exchange_rate():
    """Load latest USD/KRW exchange rate from BOK ECOS or sample data."""
    latest = _latest_bok_series_value("731Y001", "D", "0000001", days_back=45)
    if latest:
        return pd.DataFrame(
            [
                {
                    "currency": "USD/KRW",
                    "rate": latest["rate"],
                    "date": latest["date"],
                    "source": latest["source"],
                    "stat_code": "731Y001",
                    "item_code": "0000001",
                }
            ]
        )
    return _sample_exchange_rate()


def load_bond_yield_history():
    """Return Korean 3Y and 10Y government bond yield history from ECOS or samples."""
    end = date.today()
    start = end - timedelta(days=270)
    start_text = start.strftime("%Y%m%d")
    end_text = end.strftime("%Y%m%d")
    series = {
        "국고채 3년": "010200000",
        "국고채 10년": "010210000",
    }
    frames = []
    for label, item_code in series.items():
        df = fetch_bok_series("817Y002", "D", start_text, end_text, item_code)
        if not df.empty:
            frames.append(df[["date", "value"]].rename(columns={"value": label}))
    if len(frames) == 2:
        result = frames[0].merge(frames[1], on="date", how="outer").sort_values("date")
        result["source"] = "BOK ECOS"
        return result.tail(180).reset_index(drop=True)
    return _sample_kr_yield_history()


def load_yield_curve():
    """Return Korean government bond yield curve from ECOS or sample data."""
    curve_items = [
        (1, "1년", "010190000"),
        (2, "2년", "010195000"),
        (3, "3년", "010200000"),
        (5, "5년", "010200001"),
        (10, "10년", "010210000"),
        (20, "20년", "010220000"),
        (30, "30년", "010230000"),
        (50, "50년", "010240000"),
    ]
    rows = []
    for maturity, label, item_code in curve_items:
        latest = _latest_bok_series_value("817Y002", "D", item_code, days_back=45)
        if latest:
            rows.append(
                {
                    "maturity_years": maturity,
                    "label": label,
                    "yield": latest["rate"],
                    "source": latest["source"],
                    "date": latest["date"],
                    "item_code": item_code,
                }
            )
    if len(rows) >= 5:
        return pd.DataFrame(rows).sort_values("maturity_years").reset_index(drop=True)
    return _sample_kr_yield_curve()


def _sample_us_treasury_curve():
    """Return sample US Treasury curve data."""
    return pd.DataFrame(
        [
            (1 / 12, "1개월", 4.38, "DGS1MO", pd.Timestamp("2026-05-26")),
            (0.25, "3개월", 4.35, "DGS3MO", pd.Timestamp("2026-05-26")),
            (0.5, "6개월", 4.28, "DGS6MO", pd.Timestamp("2026-05-26")),
            (1, "1년", 4.12, "DGS1", pd.Timestamp("2026-05-26")),
            (2, "2년", 3.96, "DGS2", pd.Timestamp("2026-05-26")),
            (3, "3년", 3.88, "DGS3", pd.Timestamp("2026-05-26")),
            (5, "5년", 3.92, "DGS5", pd.Timestamp("2026-05-26")),
            (7, "7년", 4.02, "DGS7", pd.Timestamp("2026-05-26")),
            (10, "10년", 4.10, "DGS10", pd.Timestamp("2026-05-26")),
            (20, "20년", 4.45, "DGS20", pd.Timestamp("2026-05-26")),
            (30, "30년", 4.36, "DGS30", pd.Timestamp("2026-05-26")),
        ],
        columns=["maturity_years", "label", "yield", "series_id", "date"],
    )


def load_us_treasury_curve():
    """Load the latest US Treasury yield curve from FRED or sample data."""
    series = [
        ("DGS1MO", 1 / 12, "1개월"),
        ("DGS3MO", 0.25, "3개월"),
        ("DGS6MO", 0.5, "6개월"),
        ("DGS1", 1, "1년"),
        ("DGS2", 2, "2년"),
        ("DGS3", 3, "3년"),
        ("DGS5", 5, "5년"),
        ("DGS7", 7, "7년"),
        ("DGS10", 10, "10년"),
        ("DGS20", 20, "20년"),
        ("DGS30", 30, "30년"),
    ]
    if not get_fred_api_key():
        df = _sample_us_treasury_curve()
        df["source"] = "sample"
        return df

    rows = []
    for series_id, maturity, label in series:
        df = fetch_fred_series(series_id, limit=10)
        if not df.empty:
            latest = df.iloc[-1]
            rows.append(
                {
                    "maturity_years": maturity,
                    "label": label,
                    "yield": float(latest["value"]),
                    "series_id": series_id,
                    "date": latest["date"],
                    "source": "FRED",
                }
            )
    if len(rows) < 4:
        df = _sample_us_treasury_curve()
        df["source"] = "sample"
        return df
    return pd.DataFrame(rows).sort_values("maturity_years").reset_index(drop=True)


def load_us_treasury_history():
    """Load recent 2Y, 10Y, and 30Y US Treasury yield history from FRED or samples."""
    start = (datetime.today() - timedelta(days=370)).strftime("%Y-%m-%d")
    frames = []
    for series_id in ["DGS2", "DGS10", "DGS30"]:
        df = fetch_fred_series(series_id, observation_start=start)
        if not df.empty:
            frames.append(df.rename(columns={"value": series_id}))

    if len(frames) == 3:
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.merge(df, on="date", how="outer")
        return merged.sort_values("date").tail(250).reset_index(drop=True)

    end = pd.Timestamp(date.today())
    dates = pd.date_range(end=end, periods=250, freq="B")
    x = np.linspace(0, 1, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "DGS2": 3.95 + 0.22 * np.sin(x * 8),
            "DGS10": 4.10 + 0.18 * np.cos(x * 6),
            "DGS30": 4.35 + 0.14 * np.sin(x * 5 + 1),
        }
    )
