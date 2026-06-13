"""Streamlit app for a personal bond price analysis simulator."""

import html
from datetime import date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bond_math import (
    calculate_bond_price,
    calculate_cashflows,
    calculate_convexity,
    calculate_expected_coupon_income,
    calculate_forward_rate,
    calculate_forward_rates_from_curve,
    calculate_macaulay_duration,
    calculate_modified_duration,
    calculate_total_value,
    calculate_zero_coupon_price,
    price_bond_with_yield_curve,
)
from data_loader import (
    load_central_bank_materials,
    load_bond_yield_history,
    load_macro_rates,
    load_policy_calendar,
    load_policy_rates,
    load_exchange_rate,
    load_us_treasury_curve,
    load_us_treasury_history,
    load_yield_curve,
)
from news_crawler import fetch_bond_news, fetch_policy_rate_news
from rate_signal import classify_news_dataframe, summarize_news_flow
from visual_assets import asset_img_html, country_label_html


st.set_page_config(page_title="개인 채권 가격 분석 시뮬레이터", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --app-blue: #1552c7;
        --app-blue-strong: #0f4fd3;
        --app-blue-soft: #eaf2ff;
        --app-blue-border: #cbdaf5;
        --app-blue-ring: #dbeafe;
        --app-text: #0f172a;
        --app-muted: #64748b;
        --app-border: #dbe3ef;
        --app-card: #fbfdff;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1240px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbff 0%, #f8fafc 58%, #ffffff 100%);
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #334155;
        font-weight: 600;
    }
    html, body, [class*="css"] {
        font-family: "Pretendard", "Noto Sans KR", "Inter", "Segoe UI", sans-serif;
    }
    h1 {
        font-size: 1.65rem !important;
        letter-spacing: 0 !important;
        margin-bottom: 0.15rem !important;
        color: #0f172a !important;
    }
    h2, h3 {
        letter-spacing: 0 !important;
        color: #0f172a !important;
    }
    label, [data-testid="stWidgetLabel"] {
        color: #334155 !important;
        font-weight: 750 !important;
    }
    .section-title {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #1552c7;
        font-size: 1.3rem;
        font-weight: 800;
        margin: 0 0 14px 0;
        padding: 2px 0 10px 10px;
        border-bottom: 2px solid #dbeafe;
        border-left: 4px solid #1552c7;
        letter-spacing: 0;
    }
    .page-hero {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 52%, #eef5ff 100%);
        border: 1px solid #dbe3ef;
        border-radius: 14px;
        padding: 20px 22px;
        margin: 0 0 18px 0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
        position: relative;
        overflow: hidden;
    }
    .page-hero::before {
        content: "";
        position: absolute;
        left: 0;
        top: 18px;
        bottom: 18px;
        width: 5px;
        background: #1552c7;
        border-radius: 0 999px 999px 0;
    }
    .page-hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: #1552c7;
        background: #eaf2ff;
        border: 1px solid #cbdaf5;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 850;
        line-height: 1;
        padding: 7px 10px;
        margin-bottom: 10px;
    }
    .page-hero-title {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 900;
        line-height: 1.12;
        letter-spacing: 0;
        margin: 0;
    }
    .page-hero-desc {
        color: #64748b;
        font-size: 0.94rem;
        font-weight: 650;
        line-height: 1.45;
        margin-top: 8px;
        max-width: 820px;
    }
    .sidebar-brand {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 14px;
        padding: 15px 14px;
        margin-bottom: 18px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
    }
    .sidebar-brand-main {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sidebar-logo {
        width: 38px;
        height: 38px;
        border-radius: 11px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #1552c7;
        background: linear-gradient(135deg, #eaf2ff, #ffffff);
        border: 1px solid #cbdaf5;
        font-size: 1.35rem;
        font-weight: 900;
        line-height: 1;
        flex: 0 0 auto;
    }
    .sidebar-brand-title {
        color: #0f172a;
        font-size: 0.98rem;
        font-weight: 900;
        line-height: 1.22;
        letter-spacing: 0;
    }
    .sidebar-brand-sub {
        color: #64748b;
        font-size: 0.74rem;
        font-weight: 700;
        line-height: 1.35;
        margin-top: 10px;
    }
    .sidebar-section-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 850;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin: 8px 0 8px 2px;
        padding-left: 2px;
    }
    .sidebar-footer-card {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 12px;
        padding: 12px 12px;
        color: #64748b;
        font-size: 0.76rem;
        font-weight: 700;
        line-height: 1.4;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.035);
    }
    .subgroup-title {
        color: #334155;
        font-size: 0.86rem;
        font-weight: 800;
        margin: 8px 0 8px 0;
    }
    .small-note {
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.35;
    }
    .big-rate {
        color: #0f4fd3;
        font-size: 2rem;
        font-weight: 850;
        line-height: 1.1;
        margin: 4px 0 4px 0;
    }
    .section-card {
        background: #fbfdff;
        border: 1px solid #dbe3ef;
        border-radius: 9px;
        min-height: 142px;
        padding: 16px 16px 20px 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.035);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .card-label {
        color: #475569;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .title-row {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
    }
    .asset-icon {
        display: inline-block;
        width: 22px;
        height: 22px;
        object-fit: cover;
        border-radius: 50%;
        box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.08);
        flex: 0 0 auto;
        vertical-align: -5px;
    }
    .section-title .asset-icon {
        width: 24px;
        height: 24px;
        vertical-align: -6px;
    }
    .policy-card-title {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #475569;
        font-size: 0.92rem;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .mini-metric-card {
        background: #fbfdff;
        border: 1px solid #dbe3ef;
        border-radius: 9px;
        min-height: 82px;
        padding: 10px 9px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.035);
    }
    .mini-metric-label {
        display: flex;
        align-items: center;
        gap: 7px;
        color: #475569;
        font-size: 0.82rem;
        font-weight: 800;
        line-height: 1.18;
        margin-bottom: 7px;
    }
    .mini-metric-value {
        color: #0f4fd3;
        font-size: 1.18rem;
        font-weight: 850;
        line-height: 1.16;
        word-break: keep-all;
    }
    .rate-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(92px, 1fr));
        gap: 8px;
        width: 100%;
        margin-bottom: 14px;
    }
    .rate-card {
        background: #fbfdff;
        border: 1px solid #dbe3ef;
        border-radius: 9px;
        min-height: 78px;
        padding: 10px 7px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.035);
        overflow: visible;
    }
    .rate-card-label {
        color: #475569;
        font-size: 0.8rem;
        font-weight: 800;
        line-height: 1.15;
        text-align: center;
        white-space: normal;
        word-break: keep-all;
    }
    .rate-card-value {
        color: #0f4fd3;
        font-size: clamp(0.98rem, 1.1vw, 1.2rem);
        font-weight: 850;
        line-height: 1.15;
        margin-top: 8px;
        text-align: center;
        white-space: nowrap;
        letter-spacing: 0;
    }
    .official-material-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 10px;
        margin: 8px 0 18px 0;
    }
    .official-material-card {
        background: #fbfdff;
        border: 1px solid #dbe3ef;
        border-radius: 9px;
        padding: 13px 14px;
        min-height: 98px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.035);
    }
    .official-material-label {
        color: #334155;
        font-size: 0.88rem;
        font-weight: 850;
        margin-bottom: 8px;
    }
    .official-material-title {
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.35;
        min-height: 28px;
        margin-bottom: 10px;
    }
    .official-material-card a {
        color: #0f4fd3;
        font-weight: 850;
        text-decoration: none;
    }
    .issue-note-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 10px;
        margin: 8px 0 14px 0;
    }
    .issue-note-card {
        background: #f8fbff;
        border: 1px solid #dbe3ef;
        border-radius: 9px;
        padding: 12px 13px;
        color: #334155;
        font-size: 0.86rem;
        line-height: 1.45;
    }
    .section-bottom-spacer {
        height: 12px;
    }
    .policy-event-line {
        display: flex;
        align-items: baseline;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    .policy-event-date {
        color: #0f172a;
        font-weight: 850;
    }
    .action-row {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        align-items: center;
        margin-top: 6px;
    }
    .soft-panel {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 12px;
        padding: 18px 18px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }
    .panel-title {
        color: #0f172a;
        font-size: 1.08rem;
        font-weight: 850;
        margin: 0 0 14px 0;
        letter-spacing: 0;
    }
    .metric-panel-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
        gap: 12px;
        margin-bottom: 22px;
    }
    .value-card {
        background: #fbfdff;
        border: 1px solid #dbe3ef;
        border-radius: 10px;
        min-height: 116px;
        padding: 16px 12px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .value-card-label {
        color: #334155;
        font-size: 0.82rem;
        font-weight: 800;
        line-height: 1.25;
    }
    .value-card-value {
        color: #0f172a;
        font-size: 1.52rem;
        font-weight: 850;
        line-height: 1.15;
        margin-top: 9px;
        white-space: nowrap;
    }
    .value-card-sub {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 650;
        line-height: 1.25;
        margin-top: 10px;
    }
    .value-card-sub.positive {
        color: #dc2626;
    }
    .value-card-sub.negative {
        color: #0f62d6;
    }
    .breakdown-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 14px 0;
    }
    .breakdown-card {
        background: #fbfdff;
        border: 1px solid #dbe3ef;
        border-radius: 10px;
        min-height: 92px;
        padding: 13px 12px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.035);
    }
    .breakdown-label {
        color: #475569;
        font-size: 0.8rem;
        font-weight: 850;
        line-height: 1.2;
    }
    .breakdown-value {
        color: #1552c7;
        font-size: clamp(1.05rem, 1.7vw, 1.38rem);
        font-weight: 900;
        line-height: 1.15;
        margin-top: 8px;
        white-space: normal;
        word-break: keep-all;
    }
    .breakdown-value.positive {
        color: #dc2626;
    }
    .breakdown-value.negative {
        color: #1552c7;
    }
    .indicator-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(116px, 1fr));
        border: 1px solid #dbe3ef;
        border-radius: 10px;
        overflow: hidden;
        background: #fbfdff;
    }
    .indicator-cell {
        min-height: 100px;
        padding: 14px 9px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        border-right: 1px solid #e5eaf3;
    }
    .indicator-cell:last-child {
        border-right: 0;
    }
    .indicator-label {
        color: #334155;
        font-size: 0.8rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .indicator-value {
        color: #0f172a;
        font-size: 1.34rem;
        font-weight: 850;
        line-height: 1.15;
        margin-top: 8px;
        white-space: nowrap;
    }
    .indicator-unit {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 650;
        margin-top: 5px;
    }
    .tiny-note {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 9px;
        color: #64748b;
        font-size: 0.76rem;
        line-height: 1.35;
        padding: 9px 11px;
        margin-top: 10px;
    }
    .help-details {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 9px;
        color: #64748b;
        font-size: 0.76rem;
        line-height: 1.35;
        padding: 9px 11px;
        margin-top: 10px;
        margin-bottom: 22px;
    }
    .help-details summary {
        cursor: pointer;
        list-style: none;
        font-weight: 750;
        color: #64748b;
        outline: none;
    }
    .help-details summary::-webkit-details-marker {
        display: none;
    }
    .help-details summary::marker {
        content: "";
    }
    .help-body {
        border-top: 1px solid #e2e8f0;
        margin-top: 8px;
        padding-top: 8px;
    }
    .help-row {
        margin: 4px 0;
    }
    .help-term {
        color: #334155;
        font-weight: 800;
    }
    .summary-score {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 360px;
        flex-direction: column;
        background: linear-gradient(180deg, #fbfdff 0%, #ffffff 100%);
        border: 1px solid #dbe3ef;
        border-radius: 18px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.055);
        padding: 18px 16px;
        overflow: visible;
    }
    .score-gauge-wrap {
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 2px;
    }
    .score-donut {
        width: min(230px, 100%);
        aspect-ratio: 1 / 1;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background:
            radial-gradient(circle at center, #ffffff 0 58%, transparent 59%),
            conic-gradient(from -110deg, #1552c7 0 var(--score-pct), #e2e8f0 var(--score-pct) 100%);
        box-shadow:
            inset 0 0 0 1px rgba(219, 227, 239, 0.95),
            0 14px 30px rgba(21, 82, 199, 0.12);
        position: relative;
    }
    .score-donut::after {
        content: "";
        position: absolute;
        inset: 14px;
        border-radius: 50%;
        border: 10px solid rgba(234, 242, 255, 0.8);
        pointer-events: none;
    }
    .score-donut-inner {
        width: 62%;
        aspect-ratio: 1 / 1;
        border-radius: 50%;
        background: #ffffff;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px solid #dbe3ef;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        z-index: 1;
    }
    .score-number {
        color: #0f172a;
        font-size: 2.3rem;
        font-weight: 900;
        line-height: 1;
    }
    .score-denom {
        color: #64748b;
        font-size: 0.86rem;
        font-weight: 800;
        margin-top: 4px;
    }
    .score-label {
        color: #1552c7;
        font-size: 1.28rem;
        font-weight: 900;
        margin-top: 18px;
        text-align: center;
        width: 100%;
    }
    .score-caption {
        color: #65758f;
        font-size: 0.84rem;
        font-weight: 650;
        line-height: 1.45;
        margin-top: 8px;
        text-align: center;
        width: 100%;
    }
    .score-detail {
        width: 100%;
        margin-top: 14px;
        padding: 12px 13px;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #475569;
        font-size: 0.78rem;
        font-weight: 650;
        line-height: 1.5;
    }
    .score-detail b {
        color: #0f172a;
        font-weight: 850;
    }
    .analysis-list {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
    }
    .analysis-card {
        background: #fbfdff;
        border: 1px solid #dbe3ef;
        border-radius: 12px;
        min-height: 118px;
        padding: 14px 14px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.035);
    }
    .analysis-icon {
        width: 30px;
        height: 30px;
        border-radius: 9px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #1552c7;
        background: #eaf2ff;
        font-weight: 850;
        margin-bottom: 8px;
    }
    .analysis-item-title {
        color: #1552c7;
        font-weight: 850;
        font-size: 0.9rem;
    }
    .analysis-item-body {
        color: #334155;
        font-size: 0.82rem;
        line-height: 1.35;
        margin-top: 5px;
    }
    .summary-kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
    }
    .summary-kpi {
        background: #fbfdff;
        border: 1px solid #dbe3ef;
        border-radius: 12px;
        min-height: 124px;
        padding: 14px 12px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .summary-kpi-label {
        color: #334155;
        font-size: 0.8rem;
        font-weight: 850;
    }
    .summary-kpi-value {
        color: #0f172a;
        font-size: clamp(1.12rem, 1.5vw, 1.48rem);
        font-weight: 900;
        line-height: 1.12;
        margin-top: 8px;
        white-space: normal;
        word-break: keep-all;
    }
    .summary-kpi-sub {
        color: #64748b;
        font-size: 0.76rem;
        font-weight: 700;
        margin-top: 9px;
    }
    .summary-kpi-sub.positive {
        color: #dc2626;
    }
    .summary-kpi-sub.negative {
        color: #0f62d6;
    }
    .summary-footnote {
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.35;
        margin-top: 14px;
        padding: 10px 12px;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .summary-verdict {
        width: 100%;
        margin-top: 14px;
        padding: 12px 13px;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #334155;
        font-size: 0.82rem;
        font-weight: 750;
        line-height: 1.45;
        text-align: center;
    }
    .sidebar-spacer {
        height: 34vh;
        min-height: 160px;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        border-color: #cbdaf5 !important;
        border-radius: 9px !important;
        background: #fbfdff !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover,
    div[data-baseweb="textarea"] > div:hover {
        border-color: #1552c7 !important;
    }
    div[data-baseweb="select"] svg,
    [data-testid="stSelectbox"] svg {
        color: #1552c7 !important;
        fill: #1552c7 !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background-color: #1552c7 !important;
        border-color: #1552c7 !important;
        box-shadow: 0 0 0 2px #dbeafe !important;
    }
    [data-testid="stSlider"] div {
        color: #334155;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within,
    [data-testid="stDateInput"] div[data-baseweb="input"] > div:focus-within {
        border-color: var(--app-blue) !important;
        box-shadow: 0 0 0 2px var(--app-blue-ring) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] {
        background: transparent !important;
        padding-top: 2px !important;
        padding-bottom: 8px !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div {
        background: transparent !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        min-height: 6px !important;
        max-height: 6px !important;
        border-radius: 999px !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
        border-radius: 999px !important;
    }
    [data-testid="stSlider"] [role="slider"]:focus,
    [data-testid="stSlider"] [role="slider"]:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 4px var(--app-blue-ring) !important;
    }
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {
        color: #334155 !important;
        font-weight: 700 !important;
    }
    [data-testid="stRadio"] [aria-checked="true"] span,
    [data-testid="stCheckbox"] [aria-checked="true"] span {
        border-color: #1552c7 !important;
        background-color: #1552c7 !important;
    }
    [data-testid="stRadio"] [aria-checked="true"] svg,
    [data-testid="stCheckbox"] [aria-checked="true"] svg,
    [data-testid="stCheckbox"] svg,
    [data-testid="stRadio"] svg {
        color: var(--app-blue) !important;
        fill: var(--app-blue) !important;
    }
    button[kind="primary"],
    button[kind="secondary"],
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 9px !important;
    }
    button[kind="primary"] {
        background: var(--app-blue) !important;
        border-color: var(--app-blue) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 14px rgba(21, 82, 199, 0.18) !important;
    }
    button[kind="primary"]:hover {
        background: #0f46b7 !important;
        border-color: #0f46b7 !important;
    }
    button[kind="secondary"] {
        color: var(--app-blue) !important;
        border-color: var(--app-blue-border) !important;
        background: #ffffff !important;
    }
    button[kind="secondary"]:hover {
        color: #0f46b7 !important;
        border-color: var(--app-blue) !important;
        background: #f7fbff !important;
    }
    button:focus:not(:active),
    button:focus-visible {
        border-color: var(--app-blue) !important;
        box-shadow: 0 0 0 3px var(--app-blue-ring) !important;
        outline: none !important;
    }
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus {
        color: var(--app-blue) !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #eaf2ff !important;
        color: #1552c7 !important;
        font-weight: 850 !important;
        border-left: 4px solid #1552c7 !important;
        box-shadow: inset 0 0 0 1px #cbdaf5 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent !important;
        border: 1px solid transparent !important;
        padding: 10px 11px !important;
        transition: all 0.16s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #f1f6ff !important;
        border-color: #dbeafe !important;
    }
    div[data-testid="stMetric"] {
        background: #fbfdff;
        border: 1px solid #dbe3ef;
        border-radius: 9px;
        min-height: 82px;
        padding: 10px 9px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.035);
    }
    div[data-testid="stMetricLabel"] {
        color: #475569;
        font-weight: 700;
        font-size: 0.82rem;
        line-height: 1.18;
        white-space: normal;
    }
    div[data-testid="stMetricValue"] {
        color: #0f4fd3;
        font-weight: 850;
        font-size: 1.18rem;
        line-height: 1.16;
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
        word-break: keep-all;
    }
    div[data-testid="stMetricDelta"] {
        color: #0f8a3b;
        font-weight: 750;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 9px;
        overflow: hidden;
        margin-bottom: 14px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: #dbe3ef !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.045);
        margin-bottom: 18px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] > div:last-child {
        margin-bottom: 16px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stColumn"] .section-card:last-child,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stColumn"] .mini-metric-card:last-child,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stColumn"] .value-card:last-child {
        margin-bottom: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=1800)
def cached_macro_rates():
    """Cache macro rate data."""
    return load_macro_rates()


@st.cache_data(ttl=1800)
def cached_policy_rates():
    """Cache policy rate data."""
    return load_policy_rates()


@st.cache_data(ttl=1800)
def cached_policy_calendar():
    """Cache policy calendar data."""
    return load_policy_calendar()


@st.cache_data(ttl=1800)
def cached_central_bank_materials():
    """Cache official central bank material links."""
    cache_version = "official-materials-v2"
    _ = cache_version
    return load_central_bank_materials()


@st.cache_data(ttl=1800)
def cached_exchange_rate():
    """Cache latest USD/KRW exchange rate."""
    return load_exchange_rate()


@st.cache_data(ttl=1800)
def cached_bond_yield_history():
    """Cache Korean bond yield history."""
    return load_bond_yield_history()


@st.cache_data(ttl=1800)
def cached_kr_curve():
    """Cache Korean yield curve."""
    return load_yield_curve()


@st.cache_data(ttl=1800)
def cached_us_curve():
    """Cache US Treasury curve."""
    return load_us_treasury_curve()


@st.cache_data(ttl=1800)
def cached_us_history():
    """Cache US Treasury yield history."""
    return load_us_treasury_history()


@st.cache_data(ttl=1800)
def cached_policy_news():
    """Cache policy-rate news."""
    return fetch_policy_rate_news(12)


@st.cache_data(ttl=1800)
def cached_bond_news():
    """Cache bond-market news."""
    return fetch_bond_news(12)


def fmt_money(value, unit):
    """Format money with thousands separators."""
    return f"{value:,.2f} {unit}"


def fmt_rate(value):
    """Format a rate value."""
    if pd.isna(value):
        return "데이터 없음"
    return f"{float(value):.2f}%"


def get_curve(country):
    """Return the yield curve for the selected country."""
    return cached_kr_curve() if country == "한국채" else cached_us_curve()


def get_currency(country):
    """Return currency label for selected country."""
    return "원" if country == "한국채" else "달러"


def make_news_links(df, include_classification=False):
    """Prepare a compact news table with plain titles and separate links."""
    if df is None or df.empty:
        return pd.DataFrame()
    table = df.copy()
    table["제목"] = table["title"].astype(str).str.replace(r"^\[|\]$", "", regex=True)
    columns = {
        "source": "출처",
        "date": "날짜",
        "query": "검색어",
        "region": "대상",
        "issue_tags": "이슈 태그",
        "primary_issue": "대표 이슈",
        "matched_keywords": "근거 표현",
        "link": "링크",
    }
    visible = ["제목", "source", "date", "query"]
    if include_classification:
        for column in ["region", "primary_issue", "issue_tags", "matched_keywords"]:
            if column in table.columns:
                visible.append(column)
    visible.append("link")
    return table[visible].rename(columns=columns)


def make_raw_display_df(df):
    """Return a Streamlit-friendly copy of raw data for debugging."""
    if df is None:
        return pd.DataFrame()
    display = df.copy()
    for column in display.columns:
        if display[column].dtype == "object":
            display[column] = display[column].apply(lambda value: ", ".join(value) if isinstance(value, list) else value)
        if pd.api.types.is_datetime64_any_dtype(display[column]) or display[column].dtype == "object":
            display[column] = display[column].astype(str)
    return display


def format_policy_value(row):
    """Format a policy-rate row for display."""
    if row["country"] == "미국" and not pd.isna(row.get("lower_bound")) and not pd.isna(row.get("upper_bound")):
        return f"{row['lower_bound']:.2f}% ~ {row['upper_bound']:.2f}%"
    return fmt_rate(row["policy_rate"])


def get_latest_exchange_rate_value():
    """Return latest USD/KRW exchange-rate row as a dict."""
    df = cached_exchange_rate()
    if df is None or df.empty:
        return {"rate": 1365.0, "date": pd.Timestamp(date.today()), "source": "sample"}
    row = df.iloc[-1]
    return {"rate": float(row["rate"]), "date": pd.to_datetime(row["date"]), "source": row.get("source", "sample")}


def fmt_fx_rate(value):
    """Format USD/KRW exchange rate."""
    if pd.isna(value):
        return "데이터 없음"
    return f"{float(value):,.1f} 원/USD"


def safe_pct_spread(long_rate, short_rate):
    """Format a percent-point spread."""
    if long_rate is None or short_rate is None or pd.isna(long_rate) or pd.isna(short_rate):
        return "데이터 없음"
    return f"{float(long_rate) - float(short_rate):.2f}%p"


def render_policy_card(column, country, policy_rates, calendar):
    """Render one large policy-rate card with its next meeting date."""
    row = policy_rates[policy_rates["country"] == country].iloc[0]
    event_name = "한국은행 금통위" if country == "한국" else "FOMC"
    event = calendar[calendar["event"] == event_name].iloc[0]
    event_date = pd.to_datetime(event["next_date"]).date()
    column.markdown(
        f"""
        <div class="section-card">
          <div class="policy-card-title">{country_label_html(country, f"{country} 정책금리")}</div>
          <div class="big-rate">{format_policy_value(row)}</div>
          <div class="small-note">금리 데이터: {row['source']} · {pd.to_datetime(row['date']).date()}</div>
          <div class="policy-event-line">
            <span class="card-label">다음 {event_name}</span>
            <span class="policy-event-date">{event_date} · D-{int(event['d_day'])}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_exchange_card(column, exchange_row):
    """Render a large exchange-rate card aligned with policy cards."""
    column.markdown(
        f"""
        <div class="section-card">
          <div class="policy-card-title">{asset_img_html("exchange", width=22, height=22)}<span>원/달러 환율</span></div>
          <div class="big-rate">{fmt_fx_rate(exchange_row["rate"])}</div>
          <div class="small-note">환율 데이터: {html.escape(str(exchange_row["source"]))} · {pd.to_datetime(exchange_row["date"]).date()}</div>
          <div class="policy-event-line">
            <span class="card-label">ECOS 통계</span>
            <span class="policy-event-date">731Y001 · 0000001</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title, icon_key=None):
    """Render a section title with an optional image asset."""
    icon = asset_img_html(icon_key) if icon_key else ""
    st.markdown(f'<div class="section-title">{icon}<span>{title}</span></div>', unsafe_allow_html=True)


def render_page_header(title, description, kicker="개인 채권 가격 분석", icon_key="chart"):
    """Render a decorated page title shared by all app pages."""
    icon = asset_img_html(icon_key, width=18, height=18) if icon_key else ""
    st.markdown(
        f"""
        <div class="page-hero">
          <div class="page-hero-kicker">{icon}<span>{html.escape(kicker)}</span></div>
          <div class="page-hero-title">{html.escape(title)}</div>
          <div class="page-hero-desc">{html.escape(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mini_metric(column, label, value, icon_key=None):
    """Render a metric-like card that can include image assets in its label."""
    icon = asset_img_html(icon_key, width=20, height=20) if icon_key else ""
    column.markdown(
        f"""
        <div class="mini-metric-card">
          <div class="mini-metric-label">{icon}<span>{label}</span></div>
          <div class="mini-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(items, columns=3):
    """Render a list of metric tuples in a compact grid."""
    cols = st.columns(columns)
    for idx, (label, value) in enumerate(items):
        cols[idx % columns].metric(label, value)


def render_rate_strip(items):
    """Render ordered rate data as a continuous row of small cards."""
    if not items:
        return
    cards = "".join(
        '<div class="rate-card">'
        f'<div class="rate-card-label">{html.escape(str(label))}</div>'
        f'<div class="rate-card-value">{html.escape(str(value))}</div>'
        "</div>"
        for label, value in items
    )
    st.markdown(f'<div class="rate-grid">{cards}</div>', unsafe_allow_html=True)


def render_forward_cards(forward_df):
    """Render forward rates as user-friendly metric cards."""
    if forward_df is None or forward_df.empty:
        st.info("계산 가능한 forward rate 데이터가 없습니다.")
        return
    labels = []
    for row in forward_df.itertuples():
        labels.append(
            (
                f"{row.start_year:g}년 후 {row.forward_period:g}년",
                fmt_rate(row.forward_rate),
            )
        )
    render_rate_strip(labels)


def prepare_curve_for_forward(curve_df):
    """Return sorted curve maturities and decimal rates for forward-rate calculation."""
    if curve_df is None or curve_df.empty:
        return np.array([]), np.array([])
    curve = curve_df[["maturity_years", "yield"]].dropna().sort_values("maturity_years")
    if curve.empty:
        return np.array([]), np.array([])
    maturities = curve["maturity_years"].to_numpy(dtype=float)
    rates = curve["yield"].to_numpy(dtype=float) / 100
    return maturities, rates


def interpolate_spot_rate(maturities, rates, year):
    """Interpolate a spot rate from a curve, returning decimal units."""
    if len(maturities) == 0 or year < maturities.min() or year > maturities.max():
        return np.nan
    return float(np.interp(year, maturities, rates))


def render_forward_calculator(kr_curve, us_curve):
    """Render an interactive forward-rate calculator for Korea or the US."""
    country = st.radio("시장 선택", ["한국 국고채", "미국채"], horizontal=True, key="home_forward_country")
    curve = kr_curve if country == "한국 국고채" else us_curve
    maturities, rates = prepare_curve_for_forward(curve)

    if len(maturities) < 2:
        st.info("forward rate를 계산할 수 있는 수익률곡선 데이터가 부족합니다.")
        return

    min_year = int(np.ceil(maturities.min()))
    max_year = int(np.floor(maturities.max()))
    if max_year - min_year < 1:
        st.info("1년 단위 forward rate를 계산할 수 있는 만기 구간이 부족합니다.")
        return

    control_cols = st.columns(2)
    start_default = min(max(min_year, 1), max_year - 1)
    with control_cols[0]:
        start_year = st.slider("시작 시점(년)", min_year, max_year - 1, start_default, step=1)
    with control_cols[1]:
        max_period = max_year - start_year
        forward_period = st.slider("Forward 기간(년)", 1, max_period, min(1, max_period), step=1)

    end_year = start_year + forward_period
    start_rate = interpolate_spot_rate(maturities, rates, start_year)
    end_rate = interpolate_spot_rate(maturities, rates, end_year)
    forward_rate = calculate_forward_rate(start_rate, end_rate, start_year, forward_period)

    if np.isnan(forward_rate):
        st.info("선택한 구간의 forward rate를 계산할 수 없습니다.")
        return

    result_cols = st.columns(4)
    render_mini_metric(result_cols[0], "시장", country)
    render_mini_metric(result_cols[1], f"{start_year}년 spot", fmt_rate(start_rate * 100))
    render_mini_metric(result_cols[2], f"{end_year}년 spot", fmt_rate(end_rate * 100))
    render_mini_metric(result_cols[3], f"{start_year}년 후 {forward_period}년", fmt_rate(forward_rate * 100))
    st.caption("수익률곡선의 중간 만기는 선형보간으로 계산합니다.")


def plot_line_card(title, fig, height=260, icon_key=None):
    """Render a chart with compact dashboard styling."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=14, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e8eef7")
    render_section_title(title, icon_key)
    st.plotly_chart(fig, width="stretch")


def render_news_flow_dashboard(news_flow):
    """Render policy-rate news as market issue checkpoints."""
    region_counts = news_flow["region_counts"]
    tag_counts = news_flow["tag_counts"]

    kr_count = int(region_counts.loc[region_counts["region"] == "한국", "count"].sum()) if not region_counts.empty else 0
    us_count = int(region_counts.loc[region_counts["region"] == "미국", "count"].sum()) if not region_counts.empty else 0
    global_count = int(region_counts.loc[region_counts["region"].isin(["글로벌", "한/미", "기타"]), "count"].sum()) if not region_counts.empty else 0
    top_issue = news_flow.get("top_issue", "데이터 없음")

    metric_cols = st.columns(4)
    render_mini_metric(metric_cols[0], "한국 관련 뉴스", f"{kr_count}건", "kr_flag")
    render_mini_metric(metric_cols[1], "미국 관련 뉴스", f"{us_count}건", "us_flag")
    render_mini_metric(metric_cols[2], "기타/글로벌", f"{global_count}건", "signal")
    render_mini_metric(metric_cols[3], "주요 이슈", top_issue, "news")

    chart_col1, chart_col2 = st.columns([0.9, 1.25])
    with chart_col1:
        render_panel_title("국가/지역별 뉴스 건수")
        if region_counts.empty:
            st.info("지역별로 집계할 뉴스 데이터가 없습니다.")
        else:
            region_fig = px.bar(
                region_counts.sort_values("count"),
                x="region",
                y="count",
                text="count",
                color_discrete_sequence=["#1552c7"],
                labels={"region": "대상", "count": "뉴스 수"},
            )
            region_fig.update_layout(
                height=280,
                margin=dict(l=8, r=8, t=10, b=8),
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False,
                font=dict(size=11),
            )
            region_fig.update_yaxes(gridcolor="#e8eef7", rangemode="tozero")
            region_fig.update_xaxes(showgrid=False)
            st.plotly_chart(region_fig, width="stretch")

    with chart_col2:
        render_panel_title("주요 이슈 태그")
        if tag_counts.empty:
            st.info("이슈 태그 데이터가 없습니다.")
        else:
            tag_fig = px.bar(
                tag_counts.head(8).sort_values("count"),
                x="count",
                y="issue_tag",
                orientation="h",
                text="count",
                color_discrete_sequence=["#1552c7"],
                labels={"count": "뉴스 수", "issue_tag": "이슈"},
            )
            tag_fig.update_layout(
                height=280,
                margin=dict(l=8, r=8, t=10, b=8),
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False,
                font=dict(size=11),
            )
            tag_fig.update_xaxes(gridcolor="#e8eef7", rangemode="tozero")
            tag_fig.update_yaxes(showgrid=False)
            st.plotly_chart(tag_fig, width="stretch")

    st.info(news_flow["summary_text"])
    notes = news_flow.get("issue_notes", [])
    issues = news_flow.get("top_issues", [])
    if notes:
        cards = ""
        for idx, note in enumerate(notes):
            issue = issues[idx] if idx < len(issues) else "체크포인트"
            cards += (
                '<div class="issue-note-card">'
                f'<div style="color:#0f4fd3;font-weight:850;margin-bottom:6px;">#{html.escape(str(issue))}</div>'
                f'{html.escape(note)}'
                "</div>"
            )
        st.markdown(f'<div class="issue-note-grid">{cards}</div>', unsafe_allow_html=True)
    st.caption("뉴스 제목 기반 이슈 태깅 결과입니다. 가격·손익 계산에는 직접 사용하지 않으며, 공식 자료와 함께 확인하는 참고 자료입니다.")


def render_central_bank_materials(materials_df):
    """Render official central bank material links as compact cards."""
    if materials_df is None or materials_df.empty:
        st.info("표시할 공식 자료 링크가 없습니다.")
        return

    expected = {
        "한국은행": ["최근 금통위 결정문", "총재 기자간담회", "의사록", "금융·경제 이슈"],
        "연준": ["FOMC Statement", "FOMC Minutes", "Press Conference", "Projection Materials"],
    }
    cols = st.columns(2)
    for idx, (bank, labels) in enumerate(expected.items()):
        with cols[idx]:
            render_panel_title(bank)
            bank_df = materials_df[materials_df["bank"] == bank].copy()
            card_cols = st.columns(2)
            for card_idx, label in enumerate(labels):
                row_df = bank_df[bank_df["material"] == label]
                with card_cols[card_idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{label}**")
                        if row_df.empty:
                            st.caption("자료 없음")
                            st.button("자료 없음", disabled=True, key=f"official_empty_{bank}_{label}")
                            continue
                        row = row_df.iloc[0]
                        title = str(row.get("title", label))
                        url = str(row.get("url", "") or "")
                        source = str(row.get("source", "official"))
                        status = str(row.get("status", "official"))
                        status_label = "공식 자료" if status == "official" else "대체 링크"
                        st.caption(f"{title} · {source}")
                        if url:
                            st.link_button(status_label, url, use_container_width=True)
                        else:
                            st.button("자료 없음", disabled=True, key=f"official_no_url_{bank}_{label}")


def payment_frequency_input(key_prefix):
    """Render payment frequency selectbox and return number of payments per year."""
    label = st.selectbox("이자 지급 주기", ["연 1회", "연 2회", "연 4회"], index=1, key=f"{key_prefix}_freq")
    return {"연 1회": 1, "연 2회": 2, "연 4회": 4}[label]


def bond_inputs(key_prefix):
    """Render common bond inputs."""
    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("채권 국가", ["한국채", "미국채"], key=f"{key_prefix}_country")
        bond_type = st.selectbox("채권 유형", ["이표채", "무이표채"], key=f"{key_prefix}_type")
        pricing_method = st.selectbox(
            "가격 계산 방식",
            ["단일 시장수익률 방식", "선택 국가 수익률곡선 기반 방식"],
            key=f"{key_prefix}_method",
        )
    with col2:
        face_value = st.number_input("액면가", min_value=1.0, value=10000.0, step=1000.0, key=f"{key_prefix}_face")
        coupon_rate = st.number_input("표면금리(%)", min_value=0.0, value=3.5, step=0.1, key=f"{key_prefix}_coupon")
        ytm = st.number_input("시장수익률 또는 할인율(%)", min_value=0.0, value=3.5, step=0.1, key=f"{key_prefix}_ytm")
    with col3:
        years = st.number_input("만기(년)", min_value=0.1, value=3.0, step=0.5, key=f"{key_prefix}_years")
        payments = payment_frequency_input(key_prefix)
        quantity = st.number_input("보유 수량", min_value=1, value=10, step=1, key=f"{key_prefix}_qty")

    if bond_type == "무이표채":
        coupon_rate = 0.0
    return country, bond_type, pricing_method, face_value, coupon_rate, ytm, years, payments, quantity


def calculate_selected_price(country, pricing_method, face_value, coupon_rate, ytm, years, payments):
    """Calculate price from selected pricing method."""
    if coupon_rate == 0:
        return calculate_zero_coupon_price(face_value, ytm, years)
    if pricing_method == "선택 국가 수익률곡선 기반 방식":
        curve = get_curve(country)
        price = price_bond_with_yield_curve(face_value, coupon_rate, years, payments, curve)
        if price > 0:
            return price
    return calculate_bond_price(face_value, coupon_rate, ytm, years, payments)


def fmt_unit_money(value, unit="원", decimals=0):
    """Format a money value with unit after the number."""
    if pd.isna(value):
        return "데이터 없음"
    return f"{float(value):,.{decimals}f} {unit}"


def fmt_signed_money(value, unit="원"):
    """Format signed money for scenario tables."""
    sign = "+" if value > 0 else ""
    return f"{sign}{float(value):,.0f} {unit}"


def convert_money_value(value, from_unit, exchange_rate):
    """Convert money between USD and KRW for display."""
    if from_unit == "달러":
        return float(value) * float(exchange_rate), "원"
    return float(value) / float(exchange_rate), "달러"


def date_year_fraction(start_date, end_date):
    """Return year fraction between two dates."""
    days = max((pd.to_datetime(end_date).date() - pd.to_datetime(start_date).date()).days, 1)
    return days / 365.0


def coupon_dates_between(settlement_date, maturity_date, payments_per_year):
    """Return future coupon dates from settlement to maturity."""
    settlement = pd.Timestamp(settlement_date)
    maturity = pd.Timestamp(maturity_date)
    months = max(int(12 / max(payments_per_year, 1)), 1)
    dates = []
    current = maturity
    while current > settlement:
        dates.append(current.date())
        current = current - pd.DateOffset(months=months)
    return sorted(dates)


def estimate_accrued_interest(face_value, coupon_rate, settlement_date, maturity_date, payments_per_year):
    """Estimate accrued coupon interest per one bond."""
    if coupon_rate <= 0 or payments_per_year <= 0:
        return 0.0
    months = max(int(12 / payments_per_year), 1)
    settlement = pd.Timestamp(settlement_date)
    maturity = pd.Timestamp(maturity_date)
    next_coupon = maturity
    while next_coupon > settlement:
        previous_coupon = next_coupon - pd.DateOffset(months=months)
        if previous_coupon <= settlement < next_coupon:
            period_days = max((next_coupon - previous_coupon).days, 1)
            elapsed_days = max((settlement - previous_coupon).days, 0)
            coupon_payment = face_value * (coupon_rate / 100) / payments_per_year
            return float(coupon_payment * elapsed_days / period_days)
        next_coupon = previous_coupon
    return 0.0


def make_cashflow_schedule(face_value, coupon_rate, settlement_date, maturity_date, payments_per_year, quantity):
    """Build a dated cash-flow schedule for the basic analysis page."""
    future_dates = coupon_dates_between(settlement_date, maturity_date, payments_per_year)
    coupon_payment = face_value * (coupon_rate / 100) / max(payments_per_year, 1)
    rows = []
    cumulative = 0.0
    for coupon_date in future_dates:
        cashflow_type = "원금+이자" if coupon_date == pd.to_datetime(maturity_date).date() else "이자"
        cashflow = coupon_payment + (face_value if cashflow_type == "원금+이자" else 0.0)
        total_cashflow = cashflow * quantity
        cumulative += total_cashflow
        rows.append(
            {
                "일자": coupon_date,
                "구분": cashflow_type,
                "현금흐름": total_cashflow,
                "누적 현금흐름": cumulative,
            }
        )
    return pd.DataFrame(rows)


def render_value_cards(items):
    """Render dashboard value cards."""
    cards = "".join(
        '<div class="value-card">'
        f'<div class="value-card-label">{html.escape(str(label))}</div>'
        f'<div class="value-card-value">{html.escape(str(value))}</div>'
        f'<div class="value-card-sub {html.escape(str(tone))}">{html.escape(str(subtitle))}</div>'
        "</div>"
        for label, value, subtitle, tone in items
    )
    st.markdown(f'<div class="metric-panel-grid">{cards}</div>', unsafe_allow_html=True)


def render_indicator_strip(items):
    """Render the key bond indicator strip."""
    cells = "".join(
        '<div class="indicator-cell">'
        f'<div class="indicator-label">{html.escape(str(label))}</div>'
        f'<div class="indicator-value">{html.escape(str(value))}</div>'
        f'<div class="indicator-unit">{html.escape(str(unit))}</div>'
        "</div>"
        for label, value, unit in items
    )
    st.markdown(f'<div class="indicator-strip">{cells}</div>', unsafe_allow_html=True)


def render_panel_title(title):
    """Render a local panel title."""
    st.markdown(f'<div class="panel-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def render_help(terms):
    """Render a compact help expander with one-line term explanations."""
    rows = "".join(
        '<div class="help-row">'
        f'<span class="help-term">{html.escape(str(term))}</span>: {html.escape(str(desc))}'
        "</div>"
        for term, desc in terms
    )
    st.markdown(
        f"""
        <details class="help-details">
          <summary>ⓘ 도움말</summary>
          <div class="help-body">{rows}</div>
        </details>
        """,
        unsafe_allow_html=True,
    )


def investment_score(modified_duration, ytm, spread_to_coupon):
    """Return a simple 0-10 score for summary display."""
    duration_score = max(0, 4 - abs(modified_duration - 4) * 0.45)
    yield_score = min(max(ytm, 0), 6) * 0.55
    coupon_score = max(0, 2 - abs(spread_to_coupon) * 0.35)
    return round(min(max(duration_score + yield_score + coupon_score, 0), 10), 1)


def reset_basic_inputs():
    """Reset the basic analysis form to compact starter values."""
    today = date.today()
    defaults = {
        "basic_bond_name": "미국 국채",
        "basic_kind": "미국채",
        "basic_current_price": 100.0,
        "basic_maturity": (pd.Timestamp(today) + pd.DateOffset(years=5)).date(),
        "basic_rating": "AA+",
        "basic_coupon": 3.0,
        "basic_issue_date": today,
        "basic_date_mode": "발행일 + 만기일",
        "basic_term_years": 5,
        "basic_term_months": 0,
        "basic_purchase_date": today,
        "basic_purchase_amount": 10000.0,
        "basic_face": 100.0,
        "basic_fee": 0.0,
        "basic_frequency_mode": "자동",
        "basic_manual_ytm": False,
        "basic_manual_ytm_value": 3.0,
        "basic_tax_rate": 15.4,
        "basic_issuer": "",
    }
    for key, value in defaults.items():
        st.session_state[key] = value


def sync_basic_kind_defaults():
    """Update dependent starter values when the bond type changes."""
    bond_kind = st.session_state.get("basic_kind", "미국채")
    if bond_kind == "미국채":
        st.session_state["basic_face"] = 100.0
        st.session_state["basic_issuer"] = "미국 재무부"
    elif bond_kind in {"국고채", "회사채"}:
        st.session_state["basic_face"] = 10000.0
        st.session_state["basic_issuer"] = "대한민국"
    elif bond_kind == "무이표채":
        st.session_state["basic_coupon"] = 0.0


def infer_payments_per_year(remaining_coupon_count, years_to_maturity):
    """Infer coupon frequency from an app-displayed remaining coupon count."""
    if remaining_coupon_count <= 0 or years_to_maturity <= 0:
        return 2
    estimated = remaining_coupon_count / years_to_maturity
    if estimated <= 1.35:
        return 1
    if estimated <= 2.7:
        return 2
    return 4


def currency_to_country(currency):
    """Map displayed trading currency to bond market for calculations."""
    return "미국채" if "달러" in currency or "$" in currency else "한국채"


def default_payments_for_bond(bond_kind):
    """Return the default coupon frequency from the selected bond type."""
    if bond_kind == "미국채":
        return 2
    if bond_kind in {"국고채", "회사채"}:
        return 4
    return 1


def solve_ytm_from_price(face_value, coupon_rate, price, years_to_maturity, payments_per_year):
    """Estimate YTM from face value, coupon, maturity, frequency, and current price."""
    if not all(value > 0 for value in [face_value, price, years_to_maturity, payments_per_year]):
        return 0.0
    if coupon_rate <= 0:
        return ((face_value / price) ** (1 / years_to_maturity) - 1) * 100

    low, high = -0.95, 1.0
    for _ in range(100):
        mid = (low + high) / 2
        mid_price = calculate_bond_price(face_value, coupon_rate, mid, years_to_maturity, payments_per_year)
        if mid_price > price:
            low = mid
        else:
            high = mid
    return ((low + high) / 2) * 100


def price_bond_from_dates(face_value, coupon_rate, valuation_date, maturity_date, payments_per_year, ytm):
    """Price remaining dated cash flows using the actual valuation date."""
    valuation = pd.to_datetime(valuation_date).date()
    maturity = pd.to_datetime(maturity_date).date()
    if face_value <= 0 or maturity <= valuation:
        return float(face_value if maturity == valuation else 0.0)
    dates = coupon_dates_between(valuation, maturity, payments_per_year)
    if not dates:
        dates = [maturity]
    coupon_payment = face_value * max(coupon_rate, 0.0) / 100 / max(payments_per_year, 1)
    total = 0.0
    for cashflow_date in dates:
        cashflow = coupon_payment + (face_value if cashflow_date == maturity else 0.0)
        year_fraction = max((cashflow_date - valuation).days, 1) / 365
        total += cashflow / ((1 + ytm / max(payments_per_year, 1)) ** (payments_per_year * year_fraction))
    return float(total)


def solve_ytm_from_dates(face_value, coupon_rate, price, valuation_date, maturity_date, payments_per_year):
    """Estimate YTM from actual dated cash flows and return percent units."""
    if not all(value > 0 for value in [face_value, price, payments_per_year]):
        return 0.0
    if pd.to_datetime(maturity_date).date() <= pd.to_datetime(valuation_date).date():
        return 0.0
    low, high = -0.95, 1.0
    for _ in range(100):
        mid = (low + high) / 2
        mid_price = price_bond_from_dates(face_value, coupon_rate, valuation_date, maturity_date, payments_per_year, mid)
        if mid_price > price:
            low = mid
        else:
            high = mid
    return ((low + high) / 2) * 100


def compute_trade_quantity(purchase_budget, price_per_bond, face_value, trade_unit_face=0):
    """Calculate purchasable bond quantity, optionally rounded by face-value trade unit."""
    if purchase_budget <= 0 or price_per_bond <= 0 or face_value <= 0:
        return {"quantity": 0.0, "face_total": 0.0, "cash_used": 0.0, "unused_cash": max(purchase_budget, 0.0)}
    raw_quantity = purchase_budget / price_per_bond
    if trade_unit_face and trade_unit_face > 0:
        face_affordable = purchase_budget / (price_per_bond / face_value)
        face_total = np.floor(face_affordable / trade_unit_face) * trade_unit_face
        quantity = face_total / face_value
    else:
        quantity = raw_quantity
        face_total = face_value * quantity
    cash_used = price_per_bond * quantity
    return {
        "quantity": float(quantity),
        "face_total": float(face_total),
        "cash_used": float(cash_used),
        "unused_cash": float(max(purchase_budget - cash_used, 0.0)),
    }


def page_home():
    """Render the macro rate environment page."""
    render_page_header(
        "개인 채권 가격 분석 시뮬레이터",
        "거시 금리 환경과 개인 채권 분석에 필요한 핵심 지표를 한눈에 확인합니다.",
        "홈 · 거시 금리 환경",
        "rate",
    )

    policy_rates = cached_policy_rates()
    calendar = cached_policy_calendar()
    macro = cached_macro_rates()
    kr_history = cached_bond_yield_history()
    kr_curve = cached_kr_curve()
    us_curve = cached_us_curve()
    us_history = cached_us_history()
    exchange_rate = get_latest_exchange_rate_value()
    official_materials = cached_central_bank_materials()
    policy_news = cached_policy_news()
    news_flow = summarize_news_flow(policy_news)
    classified_news = news_flow["classified_df"]

    curve_map = dict(zip(kr_curve["maturity_years"], kr_curve["yield"]))
    us_map = dict(zip(us_curve["series_id"], us_curve["yield"]))
    kr_10_3_spread = safe_pct_spread(curve_map.get(10), curve_map.get(3))
    us_spread = safe_pct_spread(us_map.get("DGS10"), us_map.get("DGS2"))
    kr_policy = policy_rates[policy_rates["country"] == "한국"].iloc[0]
    us_policy = policy_rates[policy_rates["country"] == "미국"].iloc[0]

    with st.container(border=True):
        render_section_title("오늘의 금리 환경 요약", "rate")
        summary_cols = st.columns(5)
        render_mini_metric(summary_cols[0], "한국 정책금리", format_policy_value(kr_policy), "kr_flag")
        render_mini_metric(summary_cols[1], "미국 정책금리", format_policy_value(us_policy), "us_flag")
        render_mini_metric(summary_cols[2], "주요 뉴스 이슈", news_flow["top_issue"], "news")
        render_mini_metric(summary_cols[3], "한국 장단기 금리차", kr_10_3_spread, "chart")
        render_mini_metric(summary_cols[4], "미국 10Y-2Y", us_spread, "chart")
        st.markdown('<div class="section-bottom-spacer"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        render_section_title("한/미 정책금리", "rate")
        kr_policy_col, us_policy_col, exchange_col = st.columns(3)
        render_policy_card(kr_policy_col, "한국", policy_rates, calendar)
        render_policy_card(us_policy_col, "미국", policy_rates, calendar)
        render_exchange_card(exchange_col, exchange_rate)

    kr_col, us_col = st.columns(2)
    with kr_col:
        with st.container(border=True):
            render_section_title("한국 금리 환경", "kr_flag")
            macro_map = macro.set_index("indicator")["rate"].to_dict()
            st.markdown('<div class="subgroup-title">정책/단기 금리</div>', unsafe_allow_html=True)
            render_rate_strip(
                [
                    ("기준금리", fmt_rate(macro_map.get("한국 기준금리"))),
                    ("CD 91일물", fmt_rate(macro_map.get("CD 91일물"))),
                    ("회사채 AA-", fmt_rate(macro_map.get("회사채 AA-"))),
                ]
            )
            st.markdown('<div class="subgroup-title">국고채 만기 구조</div>', unsafe_allow_html=True)
            render_rate_strip(
                [
                    ("1년", fmt_rate(curve_map.get(1))),
                    ("2년", fmt_rate(curve_map.get(2))),
                    ("3년", fmt_rate(curve_map.get(3))),
                    ("5년", fmt_rate(curve_map.get(5))),
                    ("10년", fmt_rate(curve_map.get(10))),
                    ("20년", fmt_rate(curve_map.get(20))),
                    ("30년", fmt_rate(curve_map.get(30))),
                ]
            )
            st.markdown('<div class="subgroup-title">장단기 금리차</div>', unsafe_allow_html=True)
            render_rate_strip(
                [
                    ("10년 - 3년", safe_pct_spread(curve_map.get(10), curve_map.get(3))),
                    ("10년 - 1년", safe_pct_spread(curve_map.get(10), curve_map.get(1))),
                    ("3년 - 1년", safe_pct_spread(curve_map.get(3), curve_map.get(1))),
                ]
            )

    with us_col:
        with st.container(border=True):
            render_section_title("미국 금리 환경", "us_flag")
            if "source" in us_curve.columns and (us_curve["source"] == "sample").any():
                st.caption("FRED API key가 없거나 호출에 실패하여 sample 미국채 데이터를 표시 중입니다.")
            st.markdown('<div class="subgroup-title">미국채 만기 구조</div>', unsafe_allow_html=True)
            render_rate_strip(
                [
                    ("1M", fmt_rate(us_map.get("DGS1MO"))),
                    ("3M", fmt_rate(us_map.get("DGS3MO"))),
                    ("6M", fmt_rate(us_map.get("DGS6MO"))),
                    ("2Y", fmt_rate(us_map.get("DGS2"))),
                    ("10Y", fmt_rate(us_map.get("DGS10"))),
                    ("30Y", fmt_rate(us_map.get("DGS30"))),
                ]
            )
            st.markdown('<div class="subgroup-title">장단기 금리차</div>', unsafe_allow_html=True)
            render_rate_strip([("10년 - 2년", us_spread)])

    chart_cols = st.columns(3)
    with chart_cols[0]:
        kr_fig = px.line(
            kr_history,
            x="date",
            y=["국고채 3년", "국고채 10년"],
            labels={"value": "수익률(%)", "date": "날짜"},
        )
        plot_line_card("최근 금리 흐름 (한국)", kr_fig, height=245, icon_key="kr_flag")
    with chart_cols[1]:
        us_fig = px.line(
            us_history,
            x="date",
            y=["DGS2", "DGS10", "DGS30"],
            labels={"value": "수익률(%)", "date": "날짜"},
        )
        plot_line_card("최근 금리 흐름 (미국)", us_fig, height=245, icon_key="us_flag")
    with chart_cols[2]:
        compare = pd.concat(
            [
                kr_curve[["maturity_years", "yield"]].assign(country="한국 국고채"),
                us_curve[["maturity_years", "yield"]].assign(country="미국채"),
            ],
            ignore_index=True,
        )
        curve_fig = px.line(
            compare,
            x="maturity_years",
            y="yield",
            color="country",
            markers=True,
            labels={"yield": "수익률(%)", "maturity_years": "만기(년)"},
        )
        plot_line_card("한미 수익률곡선 비교", curve_fig, height=245, icon_key="chart")

    with st.container(border=True):
        render_section_title("Forward Rate", "chart")
        render_forward_calculator(kr_curve, us_curve)

    with st.container(border=True):
        render_section_title("한/미 중앙은행 공식 자료", "rate")
        render_central_bank_materials(official_materials)

    with st.container(border=True):
        render_section_title("최근 시장 관심 이슈", "news")
        render_news_flow_dashboard(news_flow)

    with st.container(border=True):
        render_section_title("시장 주요 뉴스", "news")
        st.dataframe(make_news_links(classified_news, include_classification=True), width="stretch", hide_index=True)


def page_basic_analysis():
    """Render the bond input and basic analysis page."""
    render_page_header(
        "채권 입력 및 기본 분석",
        "증권앱에서 확인한 채권 조건을 입력하고 가격, YTM, 듀레이션, 현금흐름을 바로 점검합니다.",
        "분석 · 보유 채권 점검",
        "chart",
    )

    with st.container(border=True):
        input_header_cols = st.columns([1, 0.16])
        with input_header_cols[0]:
            render_panel_title("채권 정보 입력")
        with input_header_cols[1]:
            st.button("초기화", use_container_width=True, on_click=reset_basic_inputs)
        basic_cols = st.columns(4)
        with basic_cols[0]:
            bond_name = st.text_input("상품명", value="미국 국채", key="basic_bond_name")
        with basic_cols[1]:
            bond_kind = st.selectbox(
                "채권 종류",
                ["미국채", "국고채", "회사채", "무이표채"],
                key="basic_kind",
                on_change=sync_basic_kind_defaults,
            )
        with basic_cols[2]:
            currency_label = "미국 달러 ($)" if bond_kind == "미국채" else "원화 (원)"
            st.text_input("거래통화", value=currency_label, disabled=True)
        with basic_cols[3]:
            credit_rating = st.selectbox(
                "신용등급",
                ["AAA", "AA+", "AA", "AA-", "A+", "A", "BBB+", "BBB", "등급 없음"],
                index=1,
                key="basic_rating",
            )

        price_cols = st.columns(3)
        with price_cols[0]:
            current_price_input = st.number_input(
                "현재 가격(액면가 기준)",
                min_value=1.0,
                value=100.0,
                step=0.1,
                format="%.2f",
                key="basic_current_price",
            )
        with price_cols[1]:
            face_default = 100.0 if "달러" in currency_label else 10000.0
            if "basic_face" not in st.session_state:
                st.session_state["basic_face"] = face_default
            face = st.number_input("액면가", min_value=1.0, step=100.0, format="%.0f", key="basic_face")
        with price_cols[2]:
            coupon = st.number_input(
                "이자율/쿠폰(연, %)",
                min_value=0.0,
                value=3.000,
                step=0.001,
                format="%.3f",
                key="basic_coupon",
            )

        info_cols = st.columns(2)
        with info_cols[0]:
            trade_cols = st.columns(3)
            with trade_cols[0]:
                purchase_date = st.date_input(
                    "매수/조회일",
                    value=date.today(),
                    min_value=date(1900, 1, 1),
                    max_value=date(2100, 12, 31),
                    key="basic_purchase_date",
                )
            with trade_cols[1]:
                purchase_amount = st.number_input(
                    "구매 금액",
                    min_value=1.0,
                    value=10000.0,
                    step=1000.0,
                    format="%.2f",
                    key="basic_purchase_amount",
                )
            with trade_cols[2]:
                fee_input = st.number_input(
                    "구매 수수료",
                    min_value=0.0,
                    value=0.0,
                    step=10.0,
                    format="%.2f",
                    key="basic_fee",
                )
        with info_cols[1]:
            date_mode = st.selectbox(
                "일정 입력 방식",
                ["발행일 + 만기일", "발행일 + 만기 기간", "만기일 + 만기 기간"],
                key="basic_date_mode",
            )
            date_cols = st.columns(2)
            with date_cols[0]:
                if date_mode in {"발행일 + 만기일", "발행일 + 만기 기간"}:
                    issue_date = st.date_input(
                        "발행일",
                        value=date.today(),
                        min_value=date(1900, 1, 1),
                        max_value=date(2100, 12, 31),
                        key="basic_issue_date",
                    )
                else:
                    maturity_for_calc = st.date_input(
                        "만기일",
                        value=date(2031, 5, 28),
                        min_value=date(1900, 1, 1),
                        max_value=date(2100, 12, 31),
                        key="basic_maturity",
                    )
            with date_cols[1]:
                if date_mode == "발행일 + 만기일":
                    maturity_date = st.date_input(
                        "만기일",
                        value=date(2031, 5, 28),
                        min_value=date(1900, 1, 1),
                        max_value=date(2100, 12, 31),
                        key="basic_maturity",
                    )
                else:
                    term_years = st.number_input("만기 기간(년)", min_value=0, max_value=100, value=5, step=1, key="basic_term_years")
            if date_mode != "발행일 + 만기일":
                term_cols = st.columns([1, 1])
                with term_cols[0]:
                    term_months = st.number_input("추가 기간(개월)", min_value=0, max_value=11, value=0, step=1, key="basic_term_months")
                term_offset = pd.DateOffset(years=int(term_years), months=int(term_months))
                if date_mode == "발행일 + 만기 기간":
                    maturity_date = (pd.Timestamp(issue_date) + term_offset).date()
                    term_cols[1].text_input("계산된 만기일", value=str(maturity_date), disabled=True)
                else:
                    maturity_date = maturity_for_calc
                    issue_date = (pd.Timestamp(maturity_date) - term_offset).date()
                    term_cols[1].text_input("계산된 발행일", value=str(issue_date), disabled=True)

        frequency_mode = st.session_state.get("basic_frequency_mode", "자동")
        manual_ytm = st.session_state.get("basic_manual_ytm", False)
        manual_ytm_value = st.session_state.get("basic_manual_ytm_value", 3.0)
        tax_rate = st.session_state.get("basic_tax_rate", 15.4)
        issuer = st.session_state.get("basic_issuer") or ("미국 재무부" if bond_kind == "미국채" else "대한민국")
        trade_unit_face = st.session_state.get("basic_trade_unit_face", 100.0 if bond_kind == "미국채" else 1.0)
        preview_payments = default_payments_for_bond(bond_kind)
        if frequency_mode == "연 1회":
            preview_payments = 1
        elif frequency_mode == "연 2회":
            preview_payments = 2
        elif frequency_mode == "연 4회":
            preview_payments = 4
        preview_coupon_dates = coupon_dates_between(purchase_date, maturity_date, preview_payments)
        remaining_coupon_count = len(preview_coupon_dates)
        elapsed_since_issue = max((pd.to_datetime(purchase_date).date() - pd.to_datetime(issue_date).date()).days, 0) / 365
        derived_cols = st.columns(4)
        derived_cols[0].text_input("자동 이자 지급 주기", value=f"연 {preview_payments}회", disabled=True)
        derived_cols[1].text_input("남은 이자 지급 횟수", value=f"{remaining_coupon_count}회", disabled=True)
        derived_cols[2].text_input("잔존 만기", value=f"{date_year_fraction(purchase_date, maturity_date):.2f}년", disabled=True)
        derived_cols[3].text_input("발행 후 경과", value=f"{elapsed_since_issue:.2f}년", disabled=True)

        with st.expander("고급 입력: 자동 계산값을 직접 조정할 때만 사용"):
            opt_cols = st.columns(5)
            with opt_cols[0]:
                frequency_mode = st.selectbox("이자 지급 주기", ["자동", "연 1회", "연 2회", "연 4회"], index=0, key="basic_frequency_mode")
                auto_frequency = default_payments_for_bond(bond_kind)
                st.caption(f"자동값: 연 {auto_frequency}회")
            with opt_cols[1]:
                manual_ytm_now = st.session_state.get("basic_manual_ytm", False)
                manual_ytm_value = st.number_input(
                    "직접 입력 YTM(%)",
                    min_value=0.0,
                    value=3.0,
                    step=0.01,
                    format="%.2f",
                    disabled=not manual_ytm_now,
                    key="basic_manual_ytm_value",
                )
                manual_ytm = st.checkbox("YTM 직접 입력", value=False, key="basic_manual_ytm")
            with opt_cols[2]:
                tax_rate = st.slider("세율 추정용(%)", min_value=0.0, max_value=30.0, value=15.4, step=0.1, key="basic_tax_rate")
            with opt_cols[3]:
                issuer_default = "미국 재무부" if bond_kind == "미국채" else "대한민국"
                issuer = st.text_input("발행 주체(선택)", value=issuer_default, key="basic_issuer")
            with opt_cols[4]:
                trade_unit_face = st.number_input(
                    "매매단위(액면)",
                    min_value=0.0,
                    value=100.0 if bond_kind == "미국채" else 1.0,
                    step=100.0 if bond_kind == "미국채" else 1.0,
                    format="%.0f",
                    key="basic_trade_unit_face",
                )
        render_help(
            [
                ("YTM", "현재 가격으로 사서 만기까지 보유할 때의 연 환산 수익률입니다."),
                ("매매단위", "증권사에서 실제로 살 수 있는 최소 액면 단위입니다. 0이면 단위 절사를 하지 않습니다."),
            ]
        )

    country = "미국채" if bond_kind == "미국채" else currency_to_country(currency_label)
    unit = "달러" if country == "미국채" else "원"
    years = date_year_fraction(purchase_date, maturity_date)
    coupon = 0.0 if bond_kind == "무이표채" else coupon
    if frequency_mode == "연 1회":
        payments = 1
    elif frequency_mode == "연 2회":
        payments = 2
    elif frequency_mode == "연 4회":
        payments = 4
    else:
        payments = default_payments_for_bond(bond_kind)
    dirty_price = min(float(current_price_input), face * 20)
    auto_ytm = solve_ytm_from_dates(face, coupon, dirty_price, purchase_date, maturity_date, payments)
    ytm = manual_ytm_value if manual_ytm else auto_ytm
    accrued_interest = estimate_accrued_interest(face, coupon, purchase_date, maturity_date, payments)
    clean_price = max(dirty_price - accrued_interest, 0.0)
    trade_info = compute_trade_quantity(purchase_amount, dirty_price, face, trade_unit_face)
    quantity = trade_info["quantity"]
    purchase_cash_used = trade_info["cash_used"]
    investment_base = purchase_cash_used + fee_input
    total_value = calculate_total_value(dirty_price, quantity)
    next_coupon_dates = coupon_dates_between(purchase_date, maturity_date, payments)
    coupon_income = face * (coupon / 100) / max(payments, 1) * len(next_coupon_dates) * quantity
    macaulay = calculate_macaulay_duration(face, coupon, ytm / 100, years, payments)
    modified = calculate_modified_duration(face, coupon, ytm / 100, years, payments)
    convexity = calculate_convexity(face, coupon, ytm / 100, years, payments)
    after_tax_yield = ytm * (1 - tax_rate / 100)
    dv01 = modified * total_value * 0.0001
    next_coupon = next_coupon_dates[0] if next_coupon_dates else maturity_date
    d_day = max((pd.to_datetime(next_coupon).date() - pd.to_datetime(purchase_date).date()).days, 0)
    maturity_cashflow = face * quantity + coupon_income
    gross_profit = maturity_cashflow - purchase_cash_used
    estimated_tax = max(coupon_income * tax_rate / 100, 0.0)
    net_profit = gross_profit - estimated_tax - fee_input
    pnl_vs_purchase = net_profit
    pnl_pct = net_profit / investment_base * 100 if investment_base else 0.0
    gross_pct = gross_profit / investment_base * 100 if investment_base else 0.0
    issued_text = f"{bond_name} · {credit_rating} · {currency_label}"
    save_basic_bond_snapshot(
        {
            "bond_name": bond_name,
            "bond_kind": bond_kind,
            "country": country,
            "currency": unit,
            "issuer": issuer,
            "credit_rating": credit_rating,
            "purchase_date": purchase_date,
            "issue_date": issue_date,
            "maturity_date": maturity_date,
            "face": face,
            "coupon": coupon,
            "current_price": dirty_price,
            "purchase_amount": purchase_amount,
            "buy_fee": fee_input,
            "payments": payments,
            "current_ytm": ytm,
            "trade_unit_face": trade_unit_face,
        }
    )

    with st.container(border=True):
        render_panel_title("현재 가격 및 수익률")
        exchange_row = get_latest_exchange_rate_value()
        converted_result_view = st.toggle(
            f"{'원화' if unit == '달러' else '달러'} 환산 결과 보기",
            value=False,
            key="basic_converted_result_view",
        )
        display_unit = unit
        display_rate_note = ""
        if converted_result_view:
            display_unit = "원" if unit == "달러" else "달러"
            display_rate_note = f" · 적용 환율 {fmt_fx_rate(exchange_row['rate'])}"

        def result_money(value, decimals=0, signed=False):
            display_value = value
            target_unit = unit
            if converted_result_view:
                display_value, target_unit = convert_money_value(value, unit, exchange_row["rate"])
            if signed:
                return fmt_signed_money(display_value, target_unit)
            return fmt_unit_money(display_value, target_unit, decimals)

        yesterday_price = dirty_price / 1.0003
        yesterday_clean = clean_price / 1.0003 if clean_price else 0
        render_value_cards(
            [
                ("실제 매수금액", result_money(purchase_cash_used), f"액면 {trade_info['face_total']:,.0f} · 미사용 {result_money(trade_info['unused_cash'])}{display_rate_note}", ""),
                ("만기 전 총 현금흐름", result_money(maturity_cashflow), f"쿠폰 수입 {result_money(coupon_income)}", ""),
                ("세전 예상손익", result_money(gross_profit, signed=True), f"수익률 {gross_pct:+.2f}%", "positive" if gross_profit >= 0 else "negative"),
                ("세후 예상손익", result_money(net_profit, signed=True), f"수익률 {pnl_pct:+.2f}%", "positive" if net_profit >= 0 else "negative"),
                ("자동 계산 YTM", f"{ytm:.2f} %", "현재 가격과 쿠폰으로 역산", "negative" if ytm < coupon else ""),
                ("이자수익률(Coupon)", f"{coupon:.2f} %", f"지급 주기 연 {payments}회", ""),
            ]
        )
        render_help(
            [
                ("만기 전 총 현금흐름", "만기까지 받을 원금과 이자를 더한 금액입니다."),
                ("자동 계산 YTM", "현재 가격과 쿠폰으로 앱이 거꾸로 계산한 만기수익률입니다."),
                ("이자수익률", "채권 액면가를 기준으로 정해진 쿠폰 이자율입니다."),
            ]
        )

        render_panel_title("주요 채권 지표")
        render_indicator_strip(
            [
                ("듀레이션(Modified)", f"{modified:.2f}", "년"),
                ("맥컬리 듀레이션", f"{macaulay:.2f}", "년"),
                ("컨벡서티", f"{convexity:.2f}", ""),
                ("DV01", f"{dv01:,.0f}", unit),
                ("가격/액면", f"{dirty_price:,.2f}", f"액면 {face:,.0f} 기준"),
                ("다음 이자 지급일", str(next_coupon), f"D-{d_day}"),
            ]
        )
        render_help(
            [
                ("듀레이션(Modified)", "금리가 1%p 움직일 때 가격이 몇 % 변할지 보는 민감도입니다."),
                ("맥컬리 듀레이션", "투자 원금을 평균적으로 회수하는 데 걸리는 기간입니다."),
                ("Convexity", "금리 변화가 클 때 듀레이션 오차를 보완하는 곡률 지표입니다."),
                ("DV01", "금리가 0.01%p 움직일 때 보유 채권 가치가 변하는 금액입니다."),
                ("가격/액면", "현재 가격이 액면가 대비 얼마인지 보여줍니다."),
                ("다음 이자 지급일", "가장 가까운 다음 쿠폰 이자 지급 예정일입니다."),
            ]
        )

        cashflow_schedule = make_cashflow_schedule(face, coupon, purchase_date, maturity_date, payments, quantity)
        display_cashflows = cashflow_schedule.head(6).copy()
        if len(cashflow_schedule) > 7:
            display_cashflows = pd.concat([display_cashflows, cashflow_schedule.tail(1)], ignore_index=True)

        content_cols = st.columns([0.82, 1.18])
        with content_cols[0]:
            render_panel_title("현금흐름 일정")
            st.dataframe(
                display_cashflows.style.format({"현금흐름": "{:,.0f}", "누적 현금흐름": "{:,.0f}"}),
                width="stretch",
                hide_index=True,
            )
            if len(cashflow_schedule) > len(display_cashflows):
                with st.expander("전체 현금흐름 보기"):
                    st.dataframe(
                        cashflow_schedule.style.format({"현금흐름": "{:,.0f}", "누적 현금흐름": "{:,.0f}"}),
                        width="stretch",
                        hide_index=True,
                    )
        with content_cols[1]:
            render_panel_title("가격 민감도")
            bp_range = list(range(-100, 101, 25))
            sensitivity = pd.DataFrame(
                {
                    "수익률 변화(bp)": bp_range,
                    "예상 가격": [
                        calculate_bond_price(face, coupon, (ytm + bp / 100) / 100, years, payments)
                        if coupon > 0
                        else calculate_zero_coupon_price(face, (ytm + bp / 100) / 100, years)
                        for bp in bp_range
                    ],
                }
            )
            fig = px.line(sensitivity, x="수익률 변화(bp)", y="예상 가격", markers=True)
            fig.add_vline(x=0, line_dash="dot", line_color="#1552c7")
            fig.update_traces(line_color="#1552c7", marker=dict(color="#1552c7", size=7))
            fig.update_layout(height=300, margin=dict(l=8, r=8, t=12, b=8), paper_bgcolor="white", plot_bgcolor="white")
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor="#e8eef7")
            st.plotly_chart(fig, width="stretch")
            st.caption("Y축은 Dirty Price 기준")

        render_panel_title("수익률 시나리오 분석")
        scenario_bps = [-100, -50, -25, 0, 25, 50, 100]
        scenario_rows = {"시나리오": [], "예상 가격": [], "가격 변화율": [], "추정 손익": []}
        for bp in scenario_bps:
            scenario_price = (
                calculate_bond_price(face, coupon, (ytm + bp / 100) / 100, years, payments)
                if coupon > 0
                else calculate_zero_coupon_price(face, (ytm + bp / 100) / 100, years)
            )
            scenario_rows["시나리오"].append("현재" if bp == 0 else f"{bp:+d}bp")
            scenario_rows["예상 가격"].append(fmt_unit_money(scenario_price, unit, 0))
            scenario_rows["가격 변화율"].append("-" if bp == 0 else f"{(scenario_price / dirty_price - 1) * 100:+.2f}%")
            scenario_rows["추정 손익"].append("-" if bp == 0 else fmt_signed_money((scenario_price - dirty_price) * quantity, unit))
        st.dataframe(pd.DataFrame(scenario_rows), width="stretch", hide_index=True)
        render_help(
            [
                ("가격 민감도", "시장금리가 변할 때 채권 가격이 얼마나 오르내리는지를 보여주는 분석입니다."),
            ]
        )

    with st.container(border=True):
        render_panel_title("종합 분석 요약")
        score = investment_score(modified, ytm, ytm - coupon)
        score_pct = max(0, min(score * 10, 100))
        score_label = "양호" if score >= 8 else "보통" if score >= 5 else "주의"
        score_verdict = (
            "수익성과 금리 민감도의 균형이 비교적 양호한 편입니다."
            if score >= 8
            else "수익 기회는 있으나 금리 변동과 만기 구조를 함께 확인해야 합니다."
            if score >= 5
            else "수익보다 가격 변동 위험이 더 크게 보일 수 있어 보수적인 검토가 필요합니다."
        )
        pnl_tone = "positive" if pnl_vs_purchase >= 0 else "negative"
        summary_cols = st.columns([0.29, 0.45, 0.26])
        with summary_cols[0]:
            st.markdown(
                f"""
                <div class="summary-score">
                  <div class="score-gauge-wrap">
                    <div class="score-donut" style="--score-pct: {score_pct:.1f}%;" role="img" aria-label="투자 매력도 {score:.1f}점">
                      <div class="score-donut-inner">
                        <div class="score-number">{score:.1f}</div>
                        <div class="score-denom">/ 10</div>
                      </div>
                    </div>
                  </div>
                  <div class="score-label">{score_label}</div>
                  <div class="score-caption">수익률, 듀레이션, 쿠폰 스프레드를 단순 점수화</div>
                  <div class="summary-verdict">{html.escape(score_verdict)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with summary_cols[1]:
            st.markdown(
                f"""
                <div class="analysis-list">
                  <div class="analysis-card">
                    <div class="analysis-icon">1</div>
                    <div class="analysis-item-title">금리 환경</div>
                    <div class="analysis-item-body">YTM이 표면이자율 대비 {abs(ytm - coupon):.2f}%p {'높아 할인 요인이 있습니다' if ytm > coupon else '낮아 프리미엄 요인이 있습니다'}.</div>
                  </div>
                  <div class="analysis-card">
                    <div class="analysis-icon">2</div>
                    <div class="analysis-item-title">듀레이션</div>
                    <div class="analysis-item-body">수정 듀레이션 {modified:.2f}년, 1%p 금리 상승 시 약 {modified:.2f}% 가격 하락 압력이 있습니다.</div>
                  </div>
                  <div class="analysis-card">
                    <div class="analysis-icon">3</div>
                    <div class="analysis-item-title">수익률</div>
                    <div class="analysis-item-body">세후 수익률 {after_tax_yield:.2f}%, 만기까지 예상 쿠폰 수입 {fmt_unit_money(coupon_income, unit, 0)}입니다.</div>
                  </div>
                  <div class="analysis-card">
                    <div class="analysis-icon">4</div>
                    <div class="analysis-item-title">리스크 체크</div>
                    <div class="analysis-item-body">DV01은 {fmt_unit_money(dv01, unit, 0)}로, 금리 1bp 변동에 따른 보유 평가액 민감도입니다.</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with summary_cols[2]:
            st.markdown(
                f"""
                <div class="summary-kpi-grid">
                  <div class="summary-kpi">
                    <div class="summary-kpi-label">채권 종류</div>
                    <div class="summary-kpi-value">{html.escape(bond_kind)}</div>
                    <div class="summary-kpi-sub">{html.escape(issuer)}</div>
                  </div>
                  <div class="summary-kpi">
                    <div class="summary-kpi-label">만기</div>
                    <div class="summary-kpi-value">{maturity_date}</div>
                    <div class="summary-kpi-sub">잔존 {years:.2f}년</div>
                  </div>
                  <div class="summary-kpi">
                    <div class="summary-kpi-label">YTM</div>
                    <div class="summary-kpi-value">{ytm:.2f}%</div>
                    <div class="summary-kpi-sub">현재 입력 기준</div>
                  </div>
                  <div class="summary-kpi">
                    <div class="summary-kpi-label">보유 평가손익</div>
                    <div class="summary-kpi-value">{fmt_signed_money(pnl_vs_purchase, unit)}</div>
                    <div class="summary-kpi-sub {pnl_tone}">{pnl_pct:+.2f}%</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="summary-footnote">입력 상품: {html.escape(issued_text)} · 본 분석은 참고용이며 실제 투자 판단의 근거로 사용할 수 없습니다.</div>',
            unsafe_allow_html=True,
        )
        render_help(
            [
                ("투자 매력도", "YTM, 수정 듀레이션, 쿠폰과 YTM 차이를 0~10점으로 단순화한 참고 지표입니다."),
                ("YTM 반영", "만기수익률이 높을수록 기대 수익성이 높다고 보고 점수에 더 크게 반영합니다."),
                ("듀레이션 반영", "수정 듀레이션이 길수록 금리 변화에 가격이 크게 흔들릴 수 있어 위험 요인으로 봅니다."),
                ("쿠폰 스프레드", "YTM과 쿠폰 차이가 크면 현재 가격이 액면가에서 벗어난 상태로 보고 점수를 조정합니다."),
                ("금리 환경", "현재 YTM이 쿠폰보다 높으면 할인 거래, 낮으면 프리미엄 거래 가능성이 있습니다."),
                ("수익률", "세후 수익률과 만기까지 받을 쿠폰 수입을 함께 보여줍니다."),
                ("리스크 체크", "DV01은 금리 1bp 변화에 따른 보유 평가액 변화를 뜻합니다."),
                ("주의", "이 점수는 신용위험, 유동성, 실제 호가 스프레드, 세금 차이를 모두 반영하지 못합니다."),
            ]
        )


def shifted_curve(curve_df, bp):
    """Return a yield curve shifted by basis points."""
    shifted = curve_df.copy()
    shifted["yield"] = shifted["yield"] + bp / 100
    return shifted


def curve_price_after_shift(country, face, coupon, years, payments, bp):
    """Price a bond with selected country's yield curve after a parallel shift."""
    return price_bond_with_yield_curve(face, coupon, years, payments, shifted_curve(get_curve(country), bp))


def clamp(value, min_value, max_value):
    """Clamp numeric value into a range."""
    return max(min(float(value), max_value), min_value)


def fmt_bp(value):
    """Format a basis-point value without hiding small automatic changes."""
    value = float(value)
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value)):+d}bp"
    return f"{value:+.1f}bp"


def basic_value(key, default):
    """Read a basic-analysis session value with a fallback."""
    return st.session_state.get(key, default)


def normalize_simulation_bond(bond):
    """Return a simulation bond dict with stable types and fallback fields."""
    normalized = dict(bond or {})
    country = normalized.get("country") or ("미국채" if normalized.get("bond_kind") == "미국채" else "한국채")
    normalized["country"] = country
    normalized.setdefault("bond_kind", "미국채" if country == "미국채" else "국고채")
    normalized.setdefault("bond_name", "미국 국채" if country == "미국채" else "국고채")
    normalized.setdefault("currency", "달러" if country == "미국채" else "원")
    normalized.setdefault("issuer", "미국 재무부" if country == "미국채" else "대한민국")
    normalized.setdefault("credit_rating", "AA+")
    normalized["purchase_date"] = pd.to_datetime(normalized.get("purchase_date", date(2026, 5, 28))).date()
    normalized["issue_date"] = pd.to_datetime(normalized.get("issue_date", normalized["purchase_date"])).date()
    normalized["maturity_date"] = pd.to_datetime(normalized.get("maturity_date", date(2031, 5, 28))).date()
    numeric_defaults = {
        "face": 100.0 if country == "미국채" else 10000.0,
        "coupon": 3.0,
        "current_price": 100.0 if country == "미국채" else 10000.0,
        "purchase_amount": 10000.0,
        "buy_fee": 0.0,
        "current_ytm": 3.0,
        "trade_unit_face": 100.0 if country == "미국채" else 1.0,
    }
    for key, default in numeric_defaults.items():
        normalized[key] = float(normalized.get(key, default))
    normalized["payments"] = int(normalized.get("payments", default_payments_for_bond(normalized["bond_kind"])))
    return normalized


def save_basic_bond_snapshot(snapshot):
    """Persist a non-widget bond snapshot for pages that do not render basic inputs."""
    st.session_state["analysis_bond_snapshot"] = normalize_simulation_bond(snapshot)


def get_simulation_bond_defaults():
    """Build simulation defaults from the basic-analysis page when available."""
    snapshot = st.session_state.get("analysis_bond_snapshot")
    if snapshot:
        return normalize_simulation_bond(snapshot)

    bond_kind = basic_value("basic_kind", "미국채")
    country = "미국채" if bond_kind == "미국채" else "한국채"
    purchase_date = basic_value("basic_purchase_date", date(2026, 5, 28))
    maturity_date = basic_value("basic_maturity", date(2031, 5, 28))
    face = float(basic_value("basic_face", 100.0 if country == "미국채" else 10000.0))
    coupon = float(basic_value("basic_coupon", 3.0))
    price = float(basic_value("basic_current_price", face))
    payments = default_payments_for_bond(bond_kind)
    frequency_mode = basic_value("basic_frequency_mode", "자동")
    if frequency_mode == "연 1회":
        payments = 1
    elif frequency_mode == "연 2회":
        payments = 2
    elif frequency_mode == "연 4회":
        payments = 4
    years = date_year_fraction(purchase_date, maturity_date)
    auto_ytm = solve_ytm_from_dates(face, coupon, price, purchase_date, maturity_date, payments)
    ytm = float(basic_value("basic_manual_ytm_value", auto_ytm)) if basic_value("basic_manual_ytm", False) else auto_ytm
    return normalize_simulation_bond({
        "bond_name": basic_value("basic_bond_name", "미국 국채" if country == "미국채" else "국고채"),
        "bond_kind": bond_kind,
        "country": country,
        "currency": "달러" if country == "미국채" else "원",
        "issuer": basic_value("basic_issuer", "미국 재무부" if country == "미국채" else "대한민국"),
        "credit_rating": basic_value("basic_rating", "AA+"),
        "purchase_date": purchase_date,
        "issue_date": basic_value("basic_issue_date", purchase_date),
        "maturity_date": maturity_date,
        "face": face,
        "coupon": 0.0 if bond_kind == "무이표채" else coupon,
        "current_price": price,
        "purchase_amount": float(basic_value("basic_purchase_amount", 10000.0)),
        "buy_fee": float(basic_value("basic_fee", 0.0)),
        "payments": payments,
        "current_ytm": ytm,
        "trade_unit_face": float(basic_value("basic_trade_unit_face", 100.0 if country == "미국채" else 1.0)),
    })


def policy_reflection_rate(years, mode, policy_change_bp):
    """Return how much policy-rate movement is reflected in YTM movement."""
    if years <= 1:
        base = 0.9
    elif years <= 3:
        base = 0.65
    elif years <= 7:
        base = 0.45
    else:
        base = 0.25
    if mode == "보수적 반영":
        return min(base * 1.25, 1.0) if policy_change_bp > 0 else base * 0.55
    if mode == "직접 입력":
        return 0.0
    return base


def curve_scenario_multiplier(years, scenario):
    """Approximate how a yield-curve scenario changes the selected bond YTM."""
    if scenario in {"평행 이동", "직접 입력"}:
        return 1.0
    if scenario == "단기금리 중심 변화":
        return 1.2 if years <= 3 else 0.65 if years <= 7 else 0.35
    if scenario == "장기금리 중심 변화":
        return 0.55 if years <= 3 else 0.9 if years <= 7 else 1.25
    if scenario == "스티프닝":
        return 0.65 if years <= 3 else 1.0 if years <= 7 else 1.35
    if scenario == "플래트닝":
        return 1.25 if years <= 3 else 0.9 if years <= 7 else 0.55
    return 1.0


def calculate_real_return(nominal_after_tax_return, inflation_rate):
    """Calculate real return from nominal after-tax return and inflation."""
    return (1 + nominal_after_tax_return) / (1 + inflation_rate) - 1


def reinvest_coupons(coupon_rows, final_date, reinvestment_rate):
    """Compound received coupons to the final date."""
    final_ts = pd.Timestamp(final_date)
    total_coupon = 0.0
    reinvested_value = 0.0
    for row in coupon_rows:
        amount = float(row["cashflow"])
        years = max((final_ts - pd.Timestamp(row["date"])).days, 0) / 365
        total_coupon += amount
        reinvested_value += amount * ((1 + reinvestment_rate) ** years)
    return total_coupon, reinvested_value, reinvested_value - total_coupon


def simulate_bond_scenario(bond, sell_date, simulated_ytm, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate):
    """Simulate holding-period proceeds, taxes, and returns for one bond setup."""
    purchase_date = pd.to_datetime(bond["purchase_date"]).date()
    maturity_date = pd.to_datetime(bond["maturity_date"]).date()
    sell_date = min(max(pd.to_datetime(sell_date).date(), purchase_date), maturity_date)
    years_held = max(date_year_fraction(purchase_date, sell_date), 1 / 365)
    years_remaining = date_year_fraction(sell_date, maturity_date)
    trade_info = compute_trade_quantity(
        bond["purchase_amount"],
        bond["current_price"],
        bond["face"],
        bond.get("trade_unit_face", 0.0),
    )
    quantity = trade_info["quantity"]
    investment_base = trade_info["cash_used"] + bond["buy_fee"]
    coupon_payment = bond["face"] * (bond["coupon"] / 100) / max(bond["payments"], 1)
    coupon_rows = [
        {"date": coupon_date, "cashflow": coupon_payment * quantity}
        for coupon_date in coupon_dates_between(purchase_date, maturity_date, bond["payments"])
        if purchase_date < coupon_date <= sell_date
    ]
    coupon_sum, reinvested_coupon_value, reinvestment_income = reinvest_coupons(coupon_rows, sell_date, reinvestment_rate)

    if sell_date >= maturity_date:
        sell_price = bond["face"]
        sell_label = "만기상환"
    else:
        sell_price = price_bond_from_dates(
            bond["face"],
            bond["coupon"],
            sell_date,
            maturity_date,
            bond["payments"],
            simulated_ytm,
        )
        sell_label = "중간매도"

    sell_value = sell_price * quantity
    final_recovery = sell_value + reinvested_coupon_value
    coupon_tax = max(coupon_sum * tax_rate, 0.0)
    capital_gain = max((sell_price - bond["current_price"]) * quantity, 0.0)
    capital_tax = capital_gain * tax_rate if capital_gain_tax else 0.0
    sell_fee = 0.0 if sell_date >= maturity_date else sell_value * sell_fee_rate
    total_fee = bond["buy_fee"] + sell_fee
    gross_profit = final_recovery - trade_info["cash_used"] - bond["buy_fee"]
    after_tax_profit = gross_profit - coupon_tax - capital_tax - sell_fee
    nominal_return = after_tax_profit / investment_base if investment_base else 0.0
    gross_return = gross_profit / investment_base if investment_base else 0.0
    annualized_return = (1 + nominal_return) ** (1 / years_held) - 1 if nominal_return > -0.999 else -1.0
    price_effect = (sell_price - bond["current_price"]) * quantity
    interest_effect = reinvested_coupon_value

    return {
        "sell_date": sell_date,
        "sell_label": sell_label,
        "quantity": quantity,
        "face_total": trade_info["face_total"],
        "cash_used": trade_info["cash_used"],
        "unused_cash": trade_info["unused_cash"],
        "investment_base": investment_base,
        "years_held": years_held,
        "years_remaining": max(years_remaining, 0.0),
        "sell_price": sell_price,
        "price_change_pct": sell_price / bond["current_price"] - 1 if bond["current_price"] else 0.0,
        "sell_value": sell_value,
        "coupon_sum": coupon_sum,
        "reinvested_coupon_value": reinvested_coupon_value,
        "reinvestment_income": reinvestment_income,
        "final_recovery": final_recovery,
        "gross_profit": gross_profit,
        "after_tax_profit": after_tax_profit,
        "gross_return": gross_return,
        "nominal_return": nominal_return,
        "annualized_return": annualized_return,
        "coupon_tax": coupon_tax,
        "capital_tax": capital_tax,
        "sell_fee": sell_fee,
        "total_fee": total_fee,
        "price_effect": price_effect,
        "interest_effect": interest_effect,
        "coupon_rows": coupon_rows,
    }


def simulate_fx_effect(foreign_final_recovery, foreign_profit, foreign_investment, buy_fx_rate, future_fx_rate, fx_fee_rate):
    """Calculate KRW proceeds, FX effect, and FX conversion fee for foreign bonds."""
    krw_recovery_before_fee = foreign_final_recovery * future_fx_rate
    krw_recovery_at_buy_fx = foreign_final_recovery * buy_fx_rate
    fx_effect = krw_recovery_before_fee - krw_recovery_at_buy_fx
    fx_fee = max(krw_recovery_before_fee * fx_fee_rate, 0.0)
    foreign_investment_krw = foreign_investment * buy_fx_rate
    krw_profit = krw_recovery_before_fee - fx_fee - foreign_investment_krw
    return {
        "krw_recovery": krw_recovery_before_fee - fx_fee,
        "krw_profit": krw_profit,
        "fx_effect": fx_effect,
        "fx_fee": fx_fee,
        "foreign_investment_krw": foreign_investment_krw,
    }


def scenario_price_for_bp(bond, sell_date, current_ytm, bp, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate):
    """Run a scenario for a single YTM shift in bp."""
    return simulate_bond_scenario(
        bond,
        sell_date,
        max(current_ytm + bp / 10000, 0.0001),
        tax_rate,
        sell_fee_rate,
        capital_gain_tax,
        reinvestment_rate,
    )


def build_holding_period_rows(bond, current_ytm, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate):
    """Build expected return rows for common holding periods."""
    purchase = pd.to_datetime(bond["purchase_date"]).date()
    maturity = pd.to_datetime(bond["maturity_date"]).date()
    candidates = [
        ("1개월", purchase + pd.DateOffset(months=1)),
        ("3개월", purchase + pd.DateOffset(months=3)),
        ("6개월", purchase + pd.DateOffset(months=6)),
        ("1년", purchase + pd.DateOffset(years=1)),
        ("3년", purchase + pd.DateOffset(years=3)),
        (f"만기({date_year_fraction(purchase, maturity):.2f}년)", maturity),
    ]
    rows = []
    for label, candidate in candidates:
        sell_date = min(pd.Timestamp(candidate).date(), maturity)
        if sell_date <= purchase:
            continue
        result = simulate_bond_scenario(
            bond,
            sell_date,
            current_ytm,
            tax_rate,
            sell_fee_rate,
            capital_gain_tax,
            reinvestment_rate,
        )
        rows.append(
            {
                "보유기간": label,
                "수령 이자": result["coupon_sum"],
                "예상 매도 가격": result["sell_price"],
                "최종 회수금": result["final_recovery"],
                "세후 수익률": result["nominal_return"],
                "연환산 수익률": result["annualized_return"],
            }
        )
    return pd.DataFrame(rows)


def page_simulation():
    """Render the rate change simulation page."""
    render_page_header(
        "금리 변화 시뮬레이션",
        "정책금리, 시장금리, 보유기간, 환율, 물가상승률, 재투자수익률을 가정하여 미래 가격과 수익률을 계산합니다.",
        "시뮬레이션 · 미래 손익",
        "signal",
    )

    bond = get_simulation_bond_defaults()
    exchange_row = get_latest_exchange_rate_value()
    analysis_date = pd.to_datetime(bond["purchase_date"]).date()
    maturity_date = pd.to_datetime(bond["maturity_date"]).date()
    base_years = date_year_fraction(analysis_date, maturity_date)
    unit = bond["currency"]

    with st.container(border=True):
        render_panel_title("시뮬레이션 대상 채권 요약")
        render_value_cards(
            [
                ("채권명", bond["bond_name"], bond["issuer"], ""),
                ("채권 종류", bond["bond_kind"], bond["credit_rating"], ""),
                ("잔존만기", f"{base_years:.2f}년", str(maturity_date), ""),
                ("현재 YTM", f"{bond['current_ytm']:.2f}%", "현재 가격과 쿠폰 기준", "negative" if bond["current_ytm"] < bond["coupon"] else ""),
                ("현재 가격", fmt_unit_money(bond["current_price"], unit, 2), f"액면 {bond['face']:,.0f}", ""),
                ("분석 기준일", str(analysis_date), f"기준 통화: {unit}", ""),
            ]
        )
        st.caption("본 시뮬레이션은 입력값과 단순화된 가정에 따른 참고용 계산입니다. 실제 채권 가격은 시장 유동성, 호가 스프레드, 세금, 수수료, 환율, 신용위험, 거래 가능 여부에 따라 달라질 수 있습니다.")

    input_col, result_col = st.columns([1.05, 1.15])
    with input_col:
        with st.container(border=True):
            render_panel_title("시나리오 입력")
            rate_tab, hold_tab, fx_tab, advanced_tab = st.tabs(["금리", "보유기간/매도", "환율", "고급 가정"])
            with rate_tab:
                st.caption("정책금리는 채권 가격에 직접 적용되는 할인율이 아니며, 시장금리/YTM 변화를 추정하기 위한 참고 변수입니다.")
                kr_policy_bp = st.radio("한국 기준금리 변화", [-50, -25, 0, 25, 50], index=2, horizontal=True, format_func=lambda x: f"{x:+d}bp")
                us_policy_bp = st.radio("미국 기준금리 변화", [-50, -25, 0, 25, 50], index=2, horizontal=True, format_func=lambda x: f"{x:+d}bp")
                reflection_mode = st.radio("시장금리 반영 방식", ["자동 반영", "보수적 반영", "직접 입력"], horizontal=True)
                curve_scenario = st.selectbox("수익률곡선 변화 방식", ["평행 이동", "단기금리 중심 변화", "장기금리 중심 변화", "스티프닝", "플래트닝", "직접 입력"])
                direct_ytm_bp = st.slider("내 채권 YTM 변화폭", -300, 300, 0, step=25, disabled=reflection_mode != "직접 입력" and curve_scenario != "직접 입력")

                selected_policy_bp = us_policy_bp if bond["country"] == "미국채" else kr_policy_bp
                if reflection_mode == "직접 입력" or curve_scenario == "직접 입력":
                    ytm_change_bp = float(direct_ytm_bp)
                else:
                    reflected = selected_policy_bp * policy_reflection_rate(base_years, reflection_mode, selected_policy_bp)
                    ytm_change_bp = clamp(reflected * curve_scenario_multiplier(base_years, curve_scenario), -300, 300)
                simulated_ytm = max(bond["current_ytm"] / 100 + ytm_change_bp / 10000, 0.0001)
                st.info(f"계산 적용 YTM 변화폭: {fmt_bp(ytm_change_bp)} · 시뮬레이션 YTM {simulated_ytm * 100:.2f}%")

            with hold_tab:
                holding_mode = st.radio("보유 방식", ["만기 보유", "중간 매도"], horizontal=True)
                quick_period = st.radio("보유기간 빠른 선택", ["1개월", "3개월", "6개월", "1년", "3년", "직접 입력"], index=3, horizontal=True, disabled=holding_mode == "만기 보유")
                default_sell_date = min((pd.Timestamp(analysis_date) + pd.DateOffset(years=1)).date(), maturity_date)
                quick_offsets = {
                    "1개월": pd.DateOffset(months=1),
                    "3개월": pd.DateOffset(months=3),
                    "6개월": pd.DateOffset(months=6),
                    "1년": pd.DateOffset(years=1),
                    "3년": pd.DateOffset(years=3),
                }
                if holding_mode == "만기 보유":
                    sell_date = maturity_date
                    st.text_input("예상 매도/평가일", value=str(maturity_date), disabled=True)
                elif quick_period == "직접 입력":
                    sell_date = st.date_input("예상 매도일", value=default_sell_date, min_value=analysis_date, max_value=maturity_date, key="sim_sell_date")
                else:
                    sell_date = min((pd.Timestamp(analysis_date) + quick_offsets[quick_period]).date(), maturity_date)
                    st.text_input("예상 매도일", value=str(sell_date), disabled=True)
                if sell_date <= analysis_date:
                    st.warning("매도일은 분석 기준일 이후여야 합니다.")

            with fx_tab:
                is_foreign = bond["country"] == "미국채"
                if not is_foreign:
                    st.info("원화채는 환율 시나리오가 손익에 직접 반영되지 않습니다.")
                if "sim_current_fx_rate" not in st.session_state:
                    st.session_state["sim_current_fx_rate"] = float(exchange_row["rate"])
                if "sim_buy_fx_rate" not in st.session_state:
                    st.session_state["sim_buy_fx_rate"] = float(exchange_row["rate"])
                fx_button_cols = st.columns([0.72, 0.28])
                with fx_button_cols[0]:
                    st.caption(f"최근 BOK 원/달러 환율: {fmt_fx_rate(exchange_row['rate'])} · {pd.to_datetime(exchange_row['date']).date()} · {exchange_row['source']}")
                with fx_button_cols[1]:
                    if st.button("현재 환율 불러오기", use_container_width=True, disabled=not is_foreign):
                        st.session_state["sim_current_fx_rate"] = float(exchange_row["rate"])
                buy_fx_rate = st.number_input("매수 환율", min_value=1.0, step=1.0, disabled=not is_foreign, key="sim_buy_fx_rate")
                current_fx_rate = st.number_input("현재 환율", min_value=1.0, step=1.0, disabled=not is_foreign, key="sim_current_fx_rate")
                fx_quick = st.radio("미래 환율 시나리오", ["-10%", "-5%", "현재", "+5%", "+10%", "직접 입력"], index=3, horizontal=True, disabled=not is_foreign)
                if fx_quick == "직접 입력":
                    if "sim_future_fx_rate" not in st.session_state:
                        st.session_state["sim_future_fx_rate"] = float(current_fx_rate)
                    future_fx_rate = st.number_input("미래 환율", min_value=1.0, step=1.0, disabled=not is_foreign, key="sim_future_fx_rate")
                else:
                    fx_multiplier = {"-10%": 0.9, "-5%": 0.95, "현재": 1.0, "+5%": 1.05, "+10%": 1.1}[fx_quick]
                    future_fx_rate = current_fx_rate * fx_multiplier
                    st.text_input("미래 환율", value=f"{future_fx_rate:,.1f} 원/USD", disabled=True)
                fx_fee_rate = st.number_input("환전 수수료율(%)", min_value=0.0, max_value=5.0, value=0.2 if is_foreign else 0.0, step=0.05, disabled=not is_foreign) / 100

            with advanced_tab:
                inflation_rate = st.slider("예상 연평균 물가상승률(%)", 0.0, 10.0, 2.0, step=0.1) / 100
                st.caption("실질수익률은 명목수익률에서 물가상승률을 반영한 구매력 기준 수익률입니다.")
                reinvestment_mode = st.selectbox("쿠폰 재투자 방식", ["재투자 없음", "현재 단기금리로 재투자", "사용자가 직접 입력", "시나리오 금리와 동일하게 적용"], disabled=bond["coupon"] <= 0)
                if reinvestment_mode == "사용자가 직접 입력":
                    reinvestment_rate = st.number_input("쿠폰 재투자수익률(%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1) / 100
                elif reinvestment_mode == "현재 단기금리로 재투자":
                    reinvestment_rate = min(max(bond["current_ytm"] / 100 * 0.55, 0.0), 0.1)
                    st.text_input("적용 재투자수익률", value=f"{reinvestment_rate * 100:.2f}%", disabled=True)
                elif reinvestment_mode == "시나리오 금리와 동일하게 적용":
                    reinvestment_rate = min(max(simulated_ytm, 0.0), 0.1)
                    st.text_input("적용 재투자수익률", value=f"{reinvestment_rate * 100:.2f}%", disabled=True)
                else:
                    reinvestment_rate = 0.0
                st.caption("재투자 시나리오는 쿠폰 이자를 매도일 또는 만기일까지 다시 투자한다고 가정한 계산입니다.")
                with st.expander("세금/수수료 상세 설정"):
                    tax_rate = st.slider("이자소득세율(%)", 0.0, 40.0, 15.4, step=0.1) / 100
                    sell_fee_rate = st.number_input("매도 수수료율(%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05) / 100
                    capital_gain_tax = st.checkbox("매매차익에도 같은 세율 적용", value=False)

    try:
        result = simulate_bond_scenario(bond, sell_date, simulated_ytm, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate)
        maturity_result = simulate_bond_scenario(bond, maturity_date, simulated_ytm, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate)
    except Exception as exc:
        st.error(f"시뮬레이션 계산 중 오류가 발생했습니다: {exc}")
        return

    if bond["country"] == "미국채":
        fx_result = simulate_fx_effect(
            result["final_recovery"],
            result["after_tax_profit"],
            result["investment_base"],
            buy_fx_rate,
            future_fx_rate,
            fx_fee_rate,
        )
        krw_profit = fx_result["krw_profit"]
    else:
        fx_result = {"krw_recovery": result["final_recovery"], "krw_profit": result["after_tax_profit"], "fx_effect": 0.0, "fx_fee": 0.0, "foreign_investment_krw": bond["purchase_amount"]}
        krw_profit = result["after_tax_profit"]

    real_return = calculate_real_return(result["nominal_return"], inflation_rate)
    purchasing_power_recovery = result["final_recovery"] / ((1 + inflation_rate) ** result["years_held"])
    maturity_gap = result["after_tax_profit"] - maturity_result["after_tax_profit"]
    judgment = "금리 부담을 상쇄" if krw_profit >= 0 and ytm_change_bp >= 0 else "손익 변동 확인 필요" if krw_profit >= 0 else "손실 가능성 주의"

    with result_col:
        with st.container(border=True):
            render_panel_title("시나리오 결과 요약")
            render_value_cards(
                [
                    ("예상 평가/매도 가격", fmt_unit_money(result["sell_price"], unit, 2), f"{result['price_change_pct'] * 100:+.2f}%", "positive" if result["price_change_pct"] >= 0 else "negative"),
                    ("총 수령 이자", fmt_unit_money(result["coupon_sum"], unit, 0), f"재투자 수익 {fmt_unit_money(result['reinvestment_income'], unit, 0)}", ""),
                    ("최종 회수금", fmt_unit_money(result["final_recovery"], unit, 0), result["sell_label"], ""),
                    ("세전 손익", fmt_signed_money(result["gross_profit"], unit), f"{result['gross_return'] * 100:+.2f}%", "positive" if result["gross_profit"] >= 0 else "negative"),
                    ("세후 손익", fmt_signed_money(result["after_tax_profit"], unit), f"{result['nominal_return'] * 100:+.2f}%", "positive" if result["after_tax_profit"] >= 0 else "negative"),
                    ("원화 기준 손익", fmt_signed_money(krw_profit, "원"), "환율 반영" if bond["country"] == "미국채" else "원화채", "positive" if krw_profit >= 0 else "negative"),
                    ("세후 명목수익률", f"{result['nominal_return'] * 100:.2f}%", f"연환산 {result['annualized_return'] * 100:.2f}%", ""),
                    ("세후 실질수익률", f"{real_return * 100:.2f}%", f"물가 {inflation_rate * 100:.1f}% 가정", "positive" if real_return >= 0 else "negative"),
                ]
            )
            breakdown_items = [
                ("채권 가격 효과", fmt_signed_money(result["price_effect"], unit), result["price_effect"]),
                ("이자 수익", fmt_signed_money(result["interest_effect"], unit), result["interest_effect"]),
                ("환율 효과", fmt_signed_money(fx_result["fx_effect"], "원"), fx_result["fx_effect"]),
                ("수수료 효과", fmt_signed_money(-(result["sell_fee"] + fx_result["fx_fee"]), "원" if bond["country"] == "미국채" else unit), -(result["sell_fee"] + fx_result["fx_fee"])),
            ]
            breakdown_cards = "".join(
                '<div class="breakdown-card">'
                f'<div class="breakdown-label">{html.escape(label)}</div>'
                f'<div class="breakdown-value {"positive" if numeric > 0 else "negative" if numeric < 0 else ""}">{html.escape(value)}</div>'
                "</div>"
                for label, value, numeric in breakdown_items
            )
            st.markdown(f'<div class="breakdown-grid">{breakdown_cards}</div>', unsafe_allow_html=True)
            st.success(f"현재 시나리오 판단: {judgment}")
            st.caption(f"만기 보유 대비 차이: {fmt_signed_money(maturity_gap, unit)} · 구매력 기준 최종 회수금: {fmt_unit_money(purchasing_power_recovery, unit, 0)}")

    ytm_bps = [-200, -100, -50, -25, 0, 25, 50, 100, 200]
    ytm_rows = []
    price_chart_rows = []
    for bp in range(-300, 301, 25):
        scenario = scenario_price_for_bp(bond, sell_date, bond["current_ytm"] / 100, bp, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate)
        price_chart_rows.append({"YTM 변화폭(bp)": bp, "예상 가격": scenario["sell_price"], "현재 시나리오": abs(bp - ytm_change_bp) < 0.01})
    if not any(row["현재 시나리오"] for row in price_chart_rows):
        scenario = scenario_price_for_bp(bond, sell_date, bond["current_ytm"] / 100, ytm_change_bp, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate)
        price_chart_rows.append({"YTM 변화폭(bp)": ytm_change_bp, "예상 가격": scenario["sell_price"], "현재 시나리오": True})
        price_chart_rows = sorted(price_chart_rows, key=lambda row: row["YTM 변화폭(bp)"])
    for bp in ytm_bps:
        scenario = scenario_price_for_bp(bond, sell_date, bond["current_ytm"] / 100, bp, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate)
        ytm_rows.append(
            {
                "YTM 변화": "현재" if bp == 0 else f"{bp:+d}bp",
                "예상 가격": scenario["sell_price"],
                "가격 변화율": scenario["price_change_pct"],
                "평가손익": scenario["price_effect"],
                "세후 손익": scenario["after_tax_profit"],
            }
        )

    holding_df = build_holding_period_rows(bond, simulated_ytm, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate)
    fx_rows = []
    for label, multiplier in [("-10%", 0.9), ("-5%", 0.95), ("현재", 1.0), ("+5%", 1.05), ("+10%", 1.1)]:
        scenario_fx = simulate_fx_effect(
            result["final_recovery"],
            result["after_tax_profit"],
            result["investment_base"],
            buy_fx_rate,
            current_fx_rate * multiplier,
            fx_fee_rate,
        )
        fx_rows.append(
            {
                "환율 변화": label,
                "미래 환율": current_fx_rate * multiplier,
                "원화 회수금": scenario_fx["krw_recovery"],
                "환율 효과": scenario_fx["fx_effect"],
                "총 손익": scenario_fx["krw_profit"],
            }
        )
    fx_df = pd.DataFrame(fx_rows)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        with st.container(border=True):
            render_panel_title("금리 변화별 예상 가격")
            price_df = pd.DataFrame(price_chart_rows)
            price_fig = px.line(price_df, x="YTM 변화폭(bp)", y="예상 가격", markers=True)
            price_fig.add_vline(x=ytm_change_bp, line_dash="dot", line_color="#1552c7")
            price_fig.update_traces(line_color="#1552c7", marker=dict(color="#1552c7", size=6))
            price_fig.update_layout(height=300, margin=dict(l=8, r=8, t=12, b=8), paper_bgcolor="white", plot_bgcolor="white")
            price_fig.update_xaxes(showgrid=False)
            price_fig.update_yaxes(gridcolor="#e8eef7", title=f"예상 가격({unit})")
            st.plotly_chart(price_fig, width="stretch")
    with chart_col2:
        with st.container(border=True):
            render_panel_title("환율 변화별 원화 손익")
            if bond["country"] == "미국채":
                fx_fig = px.line(fx_df, x="미래 환율", y="총 손익", markers=True)
                fx_fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
                fx_fig.add_vline(x=future_fx_rate, line_dash="dot", line_color="#1552c7")
                fx_fig.update_traces(line_color="#1552c7", marker=dict(color="#1552c7", size=7))
                fx_fig.update_layout(height=300, margin=dict(l=8, r=8, t=12, b=8), paper_bgcolor="white", plot_bgcolor="white")
                fx_fig.update_xaxes(showgrid=False, title="미래 환율")
                fx_fig.update_yaxes(gridcolor="#e8eef7", title="원화 손익")
                st.plotly_chart(fx_fig, width="stretch")
            else:
                st.info("원화채는 환율 변화별 손익 차트를 생략합니다.")

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        with st.container(border=True):
            render_panel_title("보유기간별 총수익률")
            hold_chart_df = holding_df.copy()
            hold_chart_df["세후 수익률(%)"] = hold_chart_df["세후 수익률"] * 100
            hold_fig = px.line(hold_chart_df, x="보유기간", y="세후 수익률(%)", markers=True)
            hold_fig.update_traces(line_color="#1552c7", marker=dict(color="#1552c7", size=7))
            hold_fig.update_layout(height=300, margin=dict(l=8, r=8, t=12, b=8), paper_bgcolor="white", plot_bgcolor="white")
            hold_fig.update_yaxes(gridcolor="#e8eef7")
            st.plotly_chart(hold_fig, width="stretch")
    with chart_col4:
        with st.container(border=True):
            render_panel_title("시나리오별 결과 비교")
            comparison = pd.DataFrame(
                [
                    ("기본", result["after_tax_profit"]),
                    ("금리 상승", scenario_price_for_bp(bond, sell_date, bond["current_ytm"] / 100, 100, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate)["after_tax_profit"]),
                    ("금리 하락", scenario_price_for_bp(bond, sell_date, bond["current_ytm"] / 100, -100, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate)["after_tax_profit"]),
                    ("보수적", scenario_price_for_bp(bond, sell_date, bond["current_ytm"] / 100, 150, tax_rate, sell_fee_rate, capital_gain_tax, 0.0)["after_tax_profit"]),
                    ("낙관적", scenario_price_for_bp(bond, sell_date, bond["current_ytm"] / 100, -75, tax_rate, sell_fee_rate, capital_gain_tax, reinvestment_rate)["after_tax_profit"]),
                ],
                columns=["시나리오", "세후 손익"],
            )
            comp_fig = px.bar(comparison, x="시나리오", y="세후 손익", color="세후 손익", color_continuous_scale=["#fee2e2", "#eaf2ff", "#dcfce7"])
            comp_fig.update_layout(height=300, margin=dict(l=8, r=8, t=12, b=8), paper_bgcolor="white", plot_bgcolor="white", showlegend=False, coloraxis_showscale=False)
            comp_fig.update_yaxes(gridcolor="#e8eef7")
            st.plotly_chart(comp_fig, width="stretch")

    table_col1, table_col2 = st.columns([1.05, 1.05])
    with table_col1:
        with st.container(border=True):
            render_panel_title("YTM 변화별 가격 민감도")
            ytm_df = pd.DataFrame(ytm_rows)
            st.dataframe(
                ytm_df.style.format(
                    {
                        "예상 가격": "{:,.2f}",
                        "가격 변화율": "{:+.2%}",
                        "평가손익": "{:+,.0f}",
                        "세후 손익": "{:+,.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
    with table_col2:
        with st.container(border=True):
            render_panel_title("보유기간별 예상 수익률")
            st.dataframe(
                holding_df.style.format(
                    {
                        "수령 이자": "{:,.0f}",
                        "예상 매도 가격": "{:,.2f}",
                        "최종 회수금": "{:,.0f}",
                        "세후 수익률": "{:.2%}",
                        "연환산 수익률": "{:.2%}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    with st.container(border=True):
        render_panel_title("환율 변화별 원화 손익")
        if bond["country"] == "미국채":
            st.dataframe(
                fx_df.style.format(
                    {
                        "미래 환율": "{:,.1f}",
                        "원화 회수금": "{:,.0f}",
                        "환율 효과": "{:+,.0f}",
                        "총 손익": "{:+,.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("원화채는 환율 변화별 원화 손익 테이블을 생략합니다.")

    with st.expander("상세 현금흐름 표 보기"):
        cashflow_detail = pd.DataFrame(result["coupon_rows"])
        if cashflow_detail.empty:
            st.info("매도/만기일까지 수령하는 쿠폰이 없습니다.")
        else:
            cashflow_detail = cashflow_detail.rename(columns={"date": "수령일", "cashflow": "쿠폰 현금흐름"})
            st.dataframe(cashflow_detail.style.format({"쿠폰 현금흐름": "{:,.0f}"}), width="stretch", hide_index=True)

    with st.expander("계산 가정 및 수식 보기"):
        st.markdown(
            f"""
            - 정책금리는 직접 할인율이 아니며, 선택한 반영 방식에 따라 YTM 변화폭 `{fmt_bp(ytm_change_bp)}`로 변환했습니다.
            - 적용 YTM은 `{simulated_ytm * 100:.2f}%`입니다.
            - 중간 매도 가격은 매도일 이후 남은 현금흐름을 적용 YTM으로 할인해 계산했습니다.
            - 만기 보유는 원금상환금과 수령 쿠폰을 합산합니다.
            - 실질수익률 = `(1 + 세후 명목수익률) / (1 + 물가상승률) - 1`입니다.
            - 쿠폰 세금은 수령 쿠폰에 세율을 곱해 계산하고, 매매차익 과세는 선택한 경우에만 반영합니다.
            """
        )


def load_forward_raw_data():
    """Return the standard forward-rate calculation table for raw-data debugging."""
    kr_forward = calculate_forward_rates_from_curve(cached_kr_curve()).assign(market="한국 국고채")
    us_forward = calculate_forward_rates_from_curve(cached_us_curve()).assign(market="미국채")
    return pd.concat([kr_forward, us_forward], ignore_index=True)


def page_raw_data():
    """Render a debugging page for major raw datasets."""
    render_page_header(
        "디버깅용 Raw Data",
        "홈 화면과 계산에 사용되는 주요 원천 데이터를 확인하는 개발용 화면입니다.",
        "데이터 · 원천 확인",
        "news",
    )

    datasets = {
        "국내 주요 금리 지표": cached_macro_rates,
        "한/미 정책금리": cached_policy_rates,
        "원/달러 환율": cached_exchange_rate,
        "정책 일정": cached_policy_calendar,
        "한/미 중앙은행 공식 자료": cached_central_bank_materials,
        "한국 국고채 수익률곡선": cached_kr_curve,
        "미국채 수익률곡선": cached_us_curve,
        "한국 국고채 금리 히스토리": cached_bond_yield_history,
        "미국채 금리 히스토리": cached_us_history,
        "정책금리 관련 뉴스": cached_policy_news,
        "채권시장 참고 뉴스": cached_bond_news,
        "Forward rate 계산용 데이터": load_forward_raw_data,
    }
    selected = st.selectbox("Raw data 선택", list(datasets.keys()))
    df = datasets[selected]()
    if selected == "정책금리 관련 뉴스":
        df = classify_news_dataframe(df)
    st.dataframe(make_raw_display_df(df), width="stretch", hide_index=True)


def main():
    """Run the Streamlit app."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-brand-main">
            <div class="sidebar-logo">↗</div>
            <div class="sidebar-brand-title">개인 채권<br>가격 분석 시뮬레이터</div>
          </div>
          <div class="sidebar-brand-sub">금리 환경, 채권 가격, 미래 손익을 한 화면 흐름으로 점검합니다.</div>
        </div>
        <div class="sidebar-section-label">PAGE</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "페이지 선택",
        ["홈: 거시 금리 환경", "채권 입력 및 기본 분석", "금리 변화 시뮬레이션"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-label">DEVELOPER</div>', unsafe_allow_html=True)
    show_raw_data = st.sidebar.toggle("디버깅용 raw data", value=False)
    st.sidebar.markdown(
        """
        <div class="sidebar-footer-card">
          데이터 출처: BOK ECOS, FRED, 네이버 뉴스 검색<br>
          계산 결과는 참고용이며 실제 투자 판단을 대체하지 않습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_raw_data:
        page_raw_data()
    elif page == "홈: 거시 금리 환경":
        page_home()
    elif page == "채권 입력 및 기본 분석":
        page_basic_analysis()
    else:
        page_simulation()


if __name__ == "__main__":
    main()
