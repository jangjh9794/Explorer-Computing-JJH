"""Small visual asset registry for the Streamlit dashboard."""


ASSETS = {
    "kr_flag": {
        "src": "https://flagcdn.com/kr.svg",
        "alt": "한국",
    },
    "us_flag": {
        "src": "https://flagcdn.com/us.svg",
        "alt": "미국",
    },
    "rate": {
        "src": "https://api.iconify.design/lucide:activity.svg?color=%231552c7",
        "alt": "금리",
    },
    "calendar": {
        "src": "https://api.iconify.design/lucide:calendar-days.svg?color=%231552c7",
        "alt": "일정",
    },
    "chart": {
        "src": "https://api.iconify.design/lucide:line-chart.svg?color=%231552c7",
        "alt": "차트",
    },
    "news": {
        "src": "https://api.iconify.design/lucide:newspaper.svg?color=%231552c7",
        "alt": "뉴스",
    },
    "signal": {
        "src": "https://api.iconify.design/lucide:gauge.svg?color=%231552c7",
        "alt": "지표",
    },
    "exchange": {
        "src": "https://api.iconify.design/lucide:circle-dollar-sign.svg?color=%231552c7",
        "alt": "환율",
    },
}


def asset_img_html(asset_key, class_name="asset-icon", width=22, height=22):
    """Return an HTML image tag for a registered visual asset."""
    asset = ASSETS.get(asset_key)
    if not asset:
        return ""
    return (
        f'<img class="{class_name}" src="{asset["src"]}" alt="{asset["alt"]}" '
        f'width="{width}" height="{height}" loading="lazy" />'
    )


def country_asset_key(country):
    """Return the flag asset key for a country label."""
    return "kr_flag" if country in {"한국", "한국채", "한국 국고채"} else "us_flag"


def country_label_html(country, label, class_name="title-row"):
    """Return a flag image plus text label for dashboard headings."""
    return f'<span class="{class_name}">{asset_img_html(country_asset_key(country))}<span>{label}</span></span>'
