"""News crawling helpers for supplementary rate-environment context."""

from datetime import date
import time
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_sample_news(category="bond"):
    """Return sample news rows when crawling is unavailable."""
    today = date.today().isoformat()
    if category == "policy":
        rows = [
            ("물가 둔화에도 금통위 기준금리 동결 관망", "sample", today, "", "물가"),
            ("FOMC 앞두고 미국 기준금리 인하 기대 약화", "sample", today, "", "FOMC"),
            ("환율 상승에 통화정책 신중론 확대", "sample", today, "", "환율"),
            ("고용 호조로 연준 매파 발언 주목", "sample", today, "", "고용지표"),
            ("경제 전망 불확실성에 속도 조절 필요", "sample", today, "", "경제 전망"),
        ]
    else:
        rows = [
            ("국고채 금리 상승, 장기물 중심 약세", "sample", today, "", "국고채 금리"),
            ("채권시장, FOMC 앞두고 변동성 확대", "sample", today, "", "채권시장"),
            ("회사채 금리 스프레드 소폭 확대", "sample", today, "", "회사채 금리"),
            ("미국채 10년 금리 상승에 국내 금리도 영향", "sample", today, "", "미국채 10년"),
            ("수익률곡선 완만한 스티프닝 흐름", "sample", today, "", "수익률곡선"),
        ]
    return pd.DataFrame(rows, columns=["title", "source", "date", "link", "query"]).assign(category=category)


def _clean_text(text):
    """Normalize whitespace in crawled text."""
    return " ".join(str(text).split())


def _news_headers():
    """Return browser-like headers for Naver search requests."""
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.naver.com/",
    }


def _extract_news_rows(soup, query, category, max_items):
    """Extract news rows from several possible Naver result structures."""
    rows = []
    seen_links = set()
    query_terms = [term for term in query.split() if len(term) >= 2]
    selectors = [
        "a.news_tit",
        "a.news_link",
        "a[class*='news_tit']",
        "a[href]",
    ]

    for selector in selectors:
        for item in soup.select(selector):
            title = _clean_text(item.get("title") or item.get_text(" ", strip=True))
            link = item.get("href", "")
            if title in {"네이버뉴스", "뉴스"} or title.startswith("언론사 선정"):
                continue
            if len(title) < 8 or len(title) > 120:
                continue
            if query_terms and not any(term in title for term in query_terms):
                continue
            if not title or not link or link in seen_links:
                continue
            container = item.find_parent(["li", "div", "section", "article"])
            source = "네이버뉴스"
            date_text = date.today().isoformat()
            if container:
                source_node = container.select_one(".press, .info.press, [class*='press'], [class*='Press']")
                info_nodes = container.select(".info_group span.info, [class*='date'], [class*='Date']")
                if source_node:
                    source = _clean_text(source_node.get_text(" ", strip=True))
                if info_nodes:
                    date_text = _clean_text(info_nodes[-1].get_text(" ", strip=True))
            rows.append(
                {
                    "title": title,
                    "source": source,
                    "date": date_text,
                    "link": link,
                    "query": query,
                    "category": category,
                }
            )
            seen_links.add(link)
            if len(rows) >= max_items:
                return rows
    return rows


def fetch_news(query, max_items=10, category="news", use_sample_fallback=True):
    """Fetch Naver News search titles for one query."""
    url = f"https://search.naver.com/search.naver?where=news&sm=tab_jum&query={quote_plus(query)}"
    headers = {
        **_news_headers(),
    }
    try:
        response = requests.get(url, headers=headers, timeout=6)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        rows = _extract_news_rows(soup, query, category, max_items)
        if not rows:
            return get_sample_news(category).head(max_items) if use_sample_fallback else pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception:
        return get_sample_news(category).head(max_items) if use_sample_fallback else pd.DataFrame()


def fetch_bond_news(max_items=10):
    """Fetch recent bond-market reference news."""
    queries = ["국고채 금리", "채권시장", "회사채 금리", "기준금리", "장단기 금리차", "수익률곡선", "미국채 금리", "미국채 10년"]
    frames = []
    per_query = max(1, (max_items + min(len(queries), max_items) - 1) // min(len(queries), max_items))
    for query in queries:
        frames.append(fetch_news(query, per_query, category="bond", use_sample_fallback=False))
        time.sleep(0.2)
        if sum(len(frame) for frame in frames) >= max_items:
            break
    real_frames = [frame for frame in frames if not frame.empty]
    if real_frames:
        df = pd.concat(real_frames, ignore_index=True)
    else:
        df = get_sample_news("bond")
    df["category"] = "bond"
    return df.head(max_items)


def fetch_policy_rate_news(max_items=10):
    """Fetch recent policy-rate indicator and schedule news."""
    queries = [
        "소비자물가지수 CPI",
        "물가",
        "고용지표",
        "실업률",
        "GDP 성장률",
        "환율",
        "FOMC",
        "한국은행 금통위",
        "기준금리 결정",
        "물가 전망",
        "경제 전망",
        "양적완화",
        "QT",
        "대차대조표 축소",
        "포워드 가이던스",
        "미국 기준금리",
        "연준 금리",
    ]
    frames = []
    per_query = max(1, (max_items + min(len(queries), max_items) - 1) // min(len(queries), max_items))
    for query in queries:
        frames.append(fetch_news(query, per_query, category="policy", use_sample_fallback=False))
        time.sleep(0.2)
        if sum(len(frame) for frame in frames) >= max_items:
            break
    real_frames = [frame for frame in frames if not frame.empty]
    if real_frames:
        df = pd.concat(real_frames, ignore_index=True)
    else:
        df = get_sample_news("policy")
    df["category"] = "policy"
    return df.head(max_items)
