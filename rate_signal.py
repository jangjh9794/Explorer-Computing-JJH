"""News issue tagging helpers for policy-rate context."""

from collections import Counter

import pandas as pd


REGION_KEYWORDS = {
    "한국": [
        "한국은행",
        "한은",
        "금통위",
        "이창용",
        "한국",
        "국내",
        "원화",
        "원/달러",
        "국고채",
        "기재부",
        "코스피",
    ],
    "미국": [
        "미국",
        "美",
        "연준",
        "fed",
        "fomc",
        "파월",
        "미국채",
        "미 국채",
        "달러",
        "pce",
        "비농업",
        "실업수당",
    ],
    "기타": [
        "일본",
        "日",
        "일본은행",
        "boj",
        "ecb",
        "유럽",
        "유로존",
        "중국",
        "인민은행",
        "영국",
        "캐나다",
        "호주",
        "엔화",
        "유로",
    ],
}

QUERY_REGION_HINTS = {
    "FOMC": "미국",
    "연준": "미국",
    "미국": "미국",
    "미국채": "미국",
    "한국은행": "한국",
    "금통위": "한국",
    "국고채": "한국",
}

ISSUE_TAG_KEYWORDS = {
    "물가": ["cpi", "pce", "소비자물가", "물가", "인플레이션", "기대인플레이션", "디플레이션"],
    "고용": ["고용", "실업률", "실업수당", "비농업", "일자리", "임금"],
    "환율": ["환율", "원화", "원/달러", "달러", "엔화", "유로", "원 약세", "원 강세", "달러 강세", "달러 약세"],
    "성장": ["gdp", "성장률", "성장", "경기", "침체", "둔화", "소비", "내수", "수출"],
    "국채금리": ["국채금리", "국고채", "미국채", "장기금리", "수익률", "수익률곡선", "장단기금리차"],
    "기준금리": ["기준금리", "정책금리", "금리 결정", "금리 인상", "금리 인하", "동결"],
    "회의": ["fomc", "금통위", "회의록", "의사록", "기자간담회"],
    "중앙은행 발언": ["파월", "이창용", "연준", "한국은행", "한은", "fed", "총재", "위원"],
    "유가": ["유가", "원유", "브렌트", "wti"],
}

ISSUE_NOTES = {
    "국채금리": "국채금리 관련 뉴스가 많아 채권 가격 민감도와 YTM 변화 시나리오를 함께 확인하는 것이 좋습니다.",
    "환율": "환율 관련 뉴스가 많아 미국채 투자자는 원화 환산 손익을 함께 점검하는 것이 좋습니다.",
    "물가": "물가 관련 뉴스가 많아 향후 정책금리 경로에 대한 시장 관심이 높아진 상태로 볼 수 있습니다.",
    "고용": "고용 관련 뉴스가 많아 경기와 금리 전망이 함께 흔들릴 수 있습니다.",
    "성장": "경기 관련 뉴스가 많아 금리 방향보다 경기 둔화 리스크를 함께 확인하는 것이 좋습니다.",
    "회의": "FOMC·금통위 일정 관련 뉴스가 많아 단기 변동성 확대 가능성을 점검하는 것이 좋습니다.",
    "기준금리": "기준금리 관련 뉴스가 많아 중앙은행 정책 결정과 시장금리 반응을 함께 확인하는 것이 좋습니다.",
    "중앙은행 발언": "중앙은행 인사 발언 관련 뉴스가 많아 공식 발언과 시장 해석의 차이를 확인하는 것이 좋습니다.",
    "유가": "유가 관련 뉴스가 많아 물가와 금리 전망에 미칠 수 있는 영향을 함께 살펴보는 것이 좋습니다.",
    "기타": "여러 이슈가 섞여 있어 뉴스 제목만으로 방향성을 판단하기보다 원문과 공식 자료를 함께 확인하는 것이 좋습니다.",
}


def _contains(text, keywords):
    """Return matched keywords in text."""
    lower_text = str(text).lower()
    return [keyword for keyword in keywords if keyword.lower() in lower_text]


def detect_news_region(title, query=""):
    """Classify a news title into Korea, US, other, or global context."""
    text = f"{title} {query}"
    kr_matches = _contains(text, REGION_KEYWORDS["한국"])
    us_matches = _contains(text, REGION_KEYWORDS["미국"])
    other_matches = _contains(text, REGION_KEYWORDS["기타"])

    if other_matches and not (kr_matches or us_matches):
        return "기타", other_matches
    if kr_matches and not us_matches:
        return "한국", kr_matches
    if us_matches and not kr_matches:
        return "미국", us_matches
    if kr_matches and us_matches:
        return "한/미", list(dict.fromkeys(kr_matches + us_matches))

    for hint, region in QUERY_REGION_HINTS.items():
        if hint.lower() in str(query).lower():
            return region, [hint]
    if other_matches:
        return "기타", other_matches
    return "글로벌", []


def extract_issue_tags(title):
    """Extract simple issue tags from a news title."""
    tags = []
    matched = []
    for tag, keywords in ISSUE_TAG_KEYWORDS.items():
        hits = _contains(title, keywords)
        if hits:
            tags.append(tag)
            matched.extend(hits)
    return tags or ["기타"], list(dict.fromkeys(matched))


def classify_news_item(title, query=""):
    """Tag one news title by region and issue category."""
    region, region_keywords = detect_news_region(title, query)
    issue_tags, issue_keywords = extract_issue_tags(title)
    return {
        "region": region,
        "issue_tags": ", ".join(issue_tags),
        "primary_issue": issue_tags[0] if issue_tags else "기타",
        "matched_keywords": ", ".join(dict.fromkeys(issue_keywords + region_keywords)),
    }


def classify_news_dataframe(news_df):
    """Add issue-tagging columns to a news DataFrame."""
    base_columns = [
        "title",
        "source",
        "date",
        "query",
        "region",
        "issue_tags",
        "primary_issue",
        "matched_keywords",
    ]
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=base_columns)

    result = news_df.copy()
    if "query" not in result.columns:
        result["query"] = ""
    classified = result.apply(
        lambda row: classify_news_item(row.get("title", ""), row.get("query", "")),
        axis=1,
    ).apply(pd.Series)
    return pd.concat([result.reset_index(drop=True), classified.reset_index(drop=True)], axis=1)


def _count_issue_tags(series):
    """Count comma-separated issue tags."""
    counter = Counter()
    for tags in series.fillna("기타"):
        for tag in str(tags).split(","):
            cleaned = tag.strip()
            if cleaned:
                counter[cleaned] += 1
    return pd.DataFrame(counter.most_common(), columns=["issue_tag", "count"])


def summarize_news_flow(news_df):
    """Summarize recent news into region counts, issue frequency, and checkpoints."""
    classified = classify_news_dataframe(news_df)
    if classified.empty:
        return {
            "classified_df": classified,
            "region_counts": pd.DataFrame(columns=["region", "count"]),
            "tag_counts": pd.DataFrame(columns=["issue_tag", "count"]),
            "top_issues": [],
            "top_issue": "데이터 없음",
            "issue_notes": ["수집된 뉴스가 없어 이슈 체크포인트를 만들 수 없습니다."],
            "summary_text": "수집된 뉴스가 없어 최근 시장 관심 이슈를 요약할 수 없습니다.",
        }

    region_counts = classified["region"].value_counts().rename_axis("region").reset_index(name="count")
    tag_counts = _count_issue_tags(classified["issue_tags"])
    top_issues = tag_counts.head(3)["issue_tag"].tolist() if not tag_counts.empty else ["데이터 없음"]
    issue_notes = [ISSUE_NOTES.get(issue, ISSUE_NOTES["기타"]) for issue in top_issues]
    issue_text = ", ".join(top_issues)
    summary_text = f"최근 뉴스 제목에서는 {issue_text} 이슈가 상대적으로 많이 언급됩니다."
    return {
        "classified_df": classified,
        "region_counts": region_counts,
        "tag_counts": tag_counts,
        "top_issues": top_issues,
        "top_issue": top_issues[0] if top_issues else "데이터 없음",
        "issue_notes": issue_notes,
        "summary_text": summary_text,
    }
