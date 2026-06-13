# 개인 채권 가격 분석 시뮬레이터

대학 교양 과목 기말 프로젝트용 Streamlit 웹앱입니다. 국내외 금리 환경을 확인하고, 사용자가 입력한 채권 조건으로 현재가격, 평가금액, 예상 이자수입, 듀레이션, 볼록성, 금리 변화 손익을 계산합니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

API 키가 없어도 sample 데이터로 앱이 실행됩니다.

## 주요 기능

- 한국 주요 금리 지표와 국고채 수익률곡선 확인
- FRED API 기반 미국채 수익률곡선과 주요 만기 금리 확인
- 한/미 정책금리와 다음 금통위/FOMC 일정 표시
- 사용자가 선택한 시장과 기간으로 Forward Rate 직접 계산
- 한국채/미국채 선택 후 가격, 평가금액, 쿠폰 이자수입 계산
- Macaulay Duration, Modified Duration, Convexity 계산
- 금리 변화폭(bp)에 따른 가격 변화와 총 손익 시뮬레이션
- BeautifulSoup 기반 뉴스 제목 수집과 정책금리 압력 참고 지표 표시

## 파일 구조

- `app.py`: Streamlit 메인 앱
- `bond_math.py`: 채권가격, 듀레이션, 볼록성, forward rate 계산 함수
- `data_loader.py`: 한국은행/FRED API 또는 sample 데이터 로딩 함수
- `news_crawler.py`: 네이버 뉴스 검색 결과 크롤링 및 sample 뉴스
- `rate_signal.py`: 뉴스 제목 기반 정책금리 압력 참고 지표
- `visual_assets.py`: 국기, 차트, 뉴스 등 화면용 이미지 자산 registry
- `requirements.txt`: 실행에 필요한 패키지 목록
- `README.md`: 프로젝트 설명

## 사용한 계산식 요약

채권가격은 각 현금흐름의 현재가치 합으로 계산합니다.

```text
쿠폰 지급액 = 액면가 * 표면금리 / 연 지급횟수
가격 = Σ [CF_t / (1 + 시장수익률 / 연 지급횟수)^t]
Macaulay Duration = Σ [(t / 연 지급횟수) * PV(CF_t)] / 가격
Modified Duration = Macaulay Duration / (1 + 시장수익률 / 연 지급횟수)
```

Forward rate는 다음 식을 사용합니다.

```text
forward_rate = ((1 + spot_rate_nt) ** (n + t) / (1 + spot_rate_n) ** n) ** (1 / t) - 1
```

수익률곡선 기반 가격 계산은 각 현금흐름 시점의 spot rate를 선형보간하여 할인합니다.

## 데이터 수집 방식

한국 금리 데이터는 한국은행 ECOS API를 우선 사용합니다. `BOK_API_KEY`가 설정되어 있으면 한국은행 기준금리, 주요 시장금리, 국고채 수익률곡선, 국고채 3년/10년 히스토리를 ECOS에서 조회합니다. API 연결에 실패하거나 키가 없으면 sample 데이터가 반환됩니다.

미국채 금리와 미국 정책금리는 FRED API를 `requests`로 직접 호출합니다. `fredapi` 라이브러리는 사용하지 않습니다. FRED 호출 실패 또는 API 키 부재 시 sample 미국채 데이터를 사용합니다.

## API 키 설정

FRED API 키는 다음 우선순위로 읽습니다.

1. `st.secrets["FRED_API_KEY"]`
2. 환경변수 `FRED_API_KEY`
3. 없으면 sample 데이터

Streamlit secrets 예시:

```toml
FRED_API_KEY = "your_fred_api_key"
```

한국은행 API 키는 `BOK_API_KEY`로 설정합니다.

```toml
BOK_API_KEY = "your_bok_api_key"
```

## 한/미 정책금리 표시 방식

한국 정책금리는 ECOS `722Y001` 통계표의 한국은행 기준금리를 우선 표시합니다. 미국 정책금리는 가능한 경우 FRED의 `DFEDTARL`, `DFEDTARU`를 사용해 target range로 표시하고, 불가능하면 `FEDFUNDS` 또는 sample 데이터를 사용합니다.

## 다음 금통위/FOMC 일정 처리 방식

`load_policy_calendar()` 내부의 sample 일정 리스트에서 현재 날짜 이후 가장 가까운 날짜를 선택하고 D-day를 계산합니다. 사용자가 나중에 날짜 리스트를 직접 수정할 수 있습니다.

## 한국채/미국채 선택 기능

한국채는 ECOS의 일별 시장금리 통계표 `817Y002`에서 가져온 한국 국고채 수익률곡선 또는 입력한 시장수익률로 계산합니다. 미국채는 FRED 미국채 수익률곡선 또는 입력한 시장수익률로 계산합니다.

## 통화 단위 표시 방식

한국채 금액은 원, 미국채 금액은 달러로 표시합니다. 별도 환율 변환은 하지 않습니다.

## BeautifulSoup 크롤링의 역할

뉴스 크롤링은 홈 화면의 보조 참고 자료입니다. 채권가격, 듀레이션, 볼록성, 손익 계산에는 직접 사용하지 않습니다. 네이버 뉴스 검색 결과를 대상으로 하며, 브라우저 형태의 요청 헤더와 여러 HTML 구조 파싱을 사용합니다. 실제 뉴스가 하나라도 수집되면 실제 뉴스만 표시하고 sample 뉴스와 섞지 않습니다. 요청 실패, 차단, HTML 구조 변경으로 실제 뉴스가 전혀 없을 때만 sample 뉴스가 표시됩니다.

## Forward Rate 계산 방식

홈 화면의 Forward Rate 영역에서는 한국 국고채 또는 미국채를 선택하고, 시작 시점과 forward 기간을 1년 단위로 조정해 직접 계산합니다. 선택한 구간의 spot rate는 수익률곡선에서 선형보간하고, 계산식은 `bond_math.py`의 `calculate_forward_rate()`를 사용합니다.

## Raw Data 확인

사이드바 하단의 `디버깅용 raw data` 페이지에서 홈 화면과 계산에 사용되는 주요 원자료를 직접 확인할 수 있습니다. 홈 화면에서는 raw table을 숨겨 결과 중심으로 표시합니다.

## 화면 이미지 자산 처리

국가 표시는 이모지 대신 `visual_assets.py`의 이미지 registry를 통해 표시합니다. 현재 국기는 FlagCDN SVG, 보조 아이콘은 Iconify SVG URL을 사용하며, 나중에 로컬 이미지 파일이나 다른 아이콘 CDN으로 쉽게 교체할 수 있습니다.

## 정책금리 압력 참고 지표의 한계

정책금리 압력 참고 지표는 뉴스 제목에 포함된 단순 키워드만 세는 규칙 기반 지표입니다. 실제 기준금리 결정을 예측하는 모델이 아니며, 금융 의사결정에 단독으로 사용하면 안 됩니다.

## 한계점 및 개선 가능성

- 한국은행 ECOS API 실시간 연동 보완
- 실제 채권 종목별 현금흐름과 발행조건 반영
- 세금, 수수료, 재투자수익률, 신용위험 반영
- 수익률곡선 보간 방식 고도화
- 뉴스 수집처 다변화와 수동 새로고침 기능 추가
