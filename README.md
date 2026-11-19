# Energy Forecasting & Analysis Portfolio

김민규 · 중앙대학교 경제학과 석사과정

전력·에너지 수요를 시계열로 다룹니다. 이 저장소에는 지금까지 한 분석 가운데 공개 자료로 다시 돌릴 수 있는 것들을 정리해 두었습니다.

| 프로젝트 | 내용 | 방법 |
|---|---|---|
| [renewable-drought](renewable-drought) | 한국 태양광·풍력의 상호보완성, 재생 가뭄, 12차 전기본 2040년 설비 시나리오 | 시간별 패널, MBT/CBT 가뭄 지표, HAC 회귀, 시나리오 시뮬레이션 |
| [holiday-effective-days](holiday-effective-days) | 평일 공휴일이 상권 유형별 수요에 미치는 영향을 유효일수로 측정 (서울 행정동) | 패널 고정효과, 이원 군집 SE, 델타법 |
| [coal-closure-population](coal-closure-population) | 석탄 폐광 이후 탄광 도시의 인구 변화 | 사건별 묶음 국소투영, 예측오차 SE |
| [korea-wind-data](korea-wind-data) | 풍력·기상 공개데이터 수집과 정리 파이프라인 | KPX, EPSIS, ASOS, ERA5, 기상청 부이 |

## 환경

```bash
pip install -r requirements.txt
```

Python 3.11 이상. 그림의 한글 글꼴은 Windows 맑은 고딕, macOS Apple SD Gothic Neo, Linux Noto Sans CJK 순으로 찾고, `KFONT` 환경변수로 직접 지정할 수도 있습니다.

## 데이터

원자료와 중간 산출물은 올리지 않았습니다. 대부분 공공데이터포털, KOSIS, 서울 열린데이터광장, 기상자료개방포털 같은 공개 자료이고 프로젝트마다 README에 출처와 입력 형식을 적어 두었습니다. 수업에서 받은 가공 자료로 분석한 부분은 해당 README에 따로 표시했습니다.
