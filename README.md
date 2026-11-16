# Acoustic Event Forensics: Multi-Task AST

**2025 INHA SW FESTIVAL Kaggle 부문 우수상** (SW중심대학사업단장상)

충격음 하나로 무엇이 부딪혔는지(분류)와 소리가 얼마나 컸는지(dB 회귀)를 동시에 맞히는 Audio Spectrogram Transformer 멀티태스크 모델입니다.

| 항목 | 내용 |
|---|---|
| 기간 | 2025.09 |
| 팀 | DeCO · 정영운 · 정준우 · 명근호 |
| 사용 기술 | Python, PyTorch, AST(Audio Spectrogram Transformer) |

## 문제

책상, 의자, 강연대, 책, 망치 5가지 충격원의 WAV 녹음을 받아 충격원을 분류하고 방사소음 크기(dB)를 예측합니다. 분류는 F1-score, 크기 예측은 MAE로 평가합니다.

## 접근

- WAV를 Mel-filterbank 스펙트로그램으로 바꾸고 길이를 맞춘 뒤 정규화했습니다.
- 사전학습된 AST를 공유 백본으로 두고, 분류 헤드와 회귀 헤드를 붙여 두 손실의 합으로 함께 학습했습니다.
- Mixed Precision과 Gradient Accumulation으로 Colab GPU에서 학습했고, 검증 성능 기준으로 최적 모델을 저장했습니다.

```
WAV → Mel 스펙트로그램 → AST 인코더 ─┬→ 분류 헤드 → 충격원
                                    └→ 회귀 헤드 → dB
```

## 결과

테스트 327개 녹음 기준입니다.

| 지표 | 값 |
|---|---|
| 충격원 분류 정확도 | 97.6% |
| 소음 크기 MAE | 1.41 dB |

파일별 예측은 [test_results_all_in_one.csv](test_results_all_in_one.csv), 에폭별 학습 기록은 [training_history.csv](training_history.csv)에 있습니다.

## 구성

```
Multi_Task_AST.ipynb   전처리, 학습, 평가 전체 과정
predict.py             저장된 모델로 추론
best_model.pth         학습된 가중치 (Git LFS)
[오픈소스 SW 페스티벌] … .pdf   제출 보고서
```

AST 구현은 [YuanGongND/ast](https://github.com/YuanGongND/ast)를 사용했습니다.
