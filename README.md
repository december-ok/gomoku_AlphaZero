# 렌주 오목 AlphaZero

자기대국 MCTS + ResNet으로 학습하는 15x15 렌주룰 오목 AI.

## 사용

```sh
.venv/bin/python gomoku.py test                      # 규칙 self-check
.venv/bin/python gomoku.py train --model renju.pt    # 자기대국 학습
.venv/bin/python gomoku.py ladder --model renju.pt   # 회차별 학습 + 직전 모델과 대국
.venv/bin/python gomoku.py duel --model a.pt --other b.pt
.venv/bin/python gomoku.py play --model renju_best.pt --first
```

## 웹

`web/gomoku.html`에 모델 가중치가 포함되어 있어 파일만 열면 동작합니다 (현재 `pure_r10.pt`).

```sh
.venv/bin/python export_web.py pure_r10.pt   # 가중치 교체
```
