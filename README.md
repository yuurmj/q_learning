# Q-learning PyTorch Implementation

책 *파이썬과 케라스로 배우는 강화학습*의 GridWorld 예제를 참고하여  
Q-learning을 PyTorch tensor 기반으로 재구현한 프로젝트입니다.

## Overview
- GridWorld 환경에서 Q-learning 학습 수행
- Q-table을 PyTorch tensor로 관리
- 상태를 index 형태로 변환하여 Q-table에 접근
- episode별 score를 기록하고 learning curve 저장
- 학습된 Q-table을 파일로 저장

## Project Structure
q_learning/
├── environment.py
├── q_learning_agent.py
├── train.py
└── results/
    ├── learning_curve.png
    └── q_table.pth

## Requirements
- Python 3.x
- torch
- matplotlib
- numpy
- Pillow

필요한 패키지는 다음과 같이 설치할 수 있습니다.

pip install torch matplotlib numpy pillow

## How to Run
프로젝트 루트 디렉토리에서 아래 명령어를 실행합니다.

python3 q_learning/train.py

## Implementation Details
- 기존 예제 구조를 참고하되, agent 부분은 PyTorch tensor 기반으로 재구성
- torch.zeros(...)를 사용해 Q-table 초기화
- 상태 [x, y]를 하나의 index로 변환하여 tensor table에 접근
- epsilon-greedy 정책으로 행동 선택
- epsilon decay를 적용하여 탐험 비중이 점차 감소하도록 구현
- 학습 결과를 results/ 폴더에 저장하도록 구성

## Output
학습이 완료되면 아래 결과물이 생성됩니다.

- q_learning/results/learning_curve.png : episode별 score 그래프
- q_learning/results/q_table.pth : 학습된 Q-table

## Notes
- environment.py는 GridWorld 환경, 상태 전이, 보상 구조를 담당합니다.
- q_learning_agent.py는 행동 선택과 Q값 업데이트를 담당합니다.
- train.py는 학습 실행, score 기록, 결과 저장을 담당합니다.ㄴ