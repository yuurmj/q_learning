import os
import matplotlib.pyplot as plt
from environment import Env
from q_learning_agent import QLearningAgent

if __name__ == "__main__":
    # 현재 파일의 디렉토리를 기준으로 결과 저장용 폴더 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    episodes = 200 # 에피소드 수 증가
    scores, episodes_list = [], []

    for episode in range(episodes):
        state = env.reset()
        score = 0

        while True:
            env.render()

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            
            # 행동을 취한 후 다음 상태, 보상, 종료 여부 획득
            next_state, reward, done = env.step(action)

            # 큐함수 업데이트
            agent.learn(state, action, reward, next_state)

            state = next_state
            score += reward

            # 화면에 큐함수 값 표시
            env.print_value_all(agent.q_table)

            if done:
                scores.append(score)
                episodes_list.append(episode)
                print(f"episode: {episode}, score: {score}, epsilon: {agent.epsilon:.2f}")
                
                # 에피소드 종료 후 입실론 감쇠
                agent.decay_epsilon()
                break
    
    # 학습 결과 저장 (results/ 폴더 안)
    agent.save_model(os.path.join(results_dir, "q_table.pth"))

    # 학습 그래프 출력 및 파일 저장
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_list, scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Q-Learning Training Performance')
    plt.grid(True)
    
    # 그래프를 파일로 저장
    graph_path = os.path.join(results_dir, "learning_curve.png")
    plt.savefig(graph_path)
    print(f"Graph saved to {graph_path}")
    
    plt.show()
