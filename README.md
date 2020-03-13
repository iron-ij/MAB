# MAB

1) 베이지안 MAB

- 슬롯머신 여러대가 있을 때, 최대한의 상금을 얻으려면 이중에서 어떤 슬롯머신을 선택해야하는가?
    - 여러 캠페인(=슬롯머신) 중에서 가장 효과가 좋은 캠페인을 선택하려면?
    - 각 캠페인에는 Try / visit / Goal(conditional, 골 달성수) 가 있는데,
    어떤 방식으로 접근해야 최적화가 가능할까?
- 베이지안 방식 : 각 슬롯머신을 실험을 해본다(pull, draw) → 통계적 성공 확률이 가장 높은 슬롯을 선택
- MAB(Multi armed bandit) : 그 시점에 기록을 통해 승률이 높은 슬롯(캠페인)을 고르는 알고리즘

2) 방식

- MAB는 1. 슬롯 정의 2. 학습전략 구현으로 구성됨
- 슬롯 정의에서는 input : bandit_probs 각 캠페인별로 성공할 확률이 정의됨 → k번째로 선택된 슬롯(캠페인)의 확률로 성공여부 도출(= 과거결과 기반으로 도출) → regret : 최대확률 대비 선택된 확률 사이의 GAP (선택으로 인한 기회비용)
1. 슬롯 정의

    # 1. class for our row of bandits
    
    class MAB:
        # initialization
        def __init__(self, bandit_probs):
            # storing bandit probs
            self.bandit_probs = bandit_probs
            
        # function that helps us draw from the bandits
        def draw(self, k):
            # we return the reward and the regret of the action
            # 총 N개 중 n개의 무작위 난수 생성 self.bandit_probs[k] 확률에 대한 난수 산출
            return np.random.binomial(1, self.bandit_probs[k]), 
    								np.max(self.bandit_probs) - self.bandit_probs[k]

2. 학습전략 (캠페인 선택 전략)

(1) e-Greedy

    # 트레이드 오프가 발생하기 때문에 다양한 정책을 두고 실험을 해보자
    ## 1. E-Greedy (입실론-그리디) policy
    ### -> 가장 확률이 높은 정책을 선택하며, 입실론(e)만큼의 확률로 랜덤 샘플링 한다.
    ### -> 단, 이때 최대 확률인 값은 제외한다
    class Greedy:
        def __init__(self, epsilon):
            self.epsilon = epsilon
            
        def choose_bandit(self, k_array, reward_array, bandit_num):
            success_count = reward_array.sum(axis = 1)
            total_count = k_array.sum(axis = 1)
            success_ratio = success_count/total_count
    
            if np.random.random() < self.epsilon:
                return np.random.choice(np.delete(list(range(bandit_num)), 
    										np.argmax(success_ratio)))
            else:
                return np.argmax(success_ratio)

(2) UCB

    ## 2. UCB policy
    ### -> 확률 + UCB_term의 합이 가장 큰 값을 고른다
    ### 적당한 확률차이는 봐주고 그 중에서 선택한다
    class UCB:
        def __init__(self):
            pass
        
        def choose_bandit(self, k_array, reward_array, bandit_num):
            success_count = reward_array.sum(axis = 1)
            total_count = k_array.sum(axis = 1)
            success_ratio = success_count/total_count
    
            #UCB term
            #sqrt(2log(sum of total count)) / total count
            ucb_value = np.sqrt(2*np.log(np.sum(total_count))/total_count)
    
            return np.argmax(success_ratio + ucb_value)

(3) Thompson sampling

    ## 3. thompson sampling
    ### -> 각 밴딧의 결과로 Beta_dist-> 랜덤sampling -> prob 도출 -> 가장 큰 밴딧 고른다
    class Thompson:
        def __init__(self):
            pass
    
        def choose_bandit(self, k_array, reward_array, bandit_num):
            success_count = reward_array.sum(axis = 1) # 성공횟수
            total_count = k_array.sum(axis = 1) # 전체 시행횟수
    
            #prob term
            prob_list = [
                np.random.beta(i+1, k+1)
                for i, k
                in zip(success_count, total_count-success_count)
            ]
    
            return np.argmax(prob_list)

3. 시뮬레이션 

- 4개 밴딧에 대해 10,000번 시뮬레이션 (각 확률 0.35, 0.40, 0.30, 0.25)

![CH%20DS%20MAB%20Research/_2020-02-16__8.43.51.png](CH%20DS%20MAB%20Research/_2020-02-16__8.43.51.png)

- 10,000번 시뮬레이션 결과 : 수렴속도 TS > e-greedy(10%) > UCB
- Sum of regret (시뮬레이션 간 기회비용의 합)

    ![CH%20DS%20MAB%20Research/_2020-02-16__8.52.57.png](CH%20DS%20MAB%20Research/_2020-02-16__8.52.57.png)

- 수렴속도가 느릴수록 기회비용이 크다 (UCB > e-greedy(10%) > TS)

3) cMAB(Contextual MAB)

- state : MAB에 추가된 개념으로 캠페인을 진행할 당시에 상황을 고려해서 확률을 도출
- 최근, 여러 방법론들이 더해지며 적용되고 있으며 다양한 방법론이 나오고 있음
- 캠페인으로 예를 들면, 각 캠페인 별로 유저에 성향에 따라 다른 밴딧의 확률이 다르게 나오도록 구성
→ 일반 MAB에서 캠페인의 특성을 반영하기 위해 생성, 기존 기록에서 모델링을 통해 성능을 개선
→ 프로필(고객 유형), 접속시간, 페이지 등에 따라 확률 모형 구축 후 학습
→ 머신러닝 방법론 → 추천 모델링 → 강화학습으로 진화되고 있음
