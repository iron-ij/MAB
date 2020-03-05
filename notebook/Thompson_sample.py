import numpy as np

class AdjThompson:
    """
    thompson sampling
    각 밴딧의 결과로 Beta_dist-> 랜덤sampling -> prob 도출 -> 가장 큰 밴딧 고른다
    
    epsilon : 매 시도당 epsilon의 확률부여 -> 지정값보다 작은 경우 랜덤선택(최대확률 캠페인 제외)
    sent : 발송수(sent)가 저장된 list 형태 (ex. 캠페인 4개 : [100, 200, 300, 400], 캠페인(슬롯)에 따라 element가 늘어남)
    view : view가 저장된 list 형태 (ex. 캠페인 4개 : [50, 70, 50, 40], 캠페인(슬롯)에 따라 element가 늘어남)
    goal(*) : 조건부이며, goal 달성 횟수가 저장된 list 형태 (ex. 캠페인 4개 : [25, 30, 20, 10], 캠페인(슬롯)에 따라 element가 늘어남)

    """
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_bandit(self, sent, view, goal=False):
        total_count = sent # 전체 시행횟수

        # goal이 있으면서, 모든 캠페인이 최소 1개 이상의 골을 달성하는 경우에 베타함수를 (goal/total)로 계산
        # goal이 없거나 1개미만의 골을 달성한 캠페인이 존재하면, 각 캠페인에 대해 view/total을 타겟으로 베타함수 계산
        if np.all((goal) & (np.min(goal) > 0)):
            success_count = goal # goal 성공횟수
        else:
            success_count = view # view 성공횟수

        # beta prob term
        # 각 캠페인에 대해 beta dist 확률 계산
        prob_list = [
            np.random.beta(i+1, k+1) # 기본가정에 의해 각 요소의 최소값은 1
            for i, k
            in zip(success_count, total_count-success_count) # 성공횟수, 실패횟수
        ]
        print("beta prob list : ", prob_list)


        if np.random.random() < self.epsilon:
            print("under epsilon -> random select campaign")
            # array에서 가장 확률이 높은 값 삭제 -> 랜덤으로 index(캠페인) 값 리턴
            # argmax는 주어진 array에서 가장 값이 큰 value의 인덱스를 리턴
            return np.random.choice(np.delete(list(range(len(prob_list)))), np.argmax(prob_list))
        else:
            # 입실론 이상이면 확률을 최대로 하는 값의 index(캠페인 순서) 리턴
            return np.argmax(prob_list)

if __name__ == "__main__":
    ATS = AdjThompson(0.2)
    sent = np.array([100, 120, 140, 160]) # sent가 0인경우 uniform dist
    view = np.array([5, 6, 6, 4])
    goal = np.array([3, 3, 3, 2])

    print("goal exists!")
    ATS.choose_bandit(sent, view, goal)

    print("no goal")
    ATS.choose_bandit(sent, view)
