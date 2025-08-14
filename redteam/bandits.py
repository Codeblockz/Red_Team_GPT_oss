import math
class UCB1:
    def __init__(self, n_arms: int):
        self.n = [0]*n_arms
        self.s = [0.0]*n_arms
        self.t = 0
    def select(self):
        self.t += 1
        for i,c in enumerate(self.n):
            if c==0: return i
        import math
        ucb = [self.s[i]/self.n[i] + math.sqrt(2*math.log(self.t)/self.n[i]) for i in range(len(self.n))]
        return max(range(len(self.n)), key=lambda i: ucb[i])
    def update(self, i, reward: float):
        self.n[i]+=1; self.s[i]+=reward
