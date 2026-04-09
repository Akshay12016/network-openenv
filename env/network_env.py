import random

class NetworkEnv:
    
    def __init__(self):
        self.max_steps = 10
        self.task = "easy"  # default

    async def reset(self,task='easy'):
        self.step_count = 0
        self.task = task

        self.capacity = {
            ("R1", "R2"): 100,
            ("R2", "R3"): 80,
            ("R1", "R3"): 60
        }

        self.traffic = {("R1", "R3"): 300}

        self.state_data = self._compute_state()
        print("DEBUG TRAFFIC:", self.traffic)
        return {
            "observation": self.state_data,
            "reward": 0,
            "done": False
        }

    async def step(self, action):
        self.step_count += 1

        action_type = action.get("action")

        if action_type == "increase_bandwidth":
            for k in self.capacity:
                self.capacity[k] += 10

        elif action_type == "reroute":
            if self.traffic[("R1", "R3")] >= 10:
                self.traffic[("R1", "R3")] -= 10
                self.traffic[("R1", "R2")] = self.traffic.get(("R1", "R2"), 0) + 10

        self.state_data = self._compute_state()
        reward = self._compute_reward()

        done = self.step_count >= self.max_steps

        return {
            "observation": self.state_data,
            "reward": reward,
            "done": done,
            "score": self._compute_score()
        }

    def state(self):
        return self.state_data

    def _compute_state(self):

        total_load = sum(self.traffic.values())
        total_capacity = sum(self.capacity.values())

        utilization = (total_load / total_capacity) * 100

        queue = max(0, total_load - total_capacity)

        latency = 50 + utilization * 0.8 + queue * 0.5
        latency += random.uniform(-2, 2)
        packet_loss = max(0, queue * 0.2)

        return {
            "latency": round(latency, 2),
            "packet_loss": round(packet_loss, 2),
            "utilization": round(utilization, 2),
            "queue": round(queue, 2)
        }

    def _compute_reward(self):
        latency = self.state_data["latency"]
        packet_loss = self.state_data["packet_loss"]
        utilization = self.state_data["utilization"]

        reward = (
            200 - latency
            - packet_loss * 5
            - abs(utilization - 70) * 2
        )
        return round(reward, 2)
    
    def _grade_easy(self):
        latency = self.state_data["latency"]
        score = max(0, 1 - latency / 300)
        return round(min(score, 1), 3)

    def _grade_medium(self):
        latency = self.state_data["latency"]
        packet_loss = self.state_data["packet_loss"]
        score = max(0, 1 - (latency + packet_loss * 10) / 400)
        return round(min(score, 1), 3)

    def _grade_hard(self):
        latency = self.state_data["latency"]
        packet_loss = self.state_data["packet_loss"]
        utilization = self.state_data["utilization"]
    
        score = max(0, 1 - (latency + packet_loss * 5 + utilization) / 500)
        return round(min(score, 1), 3)

    def _compute_score(self):
        if self.task == "easy":
            return self._grade_easy()
    
        elif self.task == "medium":
            return self._grade_medium()
    
        elif self.task == "hard":
            return self._grade_hard()
    
        return 0.0
