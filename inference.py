import asyncio
import os
from openai import OpenAI
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS = 10

def log_start():
    print(f"[START] task=network env=network_env model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def agent_policy(state):
    if state["queue"] > 40:
        return "increase_bandwidth"
    elif state["utilization"] > 85:
        return "reroute"
    return "reroute"

async def main():
    log_start()

    rewards = []
    steps = 0

    res = requests.post(f"{ENV_URL}/reset", json={"task": "hard"}).json()
    state = res["observation"]

    for step in range(1, MAX_STEPS + 1):
        action = agent_policy(state)

        res = requests.post(f"{ENV_URL}/step", json={"action": action}).json()

        reward = res["reward"]
        done = res["done"]
        state = res["observation"]

        rewards.append(reward)
        steps = step

        log_step(step, action, reward, done)

        if done:
            break

    final_score = res.get("score", 0)
    score = max(0, min(1, final_score))
    success = score > 0.5
    log_end(success, steps, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())