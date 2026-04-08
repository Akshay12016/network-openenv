from fastapi import FastAPI
from env.network_env import NetworkEnv

app = FastAPI()
env = NetworkEnv()

@app.post("/reset")
async def reset(data: dict = {}):
    task = data.get("task", "easy")
    return await env.reset(task)

@app.post("/step")
async def step(action: dict):
    return await env.step(action)

@app.get("/state")
async def state():
    return env.state()