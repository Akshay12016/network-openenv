# Network Traffic Optimization OpenEnv

## Description
Simulates network congestion and allows an agent to optimize routing and bandwidth.

## Actions
- reroute
- increase_bandwidth
- do_nothing

## State
- latency
- packet_loss
- utilization
- queue

## Run
docker build -t network-env .
docker run -p 7860:7860 network-env