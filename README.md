# rl-painter

Reinforcement Learning to Paint

## Brief

Each observation contains two images, the "target" and the "canvas".
Given the observation, you should output a list of 8 scalars within the range of [0,1] as the action, to control where to paint the next stroke.

The reward will be positive if you decreased the difference between "target" and "canvas".
The higher the total reward, the better your agent have painted.

## Usage

`python env.py` to test the environment.

`ipython -i ddpg2.py` then `r(10000)` to test the env with a naive DDPG algorithm.
