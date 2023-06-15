# Some-simple-gym-game-implemention
**My Envirnment：**\
Python3.10\
torch 2.0\
numpy 1.24.0\
gym 0.26.2\
gym  atari and box2d dependency\
_Don't worry about envirnment problems.High compatibility it has._

**install gym:**
```
pip install gym
pip install gym[atari]
pip install gym[accept-rom-license]
apt-get install -y swig
pip install box2d-py
pip install gym[box2d] 
```

## Pong , Lunerland , AirRaid 
These game's action is discreate, use simple DQN algorithms to implement it.

## BipedalWalker 
This game's action is continuous, so you can use PPO or SAC to implement it.DQN is not good at it
![example gif](https://raw.githubusercontent.com/Helloworld2345567/Some-simple-gym-game-implemention/blob/master/BipedalWalkerHardcore-SAC/gif/result_hard6000.gif)
<img src="https://github.com/Helloworld2345567/Some-simple-gym-game-implemention/blob/master/BipedalWalkerHardcore-SAC/gif/result_hard6000.gif" alt="animation">

### SAC：
[**Site Reference**](https://github.com/CoderAT13/BipedalWalkerHardcore-SAC.git)
