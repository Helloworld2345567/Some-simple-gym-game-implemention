# Some-simple-gym-game-implemention
**My Envirnment：**\
Python3.10\
torch 2.0\
numpy 1.24.0\
gym 0.26.2\
gym  atari and box2d dependency\
_Don't worry about envirnment problems.High compatibility it has._
非常感谢您提供的更新信息！下面是包含这些信息的表格整理：

| 微调类型           | Epoch | 步数  | Loss  | 时间      | GPU     |
|----------------|------ |------ |-------|----------|---------|
| 7B-Lora        | 128   | 12800 | 1.24  | 12.58小时 | 3张3090 |
| 3B-Lora        | 32    | 3200  | 1.42  | 5.43小时  | 2张3090 |
| 3B-Full-finetune | 32    | 16384 | 0.45  | 19.07小时 | 3张3090+CPU |

希望这个更新的表格能帮到您！如果还有其他问题，请随时提问。
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
<img src="https://github.com/Helloworld2345567/Some-simple-gym-game-implemention/blob/master/BipedalWalkerHardcore-SAC/gif/result_hard6000.gif" alt="animation">

## SAC：
[**Site Reference**](https://github.com/CoderAT13/BipedalWalkerHardcore-SAC.git)
