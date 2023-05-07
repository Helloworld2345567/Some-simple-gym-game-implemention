import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

#environment names
RAM_ENV_NAME = 'LunarLander-v2'
VISUAL_ENV_NAME = 'Pong-v4'
ENV_NAME1='BipedalWalker-v3'
ENV_NAME2='ALE/AirRaid-v5'
render_mode="rgb_array"
CONSTANT = 68

#Agent parameters
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
TAU = 0.005
GAMMA = 0.99

#Training parameters
RAM_NUM_EPISODE = 1000
VISUAL_NUM_EPISODE = 5000
EPS_INIT = 1
EPS_DECAY = 0.995
EPS_MIN = 0.05
MAX_T = 5000
NUM_FRAME = 3