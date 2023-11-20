import torch
from model import PpoAgent
from game import Game
import numpy as np
import time
env = Game(4)

agent = PpoAgent(env.board_size, len(env.action_space))

torch.load
checkpoint = torch.load("./model.pt")
agent.load_state_dict(checkpoint)
env.reset()
moves= 0 
agent.eval()
while (env.board.is_game_over or env.board.reached_2048) is False:
    for row in env.get_board():
        print(row*2048)
    print()
    # time.sleep(1)

    t_board = torch.tensor(env.get_board().astype(np.float32)).flatten()
    obs = torch.zeros((1, env.board_size*env.board_size))
    obs[0] = t_board
    print("T_BOARD",t_board)
    action,_,_,_ = agent.get_action_and_value(obs)
    print("ACTION",action)
    move  = action.item()
    env.step(move)
    moves+=1
print("moves", moves)