import torch
from model import PpoAgent
from game import Game
import numpy as np
import time
from utils import one_hot_encode


def evaluate_agent(agent):
    moves = 0 
    env1 = Game(4)
    env1.reset()

    while (env1.board.is_game_over or env1.board.reached_2048) is False:

        state = one_hot_encode(env1.get_board(), env1.board_size)
        t_board = torch.zeros((1, state.shape[0], state.shape[1], state.shape[2]))
        t_board[0]=state
  
        action,_,_,_ = agent.get_action_and_value(t_board)
        move  = action.item()
        env1.step(move)
        moves+=1
        
    print("BOARD MAX", np.max(env1.get_board()))
    print("POINTS", env1.board.overall_points)
    print("MOVES", moves)
    return env1.board.overall_points, moves
