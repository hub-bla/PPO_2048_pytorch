import torch
import numpy as np


def init_layer(layer, gain=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def one_hot_encode(board, board_size):
    
    encoded_state = torch.zeros((16, 4, 4))
    board = np.log2(board, out=np.zeros_like(board), where=(board!=0))

    for i in range(board_size):
        for j in range(board_size):

            number = int(board[i][j])
            if number != 0:
                encoded_state[number, i, j] = float(number)

    return encoded_state
   
