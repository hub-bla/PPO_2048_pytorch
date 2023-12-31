import random
import numpy as np


class Board:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size)).astype(np.float32)
        self.free_positions = self._check_free_positions()
        self.last_move = None
        self.is_game_over = False
        self.reached_2048 = False
        self.overall_points = 0
        self.last_received_points = 0

    def start(self):
        self._generate_new_block()

    def handle_move(self, move):
        self.last_received_points = 0
        def _handle_horizontal_move(row, p1, p2, add_or_sub, direction):
            point1_val = 0
            something_moved = False
            was_merged = False
            while True:
                if direction == "LEFT" or "UP":
                    if p2 >= len(row):
                        break
                if direction == "RIGHT" or "DOWN":
                    if 0 > p2:
                        break
                if (was_merged or point1_val != row[p2]) and row[p2] != 0:
                    p1 += add_or_sub
                    if p1 != p2:
                        something_moved = True
                    
                    temp = row[p2]
                    row[p2] = row[p1]
                    row[p1] = temp
                    point1_val = row[p1]
                    was_merged = False
                      
                elif point1_val == row[p2] and row[p2] != 0:
                    row[p1] *= 2
                    point1_val = row[p1]
                    #normalize values
                    self.last_received_points += point1_val/2048
                    something_moved = True
                    was_merged = True
                    if point1_val == 2048:
                        self.reached_2048 =True
                        self.last_received_points = 1
                    row[p2] =0
                   
                p2 += add_or_sub
            
            return something_moved
        
        should_generate = False
        if move == "RIGHT":
            for row in self.board:
              p1 = len(row)
              p2 = len(row)-1
              something_moved = _handle_horizontal_move(row, p1, p2, (-1), move)
              if should_generate is False and something_moved is True:
                  should_generate = True
                
        elif move == "LEFT":
            for row in self.board:
                p1 = -1
                p2 = 0
                something_moved = _handle_horizontal_move(row, p1, p2, 1, move)
                if should_generate is False and something_moved is True:
                    should_generate = True

        elif move == "UP":
            # transpose matrix, so we can use already written function for horizontal moves
            temp_board = np.transpose(self.board)
            for row in temp_board:
                p1 = -1
                p2 = 0
                something_moved = _handle_horizontal_move(row, p1, p2, 1, move)
                if should_generate is False and something_moved is True:
                  should_generate = True
            #transpose it back
            self.board = np.transpose(temp_board)

        elif move == "DOWN":
            # transpose matrix, so we can use already written function for horizontal moves
            temp_board = np.transpose(self.board)
            for row in temp_board:
                p1 = len(row)
                p2 = len(row)-1
                something_moved = _handle_horizontal_move(row, p1, p2, (-1), move)
                if should_generate is False and something_moved is True:
                    should_generate = True
            #transpose it back
            self.board = np.transpose(temp_board)

        self.free_positions = self._check_free_positions()

        if not something_moved:
            self.last_received_points -=0.002
        if should_generate and self.free_positions is not None:
            self._generate_new_block()

    def _check_free_positions(self):

        def check_if_has_possible_moves_():
            for row_idx in range(self.board_size):
                for col_idx in range(self.board_size-1):
                    if self.board[row_idx][col_idx] == self.board[row_idx][col_idx+1]:
                        return True
                    elif row_idx != 0 and self.board[row_idx-1][col_idx] == self.board[row_idx][col_idx]:
                        return True
            return False

        free_pos = [(i, j) for j in range(self.board_size) for i in range(self.board_size) if self.board[i][j] == 0]

        if len(free_pos) == 0:
            if check_if_has_possible_moves_() is False:
                self.is_game_over = True
                return None
    
        return free_pos

    def _generate_new_block(self):
        row, column = self.free_positions[0]
        self.board[row][column] = 2

