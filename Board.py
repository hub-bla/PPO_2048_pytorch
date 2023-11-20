import random
import numpy as np
import msvcrt
class Board():
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size)).astype(np.float32)
        self.free_positions:list[tuple] = self._check_free_positions()
        self.last_move:int = None
        self.is_game_over = False
        self.reached_2048 = False
        self.overall_points = 0
        self.last_received_points = 0
        self.episode_length = 0
        self.episodes_with_points = 0
    def start(self):
        self._generate_new_block()

    def handle_move(self, move):
        self.last_received_points = 0.0001
        def _handle_horizontal_move(row, p1, p2, add_or_sub, direction):
            point1_val =0
            something_moved =False
            was_merged=False
            while True:
                if direction=="LEFT" or "UP":
                    if p2 >= len(row):
                        break
                if direction =="RIGHT" or "DOWN":
                        if 0>p2:
                            break
                if (was_merged or point1_val != row[p2]) and row[p2] != 0:
                      p1 +=add_or_sub
                      if p1!= p2:
                          self.last_received_points = 0
                          something_moved = True
                      temp = row[p2]
                      row[p2] = row[p1]
                      row[p1] = temp
                      point1_val = row[p1]
                      was_merged = False
                      
                elif point1_val == row[p2] and row[p2] != 0:
                    row[p1] *=2
                    point1_val = row[p1]
                    if self.last_received_points <0:
                        self.last_received_points =0
                    self.last_received_points+=point1_val
                    self.episodes_with_points+=1
                    self.overall_points += point1_val
                    if point1_val == 1:
                        self.reached_2048 =True
                    row[p2] =0
                    something_moved = True
                    #if something merged then we won't be adding here more
                    was_merged = True
                p2 +=add_or_sub  
            
            return something_moved
        


        should_generate = False
        if move == "RIGHT":
            for row in self.board:
              p1 = len(row)
              p2 =len(row)-1
              something_moved = _handle_horizontal_move(row, p1, p2, (-1), move)
              if should_generate is False and something_moved is True:
                  should_generate = True
                
        elif move =="LEFT":
            for row in self.board:
                p1 = -1
                p2 = 0
                something_moved = _handle_horizontal_move(row, p1, p2, 1, move)
                if should_generate is False and something_moved is True:
                  should_generate = True

        elif move == "UP":
            # transpose matrix so we can use already written function for horizontal moves
            temp_board = np.transpose(self.board)
            for row in temp_board:
                p1 = -1
                p2 = 0
                something_moved = _handle_horizontal_move(row, p1, p2, 1, move)
                if should_generate is False and something_moved is True:
                  should_generate = True
            #tanspose it back
            self.board = np.transpose(temp_board)

        elif move == "DOWN":
            # transpose matrix so we can use already written function for horizontal moves
            temp_board = np.transpose(self.board)
            for row in temp_board:
                p1 = len(row)
                p2 =len(row)-1
                something_moved = _handle_horizontal_move(row, p1, p2, (-1), move)
                if should_generate is False and something_moved is True:
                  should_generate = True
            #tanspose it back
            self.board = np.transpose(temp_board)

       

        self.free_positions = self._check_free_positions()
        self.episode_length +=1
        if should_generate and self.free_positions is not None:
                self._generate_new_block()
        else:
             self.last_received_points =0
            

    def _check_free_positions(self):

        def check_if_has_possible_moves_():
            for row_idx in range(self.board_size):
                for col_idx in range(self.board_size-1):
                    if self.board[row_idx][col_idx] == self.board[row_idx][col_idx+1]:
                        return True
                    elif row_idx != 0 and self.board[row_idx-1][col_idx] == self.board[row_idx][col_idx]:
                        return True
            return False

        free_pos =  [(i,j)  for j in range(self.board_size) for i in range(self.board_size) if self.board[i][j]==0]


        if len(free_pos) ==0:
            self.reward = -0.2
            if check_if_has_possible_moves_() is False:
                self.is_game_over = True
                return None
    
        return free_pos

    def _generate_new_block(self):
        row, column = random.choice(self.free_positions)
        prob = random.random()
        picked_number = 2/2048 if prob < 0.9 else 4/2048
        self.board[row][column] = picked_number
        self.free_positions.remove((row,column))







if __name__ == '__main__':
    board = Board(4)
    board.start()
  
    while board.is_game_over is False and board.reached_2048 is False:
        for row in board.board:
            print(row)
        print()
        key = msvcrt.getch()
        if key == b'w':
            board.handle_move("UP")
        elif key == b's':
            board.handle_move("DOWN")
        elif key == b'a':
            board.handle_move("LEFT")
        elif key == b'd':
            board.handle_move("RIGHT")
        elif key == b'E':
            break

    print("Reached points", board.overall_points)