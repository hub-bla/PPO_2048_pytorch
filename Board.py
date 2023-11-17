import random
import numpy as np
import msvcrt
class Board():
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size)).astype(np.uint16)
        self.free_positions:list[tuple] = self.check_free_positions_()
        self.last_move:int = None
        self.generate_new_block_()

    def handle_move(self, move):
        def handle_horizontal_move_(row, p1, p2, add_or_sub, direction):
            point1_val =0
            something_moved =False
            while True:
                if direction=="LEFT" or "UP":
                    if p2 >= len(row):
                        break
                if direction =="RIGHT" or "DOWN":
                        if 0>p2:
                            break
                if point1_val != row[p2] and row[p2] != 0:
                      p1 +=add_or_sub
                      if p1!= p2:
                          something_moved = True
                      temp = row[p2]
                      row[p2] = row[p1]
                      row[p1] = temp
                      point1_val = row[p1]
                      
                elif point1_val == row[p2] and row[p2] != 0:
                    row[p1] *=2
                    point1_val = row[p1]
                    row[p2] =0
                    something_moved = True
                p2 +=add_or_sub  
            
            return something_moved
        


        should_generate = False
        if move == "RIGHT":
            for row in self.board:
              p1 = len(row)
              p2 =len(row)-1
              something_moved = handle_horizontal_move_(row, p1, p2, (-1), move)
              if should_generate is False and something_moved is True:
                  should_generate = True
                
        elif move =="LEFT":
            for row in self.board:
                p1 = -1
                p2 = 0
                something_moved = handle_horizontal_move_(row, p1, p2, 1, move)
                if should_generate is False and something_moved is True:
                  should_generate = True

        elif move == "UP":
            # transpose matrix so we can use already written function for horizontal moves
            temp_board = np.transpose(self.board)
            for row in temp_board:
                p1 = -1
                p2 = 0
                something_moved = handle_horizontal_move_(row, p1, p2, 1, move)
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
                something_moved = handle_horizontal_move_(row, p1, p2, (-1), move)
                if should_generate is False and something_moved is True:
                  should_generate = True
            #tanspose it back
            self.board = np.transpose(temp_board)

        if should_generate:
            self.free_positions = self.check_free_positions_()
            self.generate_new_block_()
            

    def check_free_positions_(self):
        return [(i,j)  for j in range(self.board_size) for i in range(self.board_size) if self.board[i][j]==0]

    def generate_new_block_(self):
        print(self.free_positions)
        row, column = random.choice(self.free_positions)
        prob = random.random()
        picked_number = 2 if prob < 0.9 else 4
        self.board[row][column] = picked_number
        self.free_positions.remove((row,column))








if __name__ == '__main__':
    board = Board(4)
    
    # for i, k in enumerate([0, 2, 2, 8]):
    #     board.board[0,i] = k
    # for row in board.board:
    #         print(row)
    # print()
    # board.handle_move("LEFT")
    # for row in board.board:
    #         print(row)
    while True:
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

