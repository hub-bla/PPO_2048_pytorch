import pygame
from board import Board

class Game():
    def __init__(self, board_size:int):
        self.board_size = board_size
        self.board = Board(self.board_size)
        self.action_space = ["RIGHT","LEFT", "UP", "DOWN"]
    def reset(self):
        self.board = Board(self.board_size)
        self.board.start()

        return self.get_board()

    def get_board(self):
        return self.board.board    

    def step(self, action:int):
        if self.board.is_game_over or self.board.reached_2048:
            self.reset()

        self.board.handle_move(self.action_space[action])
        reward = self.board.last_received_points

        # if (self.board.episode_length<1 and self.board.episodes_with_points< 1):
        #     reward = reward* (self.episodes_with_points/self.board.episode_length)
        if (self.board.reached_2048):
            reward=100
        if(self.board.is_game_over):
            reward =-1
        # elif(self.board.episode_length)< 200:
        #     reward = self.board.last_received_points/2

        
        return self.get_board(), reward, (self.board.is_game_over or self.board.reached_2048)


    def play_with_pygame(self, window_size):
        GAME_BACKGROUND_COLOR = (250,248,239)
        BOARD_BACKGROUND_COLOR = (187,173,160)
        EMPTY_POSITION_COLOR = (205,193,180)
        NUM_COLORS = {
            "2": (238, 228, 218),
            "4": (237, 224, 200),
            "8": (242, 177, 121),
            "16": (245, 149, 99),
            "32": (246, 124, 96),
            "64": (246, 94, 59),
            "128": (237, 207, 115),
            "256": (237, 204, 98),
            "512": (237, 200, 80),
            "1024": (237, 197, 63),
            "2048": (237, 194, 45)
        }
        DARK_TEXT_COLOR = (119, 110, 101)
        LIGHT_TEXT_COLOR = (249,246,242)
        pygame.init()
        def draw_text(text, pos, color):
            if text == "0":
                return
            font = pygame.font.Font(None, 40)
            txt = font.render(text, True, color)
            self.screen.blit(txt, pos)
        
        def drawGrid():
            blockSize = 100 
            start_of_grid = (((window_size[0]//2)-window_size[0]//4), ((window_size[1]//2)-window_size[1]//4))
            end_of_grid = ((start_of_grid[0]+(window_size[0]//2)), (start_of_grid[1]+(window_size[1]//2)))
            i = 0
            for y in range(start_of_grid[0], end_of_grid[0], blockSize):
                j = 0

                for x in range(start_of_grid[1], end_of_grid[1], blockSize):
                    rect = pygame.Rect(x, y, blockSize, blockSize)
                    str_num = str(int(self.board.board[i][j]*2048))
                    len_num = len(str_num)
                    
                    

                    if str_num != "0":
                        pygame.draw.rect(self.screen, NUM_COLORS[str_num], rect)
                    else:
                        pygame.draw.rect(self.screen, EMPTY_POSITION_COLOR, rect)

                    pygame.draw.rect(self.screen, BOARD_BACKGROUND_COLOR, rect, 10)
                   
                    text_color = LIGHT_TEXT_COLOR
                    if str_num =="2" or str_num =="4":
                        text_color=DARK_TEXT_COLOR
                    draw_text(str_num, ((x+blockSize//2)-8*len_num, (y+blockSize//2)-12), text_color)
                    j+=1
                i+=1
                

        self.screen = pygame.display.set_mode(window_size)

        running = True
        self.reset()
        self.screen.fill(GAME_BACKGROUND_COLOR)
        drawGrid()
        pygame.display.flip()

        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    
                        if event.key == pygame.K_a:
                            self.board.handle_move("LEFT")
                            print("test")
                        elif event.key ==  pygame.K_d:
                            self.board.handle_move("RIGHT")
                            print("test")
                        elif event.key ==  pygame.K_w:
                            self.board.handle_move("UP")
                            print("test")

                        elif event.key ==  pygame.K_s:
                            self.board.handle_move("DOWN")
                            print("test")
                        for row in self.board.board:
                                print(row)
                        print(self.board.is_game_over)
                        self.screen.fill(GAME_BACKGROUND_COLOR)
                        if self.board.is_game_over:
                            draw_text("Game over", (window_size[0]//2, window_size[1]//2),DARK_TEXT_COLOR)
                        elif self.board.reached_2048:
                            draw_text("You won", (window_size[0]//2, window_size[1]//2),DARK_TEXT_COLOR)
                        else:
                            drawGrid()

                        pygame.display.flip()


if __name__ == '__main__':

    Game(4).play_with_pygame((800,800))
    

