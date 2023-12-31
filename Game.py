import pygame
import torch
from board import Board
from utils import one_hot_encode
from model import PpoAgent



class Game:
    def __init__(self, board_size:int):
        self.board_size = board_size
        self.board = Board(self.board_size)
        self.action_space = ["RIGHT", "LEFT", "UP", "DOWN"]
        self.action_space_n = [0, 1, 2, 3]
        self.screen = None

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

        if self.board.reached_2048:
            reward=1.0
        if self.board.is_game_over:
            reward =-1.0
        
        return self.get_board(), reward, (self.board.is_game_over or self.board.reached_2048)

    def play_with_pygame(self, window_size, played_by_agent=True):
        GAME_BACKGROUND_COLOR = (250, 248, 239)
        BOARD_BACKGROUND_COLOR = (187, 173, 160)
        EMPTY_POSITION_COLOR = (205, 193, 180)
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
            "2048": (237, 194, 45),
            "4096": (237, 194, 45),
            "8192": (237, 194, 45),
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
            block_size = 100
            start_of_grid = (((window_size[0]//2)-window_size[0]//4), ((window_size[1]//2)-window_size[1]//4))
            end_of_grid = ((start_of_grid[0]+(window_size[0]//2)), (start_of_grid[1]+(window_size[1]//2)))
            i = 0
            for y in range(start_of_grid[0], end_of_grid[0], block_size):

                j = 0
                for x in range(start_of_grid[1], end_of_grid[1], block_size):
                    rect = pygame.Rect(x, y, block_size, block_size)
                    str_num = str(int(self.board.board[i][j]))
                    len_num = len(str_num)

                    if str_num != "0":
                        pygame.draw.rect(self.screen, NUM_COLORS[str_num], rect)
                    else:
                        pygame.draw.rect(self.screen, EMPTY_POSITION_COLOR, rect)

                    pygame.draw.rect(self.screen, BOARD_BACKGROUND_COLOR, rect, 10)
                   
                    text_color = LIGHT_TEXT_COLOR
                    if str_num == "2" or str_num == "4":
                        text_color = DARK_TEXT_COLOR

                    draw_text(str_num, ((x+block_size//2)-8*len_num, (y+block_size//2)-12), text_color)
                    j += 1

                i += 1

        self.screen = pygame.display.set_mode(window_size)
        running = True
        self.reset()
        self.screen.fill(GAME_BACKGROUND_COLOR)
        drawGrid()
        pygame.display.flip()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load("./model.pt", map_location=device)
        agent = PpoAgent().to(device)
        agent.load_state_dict(checkpoint)
        agent.eval()
        while running:
            if not played_by_agent:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    
                        if event.type == pygame.KEYDOWN:
                            
                                if event.key == pygame.K_a:
                                    self.board.handle_move("LEFT")
                                elif event.key == pygame.K_d:
                                    self.board.handle_move("RIGHT")
                                elif event.key == pygame.K_w:
                                    self.board.handle_move("UP")
                                elif event.key == pygame.K_s:
                                    self.board.handle_move("DOWN")
            else:
                state = one_hot_encode(self.get_board(), self.board_size)
                t_board = torch.zeros((1, state.shape[0], state.shape[1], state.shape[2]))
                t_board[0] = state
        
                action, _, _, _ = agent.get_action_and_value(t_board)
                move = action.item()
                self.step(move)
                # sleep(.05)
                # print("move", move)

            self.screen.fill(GAME_BACKGROUND_COLOR)
            if self.board.is_game_over:
                draw_text("Game over", ((window_size[1]//2)-31*2, (window_size[1]//2)-40),DARK_TEXT_COLOR)
                pygame.display.flip()
                break
            elif self.board.reached_2048:
                draw_text("You won", (window_size[0]//2, window_size[1]//2),DARK_TEXT_COLOR)
            else:
                drawGrid()
            pygame.display.flip()




if __name__ == '__main__':

    env = Game(4)
    env.play_with_pygame((800,800))