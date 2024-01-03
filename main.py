import torch
import time
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from collections import deque
from model import PpoAgent
from game import Game
from utils import one_hot_encode


def evaluate_agent(agent, device):
    moves = 0 
    env1 = Game(4)
    env1.reset()
    agent.eval()
    while (env1.board.is_game_over or env1.board.reached_2048) is False:

        state = one_hot_encode(env1.get_board(), env1.board_size)
        t_board = torch.zeros((1, state.shape[0], state.shape[1], state.shape[2])).to(device)
        t_board[0] = state

        action,_,_,_ = agent.get_action_and_value(t_board)
        move  = action.item()
        _, _, _ = env1.step(move)
        moves+=1
        
    agent.train()
    print("BOARD", env1.board.board)
    return np.max(env1.get_board())

class ReplayBuffer:
    def __init__(self, buffer_size=50000):
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        return batch

if __name__ == "__main__":
    writer = SummaryWriter("Ppo_agent_2048")
    replay_mem = ReplayBuffer()
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Game(4)  
    agent = PpoAgent().to(device)

    optimizer = torch.optim.AdamW(agent.parameters(), lr=learning_rate)
    prev_pg_loss = None
    anneal_lr = True
    num_steps = 32768//4
    batch_size = 512

    gamma = 0.998
    gae_lambda = 0.9
    clip_coef = 0.2
    num_updates = 50000 // 2
    ent_coef = 0.0001
    vf_coef = 0.5
    max_grad_norm = 0.5
    global_step = 0

    start_time = time.time()
    encoded_ob = one_hot_encode(env.reset(), env.board_size)
    next_obs = torch.zeros(1, encoded_ob.shape[0], encoded_ob.shape[1], encoded_ob.shape[2]).to(device)
    next_obs[0] = encoded_ob
    next_done = torch.tensor(0).to(device)

   

    for update in range(1, num_updates + 1):
        obs = torch.zeros((num_steps, 16, 4, 4)).to(device)
        actions = torch.zeros((num_steps, 1)).to(device)
        logprobs = torch.zeros((num_steps, 1)).to(device)
        rewards = torch.zeros((num_steps, 1)).to(device)
        dones = torch.zeros((num_steps, 1)).to(device)
        values = torch.zeros((num_steps, 1)).to(device)
        
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        agent.eval()
        for step in range(num_steps):
            global_step += 1

            obs[step] = next_obs[0]
            dones[step] = next_done
        
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, done = env.step(action.item())
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            encoded_ob = one_hot_encode(next_obs, env.board_size)
            next_obs = torch.zeros(1, encoded_ob.shape[0], encoded_ob.shape[1], encoded_ob.shape[2]).to(device)
            next_obs[0] = encoded_ob

            next_done = torch.Tensor(np.array(done).astype(float)).to(device)
        
        
        
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values
        
        b_obs = obs.reshape((-1,) + (env.board_size, env.board_size))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        for step in range(num_steps):
            replay_mem.add_experience([
                obs[step].cpu(), 
                b_logprobs[step].cpu(), 
                b_actions[step].cpu(), 
                b_advantages[step].cpu(),
                b_returns[step].cpu(),
                b_values[step].cpu()
            ])
            

        #numbers for picking random samples from our observations
        clipfracs = []

        agent.train()
        for epoch in range(1):
            
            for start in range(0, 1):
                btch = replay_mem.sample_batch(batch_size)
                obs = torch.zeros((batch_size, 16, 4, 4)).to(device)
                b_logprobs = torch.zeros(batch_size).to(device)
                b_actions = torch.zeros(batch_size).to(device)
                b_advantages = torch.zeros(batch_size).to(device)
                b_returns = torch.zeros(batch_size).to(device)
                b_values = torch.zeros(batch_size).to(device)
                
                for i in range(batch_size):
                    obs[i] = btch[i][0]
                    b_logprobs[i] = btch[i][1]
                    b_actions[i] = btch[i][2]
                    b_advantages[i] = btch[i][3]
                    b_returns[i] = btch[i][4]
                    b_values[i] = btch[i][5]
                
                if (start%100== 0):
                    print("actions", b_actions[0])
               
                _, new_logprob, entropy, new_value = agent.get_action_and_value(obs, b_actions)
                
                log_ratio = new_logprob-b_logprobs
                ratio = log_ratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio-1)-log_ratio).mean()

                    clipfracs += [((ratio-1.0).abs()> clip_coef).float().mean().item()]

                # mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_advantages = b_advantages

                #policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)

                v_loss_unclipped = (new_value - b_returns) ** 2
                v_clipped = b_values + torch.clamp(
                    new_value - b_values,
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = (pg_loss - ent_coef*entropy_loss + v_loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()


            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y ==0 else 1- np.var(y_true - y_pred)/var_y
            episode_reward, moves  = 0, 0
            if global_step > 4e5:
                print("MAX FROM EVAL: ", evaluate_agent(agent, device))
            print("PG LOSS: ", pg_loss.item())
            print("VALUE LOSS:", v_loss.item())
            print("EXPL VAR: ", explained_var)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("losses/episode_reward", episode_reward, global_step)
            writer.add_scalar("losses/moves", moves, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            if prev_pg_loss is None or episode_reward> prev_pg_loss:
                prev_pg_loss = episode_reward
                torch.save(agent.state_dict(), "model.pt")
        print("END EPOCH")
    writer.close()
