from model import PpoAgent
from game import Game
import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from utils import one_hot_encode


if __name__ == "__main__":

    num_updates = 25000

    learning_rate=0.0005
    anneal_lr = True

    num_steps = 65536
    batch_size = 128

    gamma = 0.95
    gae_lambda = 0.9
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.6
    global_step = 0 

    writer = SummaryWriter("Ppo_agent_2048")

    env = Game(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PpoAgent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-4)
   



    start_time = time.time()

    encoded_ob = one_hot_encode(env.reset(), env.board_size)
    next_obs = torch.zeros(1, encoded_ob.shape[0], encoded_ob.shape[1], encoded_ob.shape[2])
    next_obs[0] = encoded_ob
    next_done = torch.tensor(0).to(device)

    obs = torch.zeros((num_steps, 16, 4, 4)).to(device)
    actions = torch.zeros((num_steps, 1)).to(device)
    logprobs = torch.zeros((num_steps, 1)).to(device)
    rewards = torch.zeros((num_steps, 1)).to(device)
    dones = torch.zeros((num_steps, 1)).to(device)
    values = torch.zeros((num_steps, 1)).to(device)

    for update in range(1, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update-1.0)/num_updates
            lrnow = frac *learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(num_steps):
            global_step +=1 
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

        b_inds = np.arange(num_steps)
        clipfracs = []

        
        for epoch in range(1):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, batch_size):
                mb_inds = b_inds[start : start+(batch_size)]
                batch = obs[mb_inds]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(batch, b_actions[mb_inds])

                log_ratio = new_logprob-b_logprobs[mb_inds]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio-1)-log_ratio).mean()

                    clipfracs += [((ratio-1.0).abs()> clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)

                v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    new_value - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()

                loss = (pg_loss - ent_coef*entropy_loss + v_loss * vf_coef)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()


            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y ==0 else 1- np.var(y_true - y_pred)/var_y
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            
    writer.close()
