from model import PpoAgent
from game import Game
import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
if __name__ == "__main__":
    writer = SummaryWriter("Ppo_agent_2048")
    learning_rate=0.05
    env = Game(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PpoAgent(env.board_size, len(env.action_space)).to(device)
    # optimizer_actor = torch.optim.Adam(agent.actor.parameters(), lr=learning_rate)
    # optimizer_critic = torch.optim.Adam(agent.critic.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    prev_pg_loss =None
    anneal_lr = True
    num_steps = 10240
    batch_size = 1024

    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.3
    num_updates = 50000// 2
    ent_coef = 0.1
    vf_coef = 0.5
    max_grad_norm = 0.5
    global_step = 0 
    start_time = time.time()
    
    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, env.board_size*env.board_size)).to(device)
    actions = torch.zeros((num_steps, 1) + np.array(env.action_space).shape).to(device)
    logprobs = torch.zeros((num_steps, 1)).to(device)
    rewards = torch.zeros((num_steps, 1)).to(device)
    dones = torch.zeros((num_steps, 1)).to(device)
    values = torch.zeros((num_steps, 1)).to(device)
    for update in range(1, num_updates + 1):
        # if anneal_lr:
        #     frac = 1.0 - (update-1.0)/num_updates
        #     lrnow = frac *learning_rate
        #     # print(lrnow)
        #     optimizer_critic.param_groups[0]["lr"] = lrnow
        #     optimizer_actor.param_groups[0]["lr"] = lrnow
        next_obs = torch.tensor(env.reset().astype(np.float32)).to(device)
        next_done = torch.tensor(0).to(device)
        obs = []
        actions = []
        logprobs = []
        rewards = []
        dones = []
        values = []
        num_steps = 0
        while env.board.is_game_over is False and env.board.reached_2048 is False:
            num_steps +=1 
            # print(num_steps)
            step =num_steps
            obs.append(next_obs.flatten())
            dones.append(next_done)


            with torch.no_grad():
                ob_batch = torch.zeros((1, env.board_size*env.board_size)).to(device)
                ob_batch[0] = next_obs.flatten()
                agent.eval()
                action, logprob, _, value = agent.get_action_and_value(ob_batch)
            
            values.append(torch.tensor(value)) 
            actions.append(torch.tensor(action)) 
            logprobs.append(torch.tensor(logprob))
            next_obs, reward, done = env.step(action.item())

            rewards.append(torch.tensor(reward))
            next_obs, next_done = torch.Tensor(next_obs.flatten()).to(device), torch.Tensor(np.array(done).astype(float)).to(device)

        print("Episode LENGTH: ", len(obs))
        obs = torch.stack(obs)
        # print(obs.shape)
        actions = torch.stack(actions)
        values = torch.stack(values).flatten()
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        logprobs = torch.stack(logprobs)
        # print()
        with torch.no_grad():
            ob_batch = torch.zeros((1, env.board_size*env.board_size)).to(device)
            ob_batch[0] = next_obs.flatten()
            agent.eval()
            next_value = agent.get_value(ob_batch)
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
        # b_obs = obs.reshape((-1,) + (env.board_size, env.board_size))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        #numbers for picking random samples from our observations
        b_inds = np.arange(num_steps)
        clipfracs = []
        agent.train()
        # print("NUM STEPS", num_steps)
        for epoch in range(10):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, num_steps):
                mb_inds = b_inds[start : start+(num_steps//4)]
                # print("actions", b_actions[mb_inds][0])
                batch = obs
                print(batch.shape)
                _, new_logprob, entropy, new_value = agent.get_action_and_value(batch, b_actions)
                log_ratio = new_logprob-b_logprobs
                ratio = log_ratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio-1)-log_ratio).mean()

                    clipfracs += [((ratio-1.0).abs()> clip_coef).float().mean().item()]

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
                loss = pg_loss - ent_coef*entropy_loss + v_loss + vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()


            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y ==0 else 1- np.var(y_true - y_pred)/var_y
            # writer.add_scalar("charts/learning_rate", optimizer_actor.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            print("VALUE LOSS:", v_loss.item())
            print("POLICY LOSS:", pg_loss.item())
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            if prev_pg_loss is None or abs(v_loss.item())< prev_pg_loss:
                prev_pg_loss = abs(v_loss.item())
                torch.save(agent.state_dict(), "model.pt")
        print("END EPOCH")
    writer.close()
