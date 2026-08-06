import time
import os
import numpy as np
import torch
from tgnmappo.utils.shared_buffer import SharedReplayBuffer
from tgnmappo.mappo.MAPPOPolicy import MAPPOPolicy
from tgnmappo.mappo.mappo import MAPPO


def _t2n(x):
    return x.detach().cpu().numpy()


class scRunner:
    """
    Runner for TGN-MAPPO handover control.

    Architecture flow per training cycle:
      1. Rollout: TGN.step() -> node/graph embeddings -> Actor -> HO decisions
      2. Train:   PPO update (actor + critic) + TGN auxiliary update (link prediction)
      3. Both models are updated simultaneously via their respective losses.
    """
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config.get("eval_envs", None)
        self.num_agents = config["num_agents"]
        self.device = config["device"]
        self.run_dir = config.get("run_dir", None)
        self.tgn = config.get("tgn", None)  # TGN model for joint training

        # Extract commonly used args
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.episode_length = self.all_args.episode_length
        self.num_env_steps = int(self.all_args.num_env_steps)
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.env_name = self.all_args.env_name
        self.use_eval = self.all_args.use_eval
        self.eval_interval = getattr(self.all_args, 'eval_interval', 25)
        self.recurrent_N = self.all_args.recurrent_N
        self.hidden_size = self.all_args.hidden_size

        # For single env, n_rollout_threads=1
        self.n_rollout_threads = 1
        self.all_args.n_rollout_threads = 1

        # Setup logging
        if self.run_dir is not None:
            self.log_dir = str(self.run_dir / 'logs')
            os.makedirs(self.log_dir, exist_ok=True)
            self.save_dir = str(self.run_dir / 'models')
            os.makedirs(self.save_dir, exist_ok=True)

        # Use wandb or tensorboard
        self.use_wandb = self.all_args.use_wandb
        if not self.use_wandb and self.run_dir is not None:
            from tensorboardX import SummaryWriter
            self.writter = SummaryWriter(self.log_dir)

        # Build policy + trainer
        obs_space = self.envs.observation_space[0]
        act_space = self.envs.action_space[0]
        share_obs_space = self.envs.share_observation_space[0]

        self.policy = MAPPOPolicy(
            self.all_args, obs_space, share_obs_space, act_space,
            tgn=self.tgn, device=self.device)

        self.trainer = MAPPO(
            self.all_args, self.policy, tgn=self.tgn, device=self.device)

        self.buffer = None

    def _init_buffer(self):
        agent_obs_space = self.envs.observation_space[0]
        agent_action_space = self.envs.action_space[0]
        agent_share_obs_space = self.envs.share_observation_space[0]
        self.buffer = SharedReplayBuffer(
            self.all_args, self.num_agents,
            agent_obs_space, agent_share_obs_space, agent_action_space)

    def _build_share_obs(self, obs):
        """Build centralized observations: concat(graph_emb, ue_node_emb) per agent.

        obs shape: (n_rollout_threads, num_agents, embedding_dim)
        output shape: (n_rollout_threads, num_agents, embedding_dim * 2)
        """
        if not self.use_centralized_V:
            return obs
        graph_emb = None
        if hasattr(self.envs, "get_graph_emb"):
            graph_emb = self.envs.get_graph_emb()
        elif hasattr(self.envs, "graph_emb_last"):
            graph_emb = self.envs.graph_emb_last
        if graph_emb is None:
            # Fallback: duplicate obs as share_obs
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            return np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        if isinstance(graph_emb, torch.Tensor):
            graph_emb = graph_emb.detach().cpu().numpy()
        graph_emb = np.asarray(graph_emb, dtype=np.float32).flatten()  # (embedding_dim,)

        # obs: (n_rollout_threads, num_agents, embedding_dim)
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 2:
            obs_arr = obs_arr[np.newaxis, ...]  # -> (1, num_agents, embedding_dim)

        # tile graph_emb to match (n_rollout_threads, num_agents, embedding_dim)
        graph_emb_tiled = np.tile(
            graph_emb, (obs_arr.shape[0], obs_arr.shape[1], 1)
        )  # (n_rollout_threads, num_agents, embedding_dim)

        # concat along feature dim -> (n_rollout_threads, num_agents, embedding_dim * 2)
        return np.concatenate([graph_emb_tiled, obs_arr], axis=-1)

    def run(self):
        self.warmup()

        start_time = time.time()
        total_steps = 0
        segment_step = 0

        while total_steps < self.num_env_steps:
            if self.use_linear_lr_decay:
                frac = total_steps / float(self.num_env_steps) if self.num_env_steps > 0 else 0.0
                self.policy.lr_decay(frac, 1.0)

            values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, available_actions = self.collect(segment_step)
            obs, rewards, dones, infos = self.envs.step(actions_env)
            # Single-env setup: when all agents are done, reset immediately so rollout does not
            # keep consuming terminal observations in offline mode.
            if np.all(dones):
                obs = self.envs.reset()
            self.insert((obs, rewards, dones, infos, values, actions,
                         action_log_probs, rnn_states, rnn_states_critic, available_actions))
            segment_step += 1
            total_steps += self.n_rollout_threads

            if segment_step >= self.episode_length or total_steps >= self.num_env_steps:
                self.compute()
                train_infos = self.train()

                tgn_loss = train_infos.get('tgn_loss', 0.0)
                print(f"[Train] steps={total_steps}, "
                      f"policy_loss={train_infos['policy_loss']:.4f}, "
                      f"value_loss={train_infos['value_loss']:.4f}, "
                      f"tgn_loss={tgn_loss:.4f}")

                if (total_steps // self.n_rollout_threads) % self.save_interval == 0 or total_steps >= self.num_env_steps:
                    self.save()

                # Logging
                if (total_steps // self.n_rollout_threads) % self.log_interval == 0:
                    elapsed = time.time() - start_time
                    fps = int(total_steps / elapsed) if elapsed > 0 else 0
                    print(f"Steps {total_steps}/{self.num_env_steps}, FPS {fps}.")
                    env_infos = {}
                    if isinstance(infos, dict):
                        infos_list = [infos]
                    else:
                        infos_list = infos
                    for aid in range(self.num_agents):
                        rews = [
                            step_info.get(aid, {}).get('reward', 0)
                            for step_info in infos_list
                        ]
                        env_infos[f"agent{aid}/individual_rewards"] = rews
                    train_infos.update(env_infos)
                    train_infos["avg_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                    self.log_train(train_infos, total_steps)

                # Reset buffer for next segment
                self._init_buffer()
                current_obs = obs
                share_obs = self._build_share_obs(current_obs)
                self.buffer.share_obs[0] = share_obs.copy()
                self.buffer.obs[0] = current_obs.copy()
                if hasattr(self.buffer, 'rnn_states') and self.buffer.rnn_states is not None and self.buffer.rnn_states.size > 0:
                    self.buffer.rnn_states[0].fill(0)
                    self.buffer.rnn_states_critic[0].fill(0)
                    self.buffer.masks[0].fill(1)
                segment_step = 0

                if self.use_eval and total_steps % self.eval_interval == 0:
                    self.eval(total_steps)

    def warmup(self):
        self._init_buffer()
        obs = self.envs.reset()
        if obs.ndim == 3:
            self.num_agents = obs.shape[1]
        else:
            self.num_agents = obs.shape[0]

        share_obs = self._build_share_obs(obs)
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        available_actions = None
        if hasattr(self.envs, "get_available_actions"):
            available_actions = self.envs.get_available_actions()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
                available_actions=np.concatenate(available_actions) if available_actions is not None else None)

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # Convert MultiDiscrete actions to one-hot for env
        action_space = self.envs.action_space[0]
        nvec = action_space.nvec
        for i in range(len(nvec)):
            uc_actions_env = np.eye(nvec[i])[actions[:, :, i].astype(int)]
            if i == 0:
                actions_env = uc_actions_env
            else:
                actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, available_actions

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = self._build_share_obs(obs)

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                           values, rewards, masks, available_actions=available_actions)

    def compute(self):
        """Compute returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Run PPO + TGN joint training."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        return train_infos

    def save(self):
        """Save actor, critic, and TGN checkpoints."""
        if not hasattr(self, 'save_dir') or self.save_dir is None:
            return
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        # Save TGN checkpoint
        if self.tgn is not None:
            tgn_path = str(self.save_dir) + "/tgn_checkpoint.pt"
            self.tgn.save_checkpoint(tgn_path)

    def log_train(self, train_infos, total_num_steps):
        """Log training info."""
        if self.use_wandb:
            import wandb
            for k, v in train_infos.items():
                if isinstance(v, (int, float, np.floating)):
                    wandb.log({k: v}, step=total_num_steps)
        elif hasattr(self, 'writter'):
            for k, v in train_infos.items():
                if isinstance(v, (int, float, np.floating)):
                    self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """Log environment info."""
        if self.use_wandb:
            import wandb
            for k, v in env_infos.items():
                if isinstance(v, list) and len(v) > 0:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
        elif hasattr(self, 'writter'):
            for k, v in env_infos.items():
                if isinstance(v, list) and len(v) > 0:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        if self.eval_envs is None:
            return
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_available_actions = None
            if hasattr(self.eval_envs, "get_available_actions"):
                eval_available_actions = self.eval_envs.get_available_actions()

            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                available_actions=np.concatenate(eval_available_actions) if eval_available_actions is not None else None,
                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), 1))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), 1))

            # Convert to one-hot
            action_space = self.eval_envs.action_space[0]
            nvec = action_space.nvec
            for i in range(len(nvec)):
                eval_uc_actions_env = np.eye(nvec[i])[eval_actions[:, :, i].astype(int)]
                if i == 0:
                    eval_actions_env = eval_uc_actions_env
                else:
                    eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_average = np.mean(np.sum(eval_episode_rewards, axis=0))
        print(f"[Eval] step={total_num_steps}, avg_reward={eval_average:.4f}")
        self.log_env({"eval_average_episode_rewards": [eval_average]}, total_num_steps)
