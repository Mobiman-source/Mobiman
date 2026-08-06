import torch
from tgnmappo.mappo.actor_critic import Actor, Critic
from tgnmappo.utils.util import update_linear_schedule


class MAPPOPolicy:
    """
    MAPPO Policy with TGN integration.

    Architecture:
      - Actor:  TGN node embedding (per UE) -> MLP -> [handover, target_cell_id]
      - Critic: TGN graph embedding (centralized) -> MLP -> V(s)
      - TGN:    pre-trained via link prediction, fine-tuned jointly with RL

    :param args: arguments containing relevant model and policy information.
    :param obs_space: observation space (UE node embedding dim).
    :param cent_obs_space: centralized observation space (graph embedding dim).
    :param act_space: action space (MultiDiscrete([2, num_cells])).
    :param tgn: TGNStreaming instance (pre-trained, for joint training).
    :param device: torch device.
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, tgn=None, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.tgn = tgn

        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr, eps=self.opti_eps,
            weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay)

        # TGN uses its own optimizer (configured in TGNStreaming.__init__)
        # Joint training happens via tgn.train_step() called from MAPPO trainer
        # i need to change here to end to end

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                    available_actions=None, deterministic=False):
        """
        Compute actions and value predictions.
        :param cent_obs: graph embedding (centralized input for critic).
        :param obs: per-UE node embedding (actor input).
        :param available_actions: per-UE action mask [ho_mask(2) | target_mask(num_cells)].
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
