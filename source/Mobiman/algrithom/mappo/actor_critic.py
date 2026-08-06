import torch
import torch.nn as nn
from tgnmappo.mappo.utils.util import init, check
from tgnmappo.mappo.utils.mlp import MLPBase
from tgnmappo.mappo.utils.act import ACTLayer
from tgnmappo.mappo.utils.popart import PopArt
from tgnmappo.utils.util import get_shape_from_obs_space


class Actor(nn.Module):
    """
    Actor network for MAPPO.
    Input: TGN node embedding (per UE).
    Output: actions [handover(0/1), target_cell_id] via MultiDiscrete action space.
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: TGN node embedding for this UE.
        :param rnn_states: (unused in MLP mode, kept for API compatibility).
        :param masks: mask tensor.
        :param available_actions: per-UE action mask [ho_mask(2), target_mask(num_cells)].
        :param deterministic: whether to sample or return mode.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy, action_mu, action_std, all_probs = self.act.evaluate_actions_trpo(
                actor_features, action, available_actions,
                active_masks=active_masks if self._use_policy_active_masks else None)
            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features, action, available_actions,
                active_masks=active_masks if self._use_policy_active_masks else None)

        return action_log_probs, dist_entropy


class Critic(nn.Module):
    """
    Centralized Critic network for MAPPO.
    Input: TGN graph embedding (shared across all agents).
    Output: state value V(s).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.base = MLPBase(args, cent_obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute value function from graph embedding.
        :param cent_obs: TGN graph embedding (centralized observation).
        :param rnn_states: (unused in MLP mode, kept for API compatibility).
        :param masks: mask tensor.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        values = self.v_out(critic_features)

        return values, rnn_states
