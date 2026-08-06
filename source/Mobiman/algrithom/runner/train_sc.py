#!/usr/bin/env python
"""
TGN-MAPPO Joint Training for Handover Control.

Training pipeline:
  1. Load pre-trained TGN (link prediction checkpoint)
  2. Create environment with shared TGN (for embedding inference)
  3. Create MAPPO policy with TGN (for joint optimizer setup)
  4. Training loop:
     - Rollout:  TGN produces node/graph embeddings -> Actor decides HO
     - PPO:      update Actor + Critic via clipped surrogate loss
     - TGN:      update via link prediction auxiliary loss
  5. Save all checkpoints (actor, critic, TGN)

Usage:
  python -m tgnmappo.runner.train_sc \
    --algorithm_name mappo \
    --env_name ran \
    --num_ue 8 \
    --num_cell 3 \
    --tgn_checkpoint /tmp/tgn_checkpoint.pt \
    --episode_length 50 \
    --num_env_steps 100000
"""

import os
import sys
import socket
from pathlib import Path

import numpy as np
import torch
import setproctitle

from tgnmappo.config import get_config
from tgnmappo.tgn import TGNStreaming
from tgnmappo.env import TGNUEEnv


def parse_args(args, parser):
    # Mode
    parser.add_argument('--mode', type=str, default='online', choices=['online', 'offline'],
                        help='online: live RAN + UDP commands; offline: replay from DB only')

    # Env / scenario
    parser.add_argument('--scenario_name', type=str, default='multi-cell', help="SC scenario")
    parser.add_argument('--num_cell', type=int, default=3, help='number of cells')
    parser.add_argument('--num_ue', type=int, default=8, help='number of UEs to track as agents')
    parser.add_argument('--a3_offset', type=float, default=3.0, help='A3 offset for action masking')

    # TGN knobs
    parser.add_argument('--ran_db', type=str,
                        default=os.environ.get('RAN_DB', '/path/to/your/oran-sc-ric/data/ran.db'))
    parser.add_argument('--num_du_nodes', type=int, default=16, help='TGN DU node pre-allocation ceiling')
    parser.add_argument('--tgn_neighbor_size', type=int, default=10)
    parser.add_argument('--tgn_time_window_s', type=float, default=1.0)
    parser.add_argument('--tgn_checkpoint', type=str, default='/tmp/tgn_checkpoint.pt',
                        help='Path to pre-trained TGN checkpoint')
    parser.add_argument('--tgn_lr', type=float, default=0.00005,
                        help='TGN fine-tuning learning rate (lower than RL)')
    parser.add_argument('--tgn_loss_coef', type=float, default=0.1,
                        help='TGN auxiliary loss coefficient')
    parser.add_argument('--tgn_train_steps', type=int, default=5,
                        help='Number of TGN training steps per PPO training cycle')
    parser.add_argument('--actor_pretrain', type=str, default='',
                        help='Optional actor pretrain checkpoint (.pt) for MAPPO actor init')

    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # Algorithm switches
    if all_args.algorithm_name == "rmappo":
        print("Using rmappo: use_recurrent_policy=True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("Using mappo: use_recurrent_policy=False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("Using ippo: use_centralized_V=False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError(f"Unknown algorithm: {all_args.algorithm_name}")

    # CUDA
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Run directory
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / \
        all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Wandb
    if all_args.use_wandb:
        import wandb
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=f"{all_args.algorithm_name}_{all_args.experiment_name}_seed{all_args.seed}",
            group=all_args.scenario_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        exst = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir()
                if str(folder.name).startswith('run')]
        curr_run = 'run1' if len(exst) == 0 else f"run{max(exst) + 1}"
        run_dir = run_dir / curr_run
        run_dir.mkdir(parents=True, exist_ok=True)

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
    )

    # Seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # -------------------------------------------------------
    # 1. Create TGN (pre-trained, loaded from checkpoint)
    # -------------------------------------------------------
    print(f"[TGN] Loading pre-trained TGN from: {all_args.tgn_checkpoint}")
    tgn = TGNStreaming(
        db_path=all_args.ran_db,
        num_du_nodes=all_args.num_du_nodes,
        device=device,
        neighbor_size=all_args.tgn_neighbor_size,
        time_window_s=all_args.tgn_time_window_s,
        checkpoint_path=all_args.tgn_checkpoint,
        train_mode=True,  # Enable training mode for joint TGN+RL training
        learning_rate=all_args.tgn_lr,
        online_mode=(all_args.mode == 'online'),
    )

    # -------------------------------------------------------
    # 2. Create environment (shares TGN for inference)
    # -------------------------------------------------------
    print(f"[Mode] {all_args.mode}")
    envs = TGNUEEnv(
        db_path=all_args.ran_db,
        num_ue=all_args.num_ue,
        num_cells=all_args.num_cell,
        embedding_dim=32,
        control_period=0.1,
        a3_offset=all_args.a3_offset,
        tgn=tgn,  # Shared TGN instance
        mode=all_args.mode,
    )

    eval_envs = None
    if all_args.use_eval:
        eval_envs = TGNUEEnv(
            db_path=all_args.ran_db,
            num_ue=all_args.num_ue,
            num_cells=all_args.num_cell,
            embedding_dim=32,
            control_period=0.1,
            a3_offset=all_args.a3_offset,
            tgn=tgn,
            mode=all_args.mode,
        )

    # -------------------------------------------------------
    # 3. Build runner config (TGN passed to policy + trainer)
    # -------------------------------------------------------
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": envs.num_ue,
        "device": device,
        "run_dir": run_dir,
        "tgn": tgn,  # Shared TGN for joint training
    }

    from tgnmappo.runner.sc_runner import scRunner as Runner

    runner = Runner(config)

    if all_args.actor_pretrain:
        ckpt_path = all_args.actor_pretrain
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            actor_state = ckpt.get("actor_state_dict", ckpt)
            model_state = runner.policy.actor.state_dict()
            filtered = {}
            skipped = 0
            for k, v in actor_state.items():
                if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
                    filtered[k] = v
                else:
                    skipped += 1
            incompatible = runner.policy.actor.load_state_dict(filtered, strict=False)
            print(f"[Pretrain] Loaded actor pretrain from: {ckpt_path}")
            print(f"[Pretrain] Matched keys: {len(filtered)}, skipped(incompatible shape/name): {skipped}")
            print(f"[Pretrain] Missing keys: {len(incompatible.missing_keys)}, "
                  f"Unexpected keys: {len(incompatible.unexpected_keys)}")
        else:
            print(f"[Pretrain] actor_pretrain not found: {ckpt_path}")

    print(f"[TGN-MAPPO] Starting joint training:")
    print(f"  Agents (UEs): {envs.num_ue}")
    print(f"  Cells: {all_args.num_cell}")
    print(f"  Action space: MultiDiscrete([2, {all_args.num_cell}]) = [handover, target_cell]")
    print(f"  Actor input: TGN node embedding (dim=32)")
    print(f"  Critic input: TGN graph embedding (dim=32)")
    print(f"  TGN train steps per PPO cycle: {all_args.tgn_train_steps}")
    print(f"  Device: {device}")

    runner.run()

    # -------------------------------------------------------
    # 4. Cleanup
    # -------------------------------------------------------
    try:
        envs.close()
    except Exception:
        pass
    try:
        if eval_envs is not None:
            eval_envs.close()
    except Exception:
        pass
    # Save final TGN checkpoint
    tgn.save_checkpoint()
    tgn.close()

    if all_args.use_wandb:
        run.finish()
    elif hasattr(runner, 'writter'):
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
