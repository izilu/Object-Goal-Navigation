from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np

from model import TSOG_RL_Policy
from utils.storage import GlobalRolloutStorage, TSOGGlobalRolloutStorage
from envs import make_vec_envs
from arguments import get_args
import algo

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes     # 默认 5
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    g_masks = torch.ones(num_scenes).float().to(device)

    best_g_reward = -np.inf

    if args.eval:
        episode_success = []
        episode_spl = []
        episode_dist = []
        for _ in range(args.num_processes):
            episode_success.append(deque(maxlen=num_episodes))
            episode_spl.append(deque(maxlen=num_episodes))
            episode_dist.append(deque(maxlen=num_episodes))

    else:
        episode_success = deque(maxlen=1000)
        episode_spl = deque(maxlen=1000)
        episode_dist = deque(maxlen=1000)

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)      # spawn n 个环境，n 个子进程
    state, detection_results, infos = envs.reset()

    torch.set_grad_enabled(False)

    # Global policy
    g_policy = TSOG_RL_Policy(args, model_type=1).to(device)
    g_agent = algo.TSOGPPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    # Storage
    g_rollouts = TSOGGlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, (15, 1024+1+4), (512, 7, 7), 
                                      args.hidden_state_sz, args.gat_memory_len).to(device)

    if args.load != "0":
        print("Loading model {}".format(args.load))
        state_dict = torch.load(args.load,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if args.eval:
        g_policy.eval()
    
    # 物品 id
    goal_cat_id = torch.from_numpy(np.asarray(
        [infos[env_idx]['goal_cat_id'] for env_idx
         in range(num_scenes)]))        # (25,)

    g_rollouts.states[0].copy_(state)
    g_rollouts.detection_results[0].copy_(detection_results)
    g_rollouts.goal_cat_ids[0].copy_(goal_cat_id)

    # Run Global Policy (global_goals = Long-Term Goal)
    action_int, g_value, g_action, g_action_log_prob, g_rec_states, gat_embedding_memory = \
        g_policy.act(
            g_rollouts.states[0],
            g_rollouts.goal_cat_ids[0],
            (g_rollouts.rec_states[0][:, 0].transpose(0, 1), g_rollouts.rec_states[0][:, 1].transpose(0, 1)),
            g_rollouts.actions[0],
            g_rollouts.detection_results[0],
            g_rollouts.gat_embedding_memorys[0],
            training=False
        )

    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['new_goal'] = 1
        p_input['wait'] = wait_env[e] or finished[e]
        p_input['action'] = action_int[e]

    state, detection_results, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

    start = time.time()
    g_reward = 0

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break
        # num_global_steps：A2C 中前进步数，即每次更新模型的步数。 num_local_steps：每个 num_global_steps 之间的步数
        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1
                                     for x in done]).to(device)     # 统计当前step对应的scene的episode是不是完成了，local
        g_masks *= l_masks

        for e, x in enumerate(done):
            if x:
                spl = infos[e]['spl']
                success = infos[e]['success']
                dist = infos[e]['distance_to_goal']
                spl_per_category[infos[e]['goal_name']].append(spl)
                success_per_category[infos[e]['goal_name']].append(success)
                if args.eval:
                    episode_success[e].append(success)
                    episode_spl[e].append(spl)
                    episode_dist[e].append(dist)
                    if len(episode_success[e]) == num_episodes:
                        finished[e] = 1
                else:
                    episode_success.append(success)
                    episode_spl.append(spl)
                    episode_dist.append(dist)
                wait_env[e] = 1.
                g_rollouts.reset_gat_mem(e)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Global Policy
        # 每个 num_local_steps 更新一次全局策略
        if l_step == args.num_local_steps - 1:
            # For every global step, update the full and local maps
            for e in range(num_scenes):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.

            goal_cat_id = torch.from_numpy(np.asarray(
                [infos[env_idx]['goal_cat_id'] for env_idx
                 in range(num_scenes)]))        # (25,)

            g_reward = torch.from_numpy(np.asarray(
                [infos[env_idx]['g_reward'] for env_idx in range(num_scenes)])
            ).float().to(device)
            g_process_rewards += g_reward.cpu().numpy()
            g_total_rewards = g_process_rewards * \
                (1 - g_masks.cpu().numpy())
            g_process_rewards *= g_masks.cpu().numpy()
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

            if np.sum(g_total_rewards) != 0:
                for total_rew in g_total_rewards:
                    if total_rew != 0:
                        g_episode_rewards.append(total_rew)

            # Add samples to global policy storage
            if step == 0:
                g_rollouts.states[0].copy_(state)
                g_rollouts.detection_results[0].copy_(detection_results)
                g_rollouts.goal_cat_ids[0].copy_(goal_cat_id)
            else:
                g_rollouts.insert(
                    state, detection_results, g_rec_states,
                    g_action, g_action_log_prob, g_value,
                    g_reward, goal_cat_id, g_masks, gat_embedding_memory
                )

            # Sample long-term goal from global policy
            action_int, g_value, g_action, g_action_log_prob, g_rec_states, gat_embedding_memory = \
                g_policy.act(
                    g_rollouts.states[g_step + 1],
                    g_rollouts.goal_cat_ids[g_step + 1],
                    (g_rollouts.rec_states[g_step + 1][:, 0].transpose(0, 1), g_rollouts.rec_states[0][:, 1].transpose(0, 1)),
                    g_rollouts.actions[g_step],
                    g_rollouts.detection_results[g_step + 1],
                    g_rollouts.gat_embedding_memorys[g_step + 1],
                    training=False
                )

            g_reward = 0
            g_masks = torch.ones(num_scenes).float().to(device)     # 重置 mask
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['new_goal'] = l_step == args.num_local_steps - 1
            p_input['wait'] = wait_env[e] or finished[e]
            p_input['action'] = action_int[e]

        state, detection_results, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Training
        torch.set_grad_enabled(True)
        if g_step % args.num_global_steps == args.num_global_steps - 1 \
                and l_step == args.num_local_steps - 1:
            if not args.eval:
                g_next_value = g_policy.get_value(
                    g_rollouts.states[-1],
                    g_rollouts.goal_cat_ids[-1],
                    (g_rollouts.rec_states[-1][:, 0].transpose(0, 1), g_rollouts.rec_states[0][:, 1].transpose(0, 1)),
                    g_rollouts.actions[-1],
                    g_rollouts.detection_results[-1],
                    g_rollouts.gat_embedding_memorys[-1],
                ).detach()

                g_rollouts.compute_returns(g_next_value, args.use_gae,
                                           args.gamma, args.tau)
                g_value_loss, g_action_loss, g_dist_entropy = \
                    g_agent.update(g_rollouts)     # 梯度反向传播
                g_value_losses.append(g_value_loss)
                g_action_losses.append(g_action_loss)
                g_dist_entropies.append(g_dist_entropy)
            g_rollouts.after_update()

        torch.set_grad_enabled(False)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging
        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(step * num_scenes),
                "FPS {},".format(int(step * num_scenes / (end - start)))
            ])

            log += "\n\tRewards:"

            if len(g_episode_rewards) > 0:
                log += " ".join([
                    " Global step mean/med rew:",
                    "{:.4f}/{:.4f},".format(
                        np.mean(per_step_g_rewards),
                        np.median(per_step_g_rewards)),
                    " Global eps mean/med/min/max eps rew:",
                    "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(g_episode_rewards),
                        np.median(g_episode_rewards),
                        np.min(g_episode_rewards),
                        np.max(g_episode_rewards))
                ])

            if args.eval:
                total_success = []
                total_spl = []
                total_dist = []
                for e in range(args.num_processes):
                    for acc in episode_success[e]:
                        total_success.append(acc)
                    for dist in episode_dist[e]:
                        total_dist.append(dist)
                    for spl in episode_spl[e]:
                        total_spl.append(spl)

                if len(total_spl) > 0:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(total_success),
                        np.mean(total_spl),
                        np.mean(total_dist),
                        len(total_spl))
            else:
                if len(episode_success) > 100:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(episode_success),
                        np.mean(episode_spl),
                        np.mean(episode_dist),
                        len(episode_spl))

            log += "\n\tLosses:"
            if len(g_value_losses) > 0 and not args.eval:
                log += " ".join([
                    " Policy Loss value/action/dist:",
                    "{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(g_value_losses),
                        np.mean(g_action_losses),
                        np.mean(g_dist_entropies))
                ])

            print(log)
            logging.info(log)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Save best models
        if (step * num_scenes) % args.save_interval < \
                num_scenes:
            if len(g_episode_rewards) >= 1000 and \
                    (np.mean(g_episode_rewards) >= best_g_reward) \
                    and not args.eval:
                torch.save(g_policy.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))
                best_g_reward = np.mean(g_episode_rewards)

        # Save periodic models
        if (step * num_scenes) % args.save_periodic < \
                num_scenes:
            total_steps = step * num_scenes
            if not args.eval:
                torch.save(g_policy.state_dict(),
                           os.path.join(dump_dir,
                                        "periodic_{}.pth".format(total_steps)))
        # ------------------------------------------------------------------

    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")
        
        total_success = []
        total_spl = []
        total_dist = []
        for e in range(args.num_processes):
            for acc in episode_success[e]:
                total_success.append(acc)
            for dist in episode_dist[e]:
                total_dist.append(dist)
            for spl in episode_spl[e]:
                total_spl.append(spl)

        if len(total_spl) > 0:
            log = "Final ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                np.mean(total_success),
                np.mean(total_spl),
                np.mean(total_dist),
                len(total_spl))

        print(log)
        logging.info(log)
            
        # Save the spl per category
        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(key,
                                          sum(success_per_category[key]) /
                                          len(success_per_category[key]),
                                          sum(spl_per_category[key]) /
                                          len(spl_per_category[key]))

        print(log)
        logging.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)


if __name__ == "__main__":
    main()
