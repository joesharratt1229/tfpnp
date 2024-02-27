import os
import time
import torch
from os.path import join
from pathlib import Path
import json
import numpy as np
from PIL import Image

from ..utils.visualize import save_img, seq_plot
from ..utils.metric import psnr_qrnn3d
from ..utils.misc import MetricTracker
from ..utils.log import COLOR, Logger
from ..env.base import PnPEnv


class Evaluator(object):
    def __init__(self, env: PnPEnv, loader, metric=psnr_qrnn3d):
        self.env = env
        self.val_loader = loader
        self.metric = metric

        self.output_dir = 'OffPolicy/CSMRI'
        self.image_dir = 'OffPolicy/Image_Dir/CSMRI'

    @torch.no_grad()
    def eval(self, policy, step):
        policy.eval()
        total_metric = 0

        metric_tracker = MetricTracker()
        for index, data in enumerate(self.val_loader):
            assert data['gt'].shape[0] == 1

           
            # run
            psnr_init, psnr_finished, info, imgs, rewards, states = eval_single(self.env, data, policy,
                                                                max_episode_step=self.env.max_episode_step,
                                                                metric=self.metric)

            episode_steps, psnr_seq, action_seqs, run_time = info

            state_dir = []
            
            for traj, state in enumerate(states):
                x, z, u = np.split(state, 3)
                for arr, label in zip([x], ['x']):
                    arr = arr.astype(np.int32).reshape(128, 128)
                    image_path = f'{self.image_dir}/csrmi_{label}_image_{index}_trajectory_{traj}.png'
                    
                    Image.fromarray(arr).save(image_path)
                    state_dir.append(image_path)


            for key in action_seqs.keys():
                action_seqs[key] = [float(value) for value in action_seqs[key]]

            rewards = [float(rew) for rew in rewards]

            result = [rewards[-1] - element for element in rewards][:-1]
            json_file = {'RTG': result, 'Actions' : action_seqs, 'Task': 'csmri', 
                            'Output Mask': {'mu': 1, 'Sigma_d': 1, 'T': 1, 'Tau': 0},
                            'State Paths': state_dir}
            
            with open(f'{self.output_dir}/csmri_{index}.json', 'w+') as file_output:
                json_string = json.dumps(json_file)
                file_output.write(json_string)
            
            
            


def eval_single(env, data, policy, max_episode_step, metric):
    rewards_array = []

    states_array = []


    observation = env.reset(data=data)
    hidden = policy.init_state(observation.shape[0])  # TODO: add RNN support
    _, output_init, gt = env.get_images(observation)


    psnr_init = metric(output_init[0], gt[0])
    output_initial = output_init[0].astype(np.int32)
    temp_z = output_initial.copy()
    temp_u = np.zeros_like(output_initial)
    initial_ob = np.concatenate([output_initial, temp_z, temp_u])

    states_array.append(initial_ob)

    episode_steps = 0

    psnr_seq = [psnr_init]
    action_seqs = {}
    rewards_array.append(psnr_init)

    ob = observation
    time_stamp = time.time()
    while episode_steps < max_episode_step:
        action, time_probs, hidden = policy(env.get_policy_ob(ob), idx_stop=None, train=False, hidden=hidden)

        # since batch size = 1, ob and ob_masked are always identicial
        ob, _, rewards, done, _, states = env.step(action, gt)
        rewards_array.extend(rewards)
        states_array.extend(states)


        episode_steps += 1

        _, output, gt = env.get_images(ob)
        cur_psnr = metric(output[0], gt[0])
        psnr_seq.append(cur_psnr.item())

        action.pop('idx_stop')
        if 'T' not in action_seqs.keys():
            action_seqs['T'] = []

        action_seqs['T'].extend(time_probs)

        for k, v in action.items():
            if k not in action_seqs.keys():
                action_seqs[k] = []
            for i in range(v.shape[0]):
                action_seqs[k] += list(v[i].detach().cpu().numpy())
        if done:
            break

    run_time = time.time() - time_stamp
    input, output, gt = env.get_images(ob)
    psnr_finished = metric(output[0], gt[0])

    info = (episode_steps, psnr_seq, action_seqs, run_time)
    imgs = (input[0], output_init[0], output[0], gt[0])

    return psnr_init, psnr_finished, info, imgs, rewards_array, states_array
