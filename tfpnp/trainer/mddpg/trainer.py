import torch
import torch.nn as nn
from torch.optim.adam import Adam
from tensorboardX.writer import SummaryWriter
import time
import os

from ...data.batch import Batch
from ...eval import Evaluator
from ...env import PnPEnv
from ...utils.misc import DataParallel, soft_update, hard_update
from ...utils.rpm import ReplayMemory
from ...utils.log import Logger, COLOR
from ...policy.sync_batchnorm import DataParallelWithCallback
from .critic import ResNet_wobn

DataParallel = DataParallelWithCallback


class MDDPGTrainer:
    def __init__(self, 
                 opt, 
                 env: PnPEnv, 
                 policy, 
                 lr_scheduler,
                 device, 
                 log_dir, 
                 evaluator: Evaluator = None,
                 enable_tensorboard=False, 
                 logger=None
                 ):
        
        #Determines where distributed training or not
        self.data_parallel = True if torch.cuda.device_count() > 1 else False
        self.opt = opt
        #specifies environment
        self.env = env
        self.actor = policy
        #depth of target network is 18 and has one ouput unit
        self.critic = ResNet_wobn(policy.in_dim, 18, 1).to(device)
        #specifies target network
        self.critic_target = ResNet_wobn(policy.in_dim, 18, 1).to(device)
        #parameter
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.writer = SummaryWriter(os.path.join(log_dir, 'tfborad')) if enable_tensorboard else None
        self.device = device
        self.logger = Logger(log_dir) if logger is None else logger

        #TODO: Understand how Replay buffer works
        self.buffer = ReplayMemory(opt.rmsize * opt.max_episode_step)


        #config dictionary of hyperparamters for policy and value neural networks

        self.optimizer_actor = Adam(self.actor.parameters())
        self.optimizer_critic = Adam(self.critic.parameters())

        self.criterion = nn.MSELoss()   # criterion for value loss

        #initialise and copy critic networks entire weights into target critic network
        hard_update(self.critic_target, self.critic)

        self.choose_device()

        self.start_step = 1

    def train(self):
        """
        Train essential works selecting (over 15000 training steps) ->
        From the environment it gets observations and selects actions from these observations.
        When all observations in a batch through maximum episodes or agent opting to terminate PnP algorithm, states are sampled from experience replay buffer
        and actor and critic network learn the best actions through observed reward signals. 
        """
        # reset environment and get initial observation 
        ob = self.env.reset()

        #initiliase policy state
        hidden = hidden_full = self.actor.init_state(ob.shape[0]).to(self.device)
        episode, episode_step = 0, 0
        time_stamp = time.time()

        best_eval_psnr = 0
        
        for step in range(self.start_step, self.opt.train_steps+1):
            # select a action based on the state for each of observations
            #get_policy_b concatenates variables in data loader to make aware of problem setting
            action, hidden = self.run_policy(self.env.get_policy_ob(ob), hidden)

            # take one step in environment each step has five iterations
            _, ob2_masked, _, done, _ = self.env.step(action)
            episode_step += 1

            # store experience to replay buffer: observation and hidden
            self.save_experience(ob, hidden)

            #next states obersation becomes current observation
            ob = ob2_masked
            #get subset
            hidden = hidden_full[self.env.idx_left, ...]

            # if all done or reached maximum number of episode steps -> value network updated at end of one episode 
            if done or (episode_step == self.opt.max_episode_step):

                #steps more than 20 
                if step > 5:
                    #EVALUATION PHRASE?
                    if self.evaluator is not None and (episode+1) % self.opt.validate_interval == 0:
                        #Evaluate perform of PnP model 
                        eval_psnr = self.evaluator.eval(self.actor, step)
                        if eval_psnr > best_eval_psnr:
                            #determine the best psnr
                            best_eval_psnr = eval_psnr
                            self.save_model(self.opt.output, 'best_{:.2f}'.format(best_eval_psnr))
                            self.save_model(self.opt.output, 'best')
                        self.save_model(self.opt.output)
                        

                train_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                result = {'Q': 0, 'dist_entropy': 0, 'critic_loss': 0}

                #update policy network
                if step > self.opt.warmup:
                    result, tb_result = self._update_policy(self.opt.episode_train_times,
                                                            self.opt.env_batch,
                                                            step=step)
                    if self.writer is not None:
                        for k, v in tb_result.items():
                            self.writer.add_scalar(f'train/{k}', v, step)

                # handle logging of training results
                fmt_str = '#{}: Steps: {} - RPM[{}/{}] | interval_time: {:.2f} | train_time: {:.2f} | {}'
                fmt_result = ' | '.join([f'{k}: {v:.2f}' for k, v in result.items()])
                self.logger.log(fmt_str.format(episode, step, self.buffer.size(), self.buffer.capacity,
                                               train_time_interval, time.time()-time_stamp, fmt_result))

                # reset state for next episode, in reset method also get next batch of data!
                ob = self.env.reset()
                #initialise state
                hidden = hidden_full = self.actor.init_state(ob.shape[0]).to(self.device)
                #add episode
                episode += 1
                episode_step = 0
                time_stamp = time.time()

            # save model
            if step % self.opt.save_freq == 0 or step == self.opt.train_steps:
                self.evaluator.eval(self.actor, step)
                self.logger.log('Saving model at Step_{:07d}...'.format(step), color=COLOR.RED)
                self.save_model(self.opt.output, step)

    def _update_policy(self, episode_train_times, env_batch, step):
        """
        
        """

        self.actor.train()

        tot_Q, tot_value_loss, tot_dist_entropy = 0, 0, 0
        tot_actor_norm, tot_critic_norm = 0, 0
        lr = self.lr_scheduler(step)

        for _ in range(episode_train_times):
            #collect samples from environment based on the requested batch size
            samples = self.buffer.sample_batch(env_batch)
            #update actor and critic networks for the samples
            Q, value_loss, dist_entropy, actor_norm, critic_norm = self._update(samples=samples, lr=lr)

            tot_Q += Q
            tot_value_loss += value_loss
            tot_dist_entropy += dist_entropy
            tot_actor_norm += actor_norm
            tot_critic_norm += critic_norm

        mean_Q = tot_Q / episode_train_times
        mean_dist_entropy = tot_dist_entropy / episode_train_times
        mean_value_loss = tot_value_loss / episode_train_times
        mean_actor_norm = tot_actor_norm / episode_train_times
        mean_critic_norm = tot_critic_norm / episode_train_times

        tensorboard_result = {'critic_lr': lr['critic'], 'actor_lr': lr['actor'],
                              'Q': mean_Q, 'dist_entropy': mean_dist_entropy, 'critic_loss': mean_value_loss,
                              'actor_norm': mean_actor_norm, 'critic_norm': mean_critic_norm}
        result = {'Q': mean_Q, 'dist_entropy': mean_dist_entropy, 'closs': mean_value_loss,
                   'anorm': mean_actor_norm, 'cnorm': mean_critic_norm}

        return result, tensorboard_result

    def _update(self, samples, lr: dict):
        #  get lr for actor and critic based on config
        for param_group in self.optimizer_actor.param_groups:
            param_group['lr'] = lr['actor']
        for param_group in self.optimizer_critic.param_groups:
            param_group['lr'] = lr['critic']

        # convert list of named tuple into named tuple of batch
        ob = self.convert2batch(samples)
        hidden = ob.hidden

        #get observation
        policy_ob = self.env.get_policy_ob(ob)

        #implement policy network
        action, action_log_prob, dist_entropy, _ = self.actor(policy_ob, None, True, hidden)

        #iteration of plug and play framework based on chosen actions
        ob2, reward = self.env.forward(ob, action)
        #reward - penalty. Reward is the PSNR 
        reward -= self.opt.loop_penalty

        #get initial state
        eval_ob = self.env.get_eval_ob(ob)
        #get last state
        eval_ob2 = self.env.get_eval_ob(ob2)

        # get value of current observation
        V_cur = self.critic(eval_ob)
        with torch.no_grad():
            # perform based on target network
            V_next_target = self.critic_target(eval_ob2)
            #discounted value network computation
            V_next_target = (
                self.opt.discount * (1 - action['idx_stop'].float())).unsqueeze(-1) * V_next_target
            #compute q value as immediate reward plus sum of future rewards
            Q_target = V_next_target + reward
        #optimise value network based on current reward + target value network  and value network
        advantage = (Q_target - V_cur).clone().detach()
        #discrete computation
        a2c_loss = action_log_prob * advantage

        # compute ddpg loss for continuous actions 
        V_next = self.critic(eval_ob2)
        V_next = (self.opt.discount * (1 - action['idx_stop'].float())).unsqueeze(-1) * V_next
        #compute dpg loss _> continous denoising strength and penalty parameters
        ddpg_loss = V_next + reward

        # compute entroy regularization
        entroy_regularization = dist_entropy

        #compute policy and value network's losss
        policy_loss = - (a2c_loss + ddpg_loss + self.opt.lambda_e * entroy_regularization).mean()
        value_loss = self.criterion(Q_target, V_cur)

        # zero out gradients to prevent them accumulating
        self.actor.zero_grad()
        
        # computes gradient of the loss for the policy network, creates computational graph 
        policy_loss.backward(retain_graph=True)

        
        #clips graidents to prevent from becoming too large
        actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 50)
        #update step of the actor network
        self.optimizer_actor.step()

        #same for the critic network
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 50)
        self.optimizer_critic.step()

        # soft update target network -> exponentially weighted moving average of previous policies
        soft_update(self.critic_target, self.critic, self.opt.tau)

        return -policy_loss.item(), value_loss.item(), entroy_regularization.mean().item(), actor_norm.item(), critic_norm.item()

    def run_policy(self, ob, hidden=None):
        #train or eval effectes whether back propagation occurs through the network
        #this is amended since objective function for policy network is unique
        self.actor.eval()
        with torch.no_grad():
            #based on state runs policy to give actions for termination time, penalty parameter and noise level
            action, _, _, hidden = self.actor(ob, None, True, hidden)
        self.actor.train()
        return action, hidden

    def save_experience(self, ob, hidden):
        """
        Stores each state of PnP algorithm in the array
        """
        for k, v in ob.items():
            if isinstance(v, torch.Tensor):
                #clone and detach object (new tensor with same storage but doesnt require gradient computation)
                ob[k] = ob[k].clone().detach().cpu()

        hidden = hidden.clone().detach().cpu()
        ob['hidden'] = hidden

        B = ob.shape[0]
        for i in range(B):
            #store object in there
            self.buffer.store(ob[i])

    def convert2batch(self, obs):
        batch = Batch.stack(obs)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].to(self.device)
        return batch

    def save_model(self, path, step=None):
        path = os.path.join(path, 'ckpt')
        os.makedirs(path, exist_ok=True)
        if self.data_parallel:
            self.actor = self.actor.module
            self.critic = self.critic.module
            self.critic_target = self.critic_target.module

        self.actor.cpu()
        self.critic.cpu()
        if step is None:
            torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(path))
            torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(path))
        else:
            postfix = step if isinstance(step, str) else '{:07d}'.format(step)
            torch.save(self.actor.state_dict(),
                       '{}/actor_{}.pkl'.format(path, postfix))
            torch.save(self.critic.state_dict(),
                       '{}/critic_{}.pkl'.format(path, postfix))

        self.choose_device()

    def load_model(self, path, step=None):
        if step is None:
            self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
            self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))
        else:
            self.start_step = step
            self.actor.load_state_dict(torch.load('{}/actor_{:07d}.pkl'.format(path, step)))
            self.critic.load_state_dict(torch.load('{}/critic_{:07d}.pkl'.format(path, step)))

    def choose_device(self):
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        if self.data_parallel and type(self.actor) is not DataParallel:
            self.actor = DataParallel(self.actor)
            self.critic = DataParallel(self.critic)
            self.critic_target = DataParallel(self.critic_target)
