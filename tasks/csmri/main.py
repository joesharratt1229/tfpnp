#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from scipy.io import loadmat

from env import CSMRIEnv
from dataset import CSMRIDataset, CSMRIEvalDataset
from solver import create_solver_csmri

from tfpnp.policy.sync_batchnorm import DataParallelWithCallback
from tfpnp.policy import create_policy_network
from tfpnp.pnp import create_denoiser
from tfpnp.trainer import MDDPGTrainer
from tfpnp.eval import Evaluator
from tfpnp.utils.noise import GaussianModelD
from tfpnp.utils.options import Options


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']


def build_evaluator(data_dir, solver):

    sigma_ns = [5, 10, 15]
    #defines a gaussian nose model and is intended to add noise to observations that it recieves
    noise_model = GaussianModelD(sigma_ns)
    mask_dir = data_dir / 'csmri' / 'masks'

    #data directory for training dataset
    train_root = data_dir / 'Images_128'
    masks = [loadmat(mask_dir / f'{sampling_mask}.mat').get('mask') for sampling_mask in sampling_masks]   

    
    dataset = CSMRIEvalDataset(train_root, masks, noise_model)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    eval_env = CSMRIEnv(loader, solver, max_episode_step=opt.max_episode_step).to(device)
    evaluator = Evaluator(eval_env, loader)
    return evaluator


def train(opt, data_dir, mask_dir, policy, solver, log_dir):
    #noise level
    sigma_ns = [5, 10, 15]
    #defines a gaussian nose model and is intended to add noise to observations that it recieves
    noise_model = GaussianModelD(sigma_ns)

    #data directory for training dataset
    train_root = data_dir / 'Images_128'
    masks = [loadmat(mask_dir / f'{sampling_mask}.mat').get('mask') for sampling_mask in sampling_masks]
    #initialises training dataset -> here series of transformations including fast fourier and noise degradation applied
    train_dataset = CSMRIDataset(train_root, fns=None, masks=masks, noise_model=noise_model)
    #data loader observation as specified by pytorch
    train_loader = DataLoader(train_dataset, 8, shuffle=True)
    

    eval = build_evaluator(data_dir, solver, '15', log_dir / 'eval_results')
    
    env = CSMRIEnv(train_loader, solver, max_episode_step=opt.max_episode_step).to(device)

    def lr_scheduler(step):
        if step < 10000:
            return {'critic': 3e-4, 'actor': 1e-3}
        else:
            return {'critic': 1e-4, 'actor': 3e-4}

    trainer = MDDPGTrainer(opt, env, policy,
                           lr_scheduler=lr_scheduler, 
                           device=device,
                           log_dir=log_dir,
                           evaluator=eval, 
                           enable_tensorboard=True)
    if opt.resume:
        trainer.load_model(opt.resume, opt.resume_step)
    trainer.train()


def main(opt):
    data_dir = Path('data')
    log_dir = Path(opt.output)
    mask_dir = data_dir / 'csmri' / 'masks'

    #initialises environment and creates policy network, denoiser, csmri, solver and also initialises distributed training if available
    base_dim = CSMRIEnv.ob_base_dim
    policy = create_policy_network(opt, base_dim).to(device)  # policy network
    denoiser = create_denoiser(opt).to(device)
    solver = create_solver_csmri(opt, denoiser).to(device)
    if torch.cuda.device_count() > 1:
        solver = DataParallelWithCallback(solver)

    if opt.eval:
        ckpt = torch.load(opt.resume)
        policy.load_state_dict(ckpt)
    
        e = build_evaluator(data_dir, solver)
        e.eval(policy, step=opt.resume_step)
        print('--------------------------')
        return 
    #maximum step is 6
    train(opt, data_dir, mask_dir, policy, solver, log_dir)

if __name__ == "__main__":
    option = Options()
    #parse the arguments given in main.Py
    opt = option.parse()
    main(opt)
