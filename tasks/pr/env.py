
import torch
from tfpnp.data.batch import Batch
from tfpnp.env import PnPEnv
from tfpnp.utils.transforms import complex2channel, complex2real


class PREnv(PnPEnv):
    def __init__(self, data_loader, solver, max_episode_step, device):
        super().__init__(data_loader, solver, max_episode_step, device)
    
    def get_policy_state(self, state):
        return torch.cat([
            complex2real(state.variables),
            state.y0,
            complex2channel(state.mask),
            state.T,
            state.sigma_n,
        ], 1)
    
    def get_eval_state(self, state):
        return self.get_policy_state(state)
    
    def _get_attribute(self, state, key):
        if key == 'gt':
            return state.gt
        elif key == 'output':
            return self.solver.get_output(state.variables)
        elif key == 'input':
            return state.x0
        elif key == 'solver_input':
            return (state.variables, (state.y0, state.mask))
        else:
            raise NotImplementedError('key is not supported, ' + str(key))
        
    def _build_next_state(self, state, solver_state):
        return Batch(gt=state.gt,
                     y0=state.y0,
                     x0=state.x0,
                     variables=solver_state,
                     mask=state.mask,
                     sigma_n=state.sigma_n,
                     T=state.T + 1/self.max_episode_step)
    
    def _observation(self):
        idx_left = self.idx_left
        return Batch(gt=self.state['gt'][idx_left, ...],
                     y0=self.state['y0'][idx_left, ...],
                     x0=self.state['x0'][idx_left, ...],
                     variables=self.state['solver'][idx_left, ...],
                     mask=self.state['mask'][idx_left, ...].float(),
                     sigma_n=self.state['sigma_n'][idx_left, ...],
                     T=self.state['T'][idx_left, ...])