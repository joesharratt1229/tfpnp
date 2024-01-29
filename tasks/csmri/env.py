
import torch
from tfpnp.data.batch import Batch
from tfpnp.env import PnPEnv
from tfpnp.utils.transforms import complex2channel, complex2real


class CSMRIEnv(PnPEnv):
    # class attribute: the dimension of ob (exclude solver variable)
    ob_base_dim = 6  
    def __init__(self, data_loader, solver, max_episode_step):
        super().__init__(data_loader, solver, max_episode_step)
    
    def get_policy_ob(self, ob):
        """
        Extracts thr input into policy network based on observation,
        the input dimension into policy network is set to 6 and here we see
        that the ob tensor has 6 dimensions

        """
        
        ob= torch.cat([
            #concatenates x, z, u along with degraded observation -> this incorporates penalty and noise parameters since variables a non-linear function of this
            complex2real(ob.variables),
            #masked fourier transform data with noise -> original k space data
            complex2channel(ob.y0),
            #reconstricuted image
            complex2real(ob.ATy0),
            #sampling mask
            ob.mask,
            #time step
            ob.T,
            #noise added to obtain initial degraded observation
            complex2real(ob.sigma_n),
        ], 1)
        return ob
    
    def get_eval_ob(self, ob):
        return self.get_policy_ob(ob)
    
    def _get_attribute(self, ob, key):
        if key == 'gt':
            return ob.gt
        elif key == 'output':
            return self.solver.get_output(ob.variables)
        elif key == 'input':
            return ob.ATy0
        elif key == 'solver_input':
            return ob.variables, (ob.y0, ob.mask.bool())
        else:
            raise NotImplementedError('key is not supported, ' + str(key))
        
    def _build_next_ob(self, ob, solver_state):
        return Batch(gt=ob.gt,
                     y0=ob.y0,
                     ATy0=ob.ATy0,
                     variables=solver_state,
                     mask=ob.mask,
                     sigma_n=ob.sigma_n,
                     T=ob.T + 1/self.max_episode_step)
    
    def _observation(self):
        idx_left = self.idx_left
        return Batch(gt=self.state['gt'][idx_left, ...],
                     y0=self.state['y0'][idx_left, ...],
                     ATy0=self.state['ATy0'][idx_left, ...],
                     #x, z, u
                     variables=self.state['solver'][idx_left, ...],
                     mask=self.state['mask'][idx_left, ...].float(),
                     sigma_n=self.state['sigma_n'][idx_left, ...],
                     T=self.state['T'][idx_left, ...])
