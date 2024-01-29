import torch.nn as nn
import torch

##THE ABSTRACT METHODS ARE IMPLEMENTED IN DERIVED CHILD CLASSES IN TASKS SECTION FOR EACH TASK ACCORDINGLY


class PnPSolver(nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        #Parent class for PnP solver
        self.denoiser = denoiser

    def reset(self, data):
        """ Resets internal states according to input solver
        """
        raise NotImplementedError

    def forward(self, inputs, parameters, iter_num):
        """ Solves the PnP algorithm for a certain number of steps
        inputs -> input x_k, z_k, u_k
        parameters -> hyperparameters we are using RL to optimise
        iter_num -> number of steps to run PnP algorith for -> default number is 6
        """
        raise NotImplementedError

    def get_output(self, state):
        """ 
        Not implemented method that gets output of intermediate state
        """
        raise NotImplementedError

    def prox_mapping(self, x, sigma):
        ##runs a forward iteration of the UNET DENOISER. It is called a proximal mapping because along with input image
        ## noise is concatenated over the top of the input image which is a necessary step for inverse problems to simulate AWGN
        return self.denoiser(x, sigma)

    @property
    def num_var(self):
        """ Number of the variables to be optimized in this PnP algorithm.
        """
        raise NotImplementedError

    def filter_aux_inputs(self, state):
        """ Filter any auxiliary data except last solver state from the given dictionary. Most likely intermediate states
        are kept within dictionary which are not relvant.
        """
        raise NotImplementedError

    def filter_hyperparameter(self, action):
        """ Filter any hyperparamters needed during the iteration from a given action dictionary.. 
        """
        raise NotImplementedError


class ADMMSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)
    
    @property
    def num_var(self):
        return 3
        
    def reset(self, data):
        """
        This method resets, and initilises states for
        PnP model at time step 0, such that z = x,
        and U = 0
        """
        x = data['x0'].clone().detach() # [B,1,W,H,2]
        z = x.clone().detach()          # [B,1,W,H,2]
        u = torch.zeros_like(x)         # [B,1,W,H,2]
        return torch.cat((x, z, u), dim=1)        
    
    def get_output(self, state):
        """x's shape [B,1,W,H]. Gets the X variable from
        the state tensor which is a concatenation of X, Z, U
        only see X variable""" 
    
        x, _, _ = torch.split(state, state.shape[1] // 3, dim=1)
        return x
    
    def filter_hyperparameter(self, action):
        #Filters hyperparameters when needed
        return action['sigma_d'], action['mu']


class IADMMSolver(ADMMSolver):
    # Inexact ADMM
    def __init__(self, denoiser):
        super().__init__(denoiser)
    
    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['mu'], action['tau']

class HQSSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    @property
    def num_var(self):
        return 2

    def reset(self, data):
        x = data['x0'].clone().detach()
        z = x.clone().detach()
        variables = torch.cat([x, z], dim=1)

        return variables

    def get_output(self, state):
        x, _, = torch.split(state, state.shape[1] // 2, dim=1)
        return x
    
    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['mu']

class PGSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    @property
    def num_var(self):
        return 1

    def reset(self, data):
        x = data['x0'].clone().detach()
        variables = x
        return variables

    def get_output(self, state):
        x = state
        return x
    
    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['tau']

class APGSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)
        import numpy as np
        self.qs = np.zeros(30)
        q = 1
        for i in range(30):
            self.qs[i] = q
            q_prev = q
            q = (1 + (1 + 4 * q_prev**2)**(0.5)) / 2

    @property
    def num_var(self):
        return 2

    def reset(self, data):
        x = data['x0'].clone().detach()
        s = x.clone().detach()
        variables = torch.cat([x, s], dim=1)
        return variables

    def get_output(self, state):
        x, _, = torch.split(state, state.shape[1] // 2, dim=1)
        return x    

    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['tau'], action['beta']


class REDADMMSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    @property
    def num_var(self):
        return 3

    def reset(self, data):
        x = data['x0'].clone().detach()
        z = x.clone().detach()
        u = torch.zeros_like(x)
        variables = torch.cat([x, z, u], dim=1)
        return variables

    def get_output(self, state):
        x, _, _ = torch.split(state, state.shape[1] // 3, dim=1)
        return x

    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['mu'], action['lamda']


class AMPSolver(PnPSolver):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    @property
    def num_var(self):
        return 2        

    def reset(self, data):
        z = data['y0'].clone().detach()
        x = torch.zeros_like(data['x0'])
        variables = torch.cat([x, z], dim=1)

        return variables

    def get_output(self, state):
        x, _ = torch.split(state, state.shape[1] // 2, dim=1)
        return x

    def filter_hyperparameter(self, action):
        return action['sigma_d']
