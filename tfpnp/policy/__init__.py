from .network import ResNetActor_ADMM, ResNetActor_HQS, ResNetActor_PG, ResNetActor_APG, ResNetActor_IADMM, ResNetActor_RED, ResNetActor_AMP, ResNetActor_SPI


_policy_map = {
    'admm': ResNetActor_ADMM,
    'hqs': ResNetActor_HQS,
    'pg': ResNetActor_PG,
    'apg': ResNetActor_APG,
    'redadmm': ResNetActor_RED,
    'amp': ResNetActor_AMP,
    'iadmm': ResNetActor_IADMM,
    'admm_spi': ResNetActor_SPI  
}


def create_policy_network(opt, num_aux_inputs, action_range=None):
    if opt.solver in _policy_map:
        #gets policy based on argument parsed in user input
        Policy = _policy_map[opt.solver]
        #encodes additional inputs and action pack
        #num of auxillary inputs is set to 6
        actor = Policy(num_aux_inputs, opt.action_pack, action_range)
    else:
        raise NotImplementedError
    return actor
