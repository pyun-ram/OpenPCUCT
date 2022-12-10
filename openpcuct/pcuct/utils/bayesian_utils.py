from torch import distributions as dists

def sample_from_weight_distribution(
    model,
    mean_dict,
    var_dict,
    modify_key_fn=lambda k: k):
    '''
    Sample a point from the weight distribution,
    which is defined by <mean_dict> and <var_dict>.
    Args:
        mean_dict
        var_dict
        modify_key_fn(lambda fn)
    Returns:
        model
    '''
    return sample_with_diagonal_cov(
        model, mean_dict, var_dict, modify_key_fn)

def sample_with_diagonal_cov(
    model,
    mean_dict,
    var_dict,
    modify_key_fn=lambda k: k):
    '''
    Sample a point from the diagonal Normal posterior distribution,
    which is defined by <mean_dict> and <var_dict>.
    Args:
        mean_dict
        var_dict
        modify_key_fn(lambda fn)
    Returns:
        model
    '''
    std_dict = {k: v**0.5 for k, v in var_dict.items()}
    state_dict = model.state_dict()
    for name, _ in model.named_parameters():
        mean = mean_dict[modify_key_fn(name)]
        std = std_dict[modify_key_fn(name)]
        # only sample those parameters with positive std
        sample_mask = std > 0
        state_dict[name] = mean.clone()
        state_dict[name][sample_mask] = dists.normal.Normal(mean[sample_mask], std[sample_mask]).sample()
    model.load_state_dict(state_dict)
    return model
