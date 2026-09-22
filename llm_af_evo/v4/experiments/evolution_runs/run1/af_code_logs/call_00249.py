def score_pool(context):
    """Invert acquisition value scaling with uncertainty to prefer low-confidence candidates that still show high hypervolume potential."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        
        # Prefer candidates with low acquisition value but high uncertainty,
        # i.e., those that are not yet well-characterized by the model
        if acq > 0:
            scores.append((1.0 - sigma_sum) * (acq ** 0.5))
        else:
            scores.append(0.0)
    
    return scores