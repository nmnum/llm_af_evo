def score_pool(context):
    """Invert acquisition value with uncertainty-based preference to explore diverse yet promising regions."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]

    scores = []
    for cand in context["pool"]:
        acq = 1.0 - cand["acq_value_norm"]  # invert so lower is better
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        
        # Encourage exploration of less confident candidates, but not at expense of quality
        scores.append(acq + (0.3 * sigma_sum))
    
    return scores