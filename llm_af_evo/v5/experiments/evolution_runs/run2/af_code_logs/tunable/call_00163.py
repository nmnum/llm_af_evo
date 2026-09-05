def score_pool(context):
    """Blend acquisition value with a progress-aware uncertainty bonus that scales inversely with observed improvement rate and rewards candidates near underexplored regions of objective space."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute the average improvement per observation so far
    n_obs = len(context['Y_obs'])
    if n_obs < 2:
        avg_improvement = 1.0
    else:
        improvements = np.diff(np.max(context['Y_obs'], axis=0))
        avg_improvement = np.mean(improvements) + 1e-9

    # Scale uncertainty bonus by inverse of improvement rate (lower progress -> higher reward)
    scale_factor = min(5.0, max(0.2, 1.0 / avg_improvement))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Base acquisition value
        base_score = cand['acq_value_norm']
        
        # Compute uncertainty bonus: normalized sum of stds 
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Add scaled uncertainty term, but only if candidate is not too close to existing observations  
        distances_to_observed = [
            np.min(np.linalg.norm(cand["x"] - context['X_obs'], axis=1)) 
            if len(context['X_obs']) > 0 else float('inf')
        ][0]
        
        # Reward candidates in sparse regions (large distance from nearest observation)
        novelty_bonus = max(0.0, distances_to_observed / (np.linalg.norm(front_range) + 1e-9))
            
        score = base_score + scale_factor * sigma_sum * novelty_bonus
        scores.append(score)

    return scores