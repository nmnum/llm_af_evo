def score_pool(context):
    """Blend acquisition value with an entropy-based exploration bonus that scales adaptively based on campaign progress and front density."""
    names = context["objective_names"]
    ref_point = np.array(context["ref_point"])
    front_range = context["pareto_front_range"]
    
    # Compute a proxy for the current Pareto front's density
    if len(context['pareto_front']) >= 2:
        front_diffs = np.diff(np.sort(context['pareto_front'], axis=0), axis=0)
        avg_density = np.mean(front_diffs) / max(1e-6, min(front_range[name] for name in names))
    else:
        # Fallback when not enough points to compute density
        avg_density = 1.0

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        acq_val = cand['acq_value_norm']
        
        # Entropy-based uncertainty score: product of standard deviations (higher is more uncertain)
        sigma_product = np.prod([gp[name]["std"] / front_range[name] for name in names])
                
        # Progress-aware scaling factor that decreases exploration as campaign progresses
        progress_factor = 1.0 - context["campaign"]["progress"]
        
        # Adaptive entropy bonus: scale by both uncertainty and how early we are, 
        # but penalize dense regions (low density is more informative)
        adaptive_bonus = sigma_product * progress_factor / max(1e-6, avg_density)

        scores.append(acq_val + 0.5 * adaptive_bonus) 

    return scores