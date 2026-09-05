def score_pool(context):
    """Blend acquisition value with an entropy-based bonus that rewards candidates near under-covered regions of objective space, using mutual information between objectives to guide exploration."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute candidate entropies based on GP posteriors
    entropies = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        entropy = sum(-0.5 * np.log(2 * np.pi * e**2) - 0.5 for name, e in gp.items())
        entropies.append(entropy)
    
    # Normalize acquisition values
    acq_values = [cand['acq_value_norm'] for cand in context["pool"]]
    if len(acq_values) > 1:
        min_acq, max_acq = np.min(acq_values), np.max(acq_values)
        normalized_acqs = [(v - min_acq)/(max_acq - min_acq + 1e-9) for v in acq_values]
    else:
        normalized_acqs = [0.5] * len(acq_values)

    # Adjust scores by entropy and proximity to front
    base_scores = []
    for i, cand in enumerate(context["pool"]):
        gp = cand['gp_posterior']
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = np.mean([gp[name]["std"]/front_range[name] for name in names])
            
        # Reward candidates near under-covered regions using entropy
        score_contribution = normalized_acqs[i]
        if len(context["pareto_front"]) > 0:
            front_dists = [np.linalg.norm(np.array(y) - np.array([gp[n]["mean"] for n in names])) 
                           for y in context['pareto_front']]
            
            # Use entropy to assess how much information we gain near under-covered areas
            coverage_bonus = entropies[i] * (1.0 / (np.mean(front_dists)+ 1e-9))
        else:
            coverage_bonus = 2.5*entropies[i]
        
        final_score = score_contribution + max(coverage_bonus, -score_contribution/3)
        base_scores.append(final_score)

    return [s for s in base_scores]