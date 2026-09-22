def score_pool(context):
    """Blend acquisition value with an entropy-based diversity reward that penalizes over-reliance on any single objective's uncertainty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute normalized uncertainty per candidate (entropy-like)
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Entropy-based diversity: sum of relative stds
        total_std_norm = 0.0
        mean_sum = 0.0
        
        for name in names:
            mu, sigma = gp[name]["mean"], gp[name]["std"] 
            total_std_norm += (sigma / front_range[name])
            mean_sum += mu
            
        # Base acquisition value with entropy diversity bonus  
        score = cand["acq_value_norm"]
        
        if len(names) > 1:            
            # Add a small penalty to favor candidates that don't overly rely on one objective's uncertainty
            std_ratios = [gp[name]["std"] / front_range[name] for name in names]
            entropy_penalty = np.std(std_ratios)
            score += -0.2 * entropy_penalty  # Negative: reduce scores of high-uncertainty concentration
            
        scores.append(score)

    return scores