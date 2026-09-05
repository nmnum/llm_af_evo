def score_pool(context):
    """Integrate hypervolume acquisition with dynamic uncertainty-aware exploration and progress-driven exploitation."""
    names = context["objective_names"]
    
    # Compute GP means for all candidates 
    gp_means = np.array([[cand["gp_posterior"][name]["mean"] for name in names] for cand in context["pool"]])
    
    scores = []
    campaign_progress = context["campaign"]["progress"]

    for i, cand in enumerate(context["pool"]):
        acq_value_norm = cand["acq_value_norm"]
        
        # Base score on acquisition value
        base_score = acq_value_norm
        
        # Add uncertainty component that adapts based on progress and front sparsity 
        gp_std = np.array([cand["gp_posterior"][name]["std"] for name in names])
        ucb_bonus = 2.0 * np.mean(gp_std) / max(np.std(context["Y_obs"], axis=0)) if len(context["Y_obs"]) > 1 else 0
        
        # Dynamic weighting between exploitation and exploration based on progress
        exploit_weight = min(1.0, campaign_progress ** -0.5)
        
        final_score = (exploit_weight * base_score + 
                      (1 - exploit_weight) * ucb_bonus)

        scores.append(final_score)
    
    return scores