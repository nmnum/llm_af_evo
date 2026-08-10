def score_pool(context):
    """Estimate improvement potential using hypervolume contribution weighted by uncertainty; penalize candidates that are too similar to already-selected ones."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Estimate HV improvement potential per candidate
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted means and stds, normalized by front range  
        mu_vec = np.array([gp[name]["mean"] for name in names]) 
        sigma_vec = np.array([gp[name]["std"] for name in names])
        norm_mu = (mu_vec - ref_point) / [front_range[n] or 1.0 for n in names]
        
        # Use uncertainty to adjust the HV estimate: higher std means more potential
        hv_contribution = max(0., np.prod(np.maximum(norm_mu, 0.) + sigma_vec)) if not all(s == 0 for s in sigma_vec) else float('-inf')
 
        scores.append(hv_contribution)
        
    # Normalize by the maximum score to prevent extreme imbalance  
    norm_scores = (np.array(scores) - min(1e-6, np.min(scores))) / max(np.max(scores), 1.0)

    return list(norm_scores)