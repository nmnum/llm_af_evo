def score_pool(context):
    """Diversity-promoting score based on repelling previously selected points and penalizing duplicate regions."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    progress = context["campaign"]["progress"]

    # Use a noise-aware resampling to estimate how much each candidate would improve the hypervolume
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Estimate uncertainty-normalized mean (as baseline)
        mu_sum = sum(gp[name]["mean"] for name in names) 
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        # Add a repulsion term: how close is this candidate to already observed points?
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            min_distance = float(distances.min())
            
            # Scale penalty by progress — less penalize later stages
            repulsion_penalty = (min_distance / max(1e-6, np.sqrt(len(names)))) * (1.0 - progress) 
        else:
            repulsion_penalty = 0.0

        score = mu_sum + sigma_norm - repulsion_penalty  
        scores.append(score)

    return scores