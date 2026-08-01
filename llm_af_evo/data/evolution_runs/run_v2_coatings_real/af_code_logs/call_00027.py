def score_pool(context):
    """Exploitation-weighted uncertainty bonus: blend of predicted mean and uncertainty, tuned by progress, with novelty penalty."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Weight exploitation vs exploration based on progress: early = more explore, late = more exploit
        w_exploit = 0.3 + 0.7 * (1 - progress)
        score = w_exploit * mu_sum + (1 - w_exploit) * sigma_sum
        
        # Add a novelty penalty: candidates close to observed points get penalized
        if len(X_obs) > 0:
            dist_to_obs = np.min(np.linalg.norm(cand["x"] - X_obs, axis=1))
            # Normalize the distance by the range of features (assumed unit hypercube)
            novelty_penalty = 0.05 * dist_to_obs
            score -= novelty_penalty
            
        scores.append(score)
    return scores