def score_pool(context):
    """Exploitation-weighted uncertainty bonus with novelty penalty: blend predicted sum and uncertainty, adjusted by progress and penalizing similarity to observed points."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Weight exploitation vs uncertainty: early = more uncertain, late = more exploitative
        w_exploit = 0.3 + 0.7 * (1 - progress)
        base_score = w_exploit * mu_sum + (1 - w_exploit) * sigma_sum
        
        # Add novelty penalty: candidates close to observed points get lower scores
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand["x"], axis=1)
            min_distance = np.min(distances)
            novelty_penalty = np.exp(-min_distance * 5)  # Adjust penalty strength as needed
            score = base_score * novelty_penalty
        else:
            score = base_score
            
        scores.append(score)
    return scores