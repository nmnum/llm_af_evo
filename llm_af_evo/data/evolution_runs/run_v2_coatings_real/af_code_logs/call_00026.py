def score_pool(context):
    """Exploitation-weighted uncertainty bonus with dynamic blending and novelty penalty."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Dynamic exploitation weight: early = more explore, late = more exploit
        w_exploit = 0.3 + 0.7 * (1 - progress)
        # Combine exploitation and exploration
        score = w_exploit * mu_sum + (1 - w_exploit) * sigma_sum
        # Add novelty penalty: candidates closer to observed points get lower scores
        if len(X_obs) > 0:
            dist_to_obs = np.min(np.linalg.norm(cand["x"] - X_obs, axis=1))
            score -= 0.1 * dist_to_obs  # Penalty scales with proximity
        scores.append(score)
    return scores