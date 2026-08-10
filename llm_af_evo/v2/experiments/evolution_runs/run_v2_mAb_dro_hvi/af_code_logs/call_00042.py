def score_pool(context):
    """Exploitation with novelty bonus: rank by predicted objective sum, then add a per-candidate novelty reward based on distance to nearest observed point."""
    names = context["objective_names"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Compute squared Euclidean distance from candidate to the closest observation
        x_cand = cand["x"]
        if len(context["X_obs"]) == 0:
            novelty_score = 1.0  # No observations yet, so all candidates are equally novel
        else:
            distances_sq = np.sum((context["X_obs"] - x_cand) ** 2, axis=1)
            min_distance_sq = np.min(distances_sq)
            # Convert to a score: higher means more novel (smaller distance is less novel).
            novelty_score = 1.0 / (1e-8 + min_distance_sq)

        scores.append(mu_sum + 0.5 * novelty_score)
    return scores