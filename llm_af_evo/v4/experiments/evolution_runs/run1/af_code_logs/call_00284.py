def score_pool(context):
    """Rank candidates by their novelty in objective space, favoring those farthest from all previously observed outcomes."""
    scores = []
    y_obs = context["Y_obs"]
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        
        # Compute distances to all observations
        dists = np.linalg.norm(y_obs - gp_mean, axis=1)
        
        # Novelty score is the distance to the nearest observation
        novelty_score = np.min(dists)
        scores.append(novelty_score)

    return scores