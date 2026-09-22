def score_pool(context):
    """Suppress candidates that are too similar to already-observed points, favouring diverse high-acquisition ones."""
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        
        # Compute distance from this candidate to the nearest observed point
        dist_to_observed = np.inf
        if len(context["X_obs"]) > 0:
            distances = np.linalg.norm(context["X_obs"] - cand["x"], axis=1)
            dist_to_observed = np.min(distances)

        # Scale distance by input space range to normalize
        x_range = np.max(context["X_obs"], axis=0) - np.min(context["X_obs"], axis=0)
        if not np.any(x_range == 0):
            normalized_dist = dist_to_observed / np.mean(x_range)
        else:
            normalized_dist = float('inf') if dist_to_observed != 0 else 0.0

        # Suppress candidates that are too close to observed points (novelty penalty)
        novelty_penalty = -1.5 * max(0, 1.0 - np.exp(-normalized_dist))
        
        scores.append(acq + novelty_penalty)

    return scores