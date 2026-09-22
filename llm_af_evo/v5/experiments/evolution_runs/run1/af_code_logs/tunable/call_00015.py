def score_pool(context):
    """Blend acquisition value with novelty and coverage gap scoring, adjusting weights based on campaign progress to balance exploration and exploitation."""
    from scipy.spatial.distance import cdist
    
    names = context["objective_names"]
    pf = context["pareto_front"]
    X_obs = context["X_obs"]
    
    # Compute novelty scores
    dists = np.array([
        np.linalg.norm(X_obs - cand["x"], axis=1).min()
        for cand in context["pool"]
    ])
    d_min, d_max = dists.min(), dists.max()
    if d_max - d_min > 1e-12:
        nov_norm = (dists - d_min) / (d_max - d_min)
    else:
        nov_norm = np.full(len(dists), 0.5)

    # Compute coverage gap scores
    use_pf = len(pf) >= 3
    ref_points = pf if use_pf else context["Y_obs"]
    
    cover_scores = []
    for cand in context["pool"]:
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        dists_to_ref = cdist(pred_obj.reshape(1, -1), ref_points)[0]
        nearest_dists = sorted(dists_to_ref)[:3 if use_pf else min(len(ref_points), 3)]
        cover_scores.append(np.mean(nearest_dists))
    
    # Normalize coverage scores
    cov_min, cov_max = np.min(cover_scores), np.max(cover_scores)
    if cov_max - cov_min > 1e-12:
        cov_norm = (np.array(cover_scores) - cov_min) / (cov_max - cov_min)
    else:
        cov_norm = np.full(len(cover_scores), 0.5)

    # Adjust weights based on campaign progress
    progress = context["campaign"]["progress"]
    
    w_acq = max(0.7, 1.0 - progress * 0.3) 
    w_nov = min(0.3, progress * 0.2)
    w_cov = (1.0 - w_acq - w_nov)

    acq_norm = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    return list(w_acq * acq_norm + 
                w_nov * nov_norm + 
                w_cov * cov_norm)