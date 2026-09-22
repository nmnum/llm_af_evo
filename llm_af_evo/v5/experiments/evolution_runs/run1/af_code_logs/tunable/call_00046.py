def score_pool(context):
    """Score candidates based on how much they expand under-covered regions of the Pareto front by incorporating a coverage-gap term into the acquisition value."""
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"] 
    X_obs = context["X_obs"]
    
    # Use Y_obs if pareto_front is too small
    use_y_obs = len(pf) < 3
    ref_points = pf if not use_y_obs else context["Y_obs"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        pred_obj = np.array([gp[name]["mean"] for name in names])
        
        # Compute distances to nearest reference points
        dists = [np.linalg.norm(pred_obj - ref_pt) for ref_pt in ref_points]
        k_nearest_dists = sorted(dists)[:3]  # Top-3 smallest distances
        
        coverage_gap_score = np.mean(k_nearest_dists)
        
        blended_score = context["pool"][0]["acq_value_norm"] + 0.1 * coverage_gap_score
        scores.append(blended_score)

    return scores