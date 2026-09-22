def score_pool(context):
    """Blend hypervolume improvement with a coverage-gap term: reward candidates near sparse front regions."""
    names = context["objective_names"]
    pf = context["pareto_front"] 
    acq_norm = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Early campaign fallback
    if len(pf) < 3:
        ref_points = context["Y_obs"]
    else:
        ref_points = pf
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        x_cand = cand["x"]
        
        distances = [np.linalg.norm(x_cand - x_ref, ord=2) for x_ref in ref_points]
        k_nearest = sorted(distances)[:3]  # Top-3 nearest
        coverage_gap_score = np.mean(k_nearest)
        
        blended score = acq_norm[i] + 0.1 * (coverage_gap_score / max(1e-8, context["pareto_front_range"][names[0]]))
        scores.append(blended_score)

    return scores