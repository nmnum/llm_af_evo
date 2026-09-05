def modifier(context):
    """Add uncertainty bonus scaled by how close candidates are to the Pareto front edge."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    pf = context["pareto_front"]
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Compute distance from candidate's mean prediction to the Pareto front
        pred_means = np.array([gp[name]["mean"] for name in names])
        dist_to_pf_sq = float('inf')
        if len(pf) > 0:
            pf_dists_sq = np.sum((pred_means - pf)**2, axis=1)
            dist_to_pf_sq = np.min(pf_dists_sq)

        # If candidate is far from the front (in objective space), add bonus
        distance_from_front = np.sqrt(dist_to_pf_sq) if dist_to_pf_sq != float('inf') else 0.0
        
        # Bonus increases with uncertainty and decreases as we approach Pareto frontier
        sigma_sum = sum(gp[name]["std"] for name in names)
        
        front_distance_penalty = (1 - distance_from_front / np.linalg.norm(ref_point)) if not np.allclose(ref_point, 0) else 0.0
        
        values.append(sigma_sum * front_distance_penalty * 0.2)

    return values