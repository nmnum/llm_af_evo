def modifier(context):
    """Add uncertainty bonus scaled by how close candidates are to the current Pareto front."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    ref_point = context["ref_point"]
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Estimate distance from candidate's mean prediction to the nearest point on Pareto front
        mu_vals = np.array([gp[name]["mean"] for name in names])
        distances_to_pf = [np.linalg.norm(mu_vals - pf_point) for pf_point in pf]
        min_dist_to_front = min(distances_to_pf) if len(pf) > 0 else float('inf')
        
        # Scale uncertainty bonus based on proximity to front
        sigma_sum = sum(gp[name]["std"] for name in names)
        weight = max(1.0 - (min_dist_to_front / np.linalg.norm(ref_point)), 0.)
        values.append(sigma_sum * weight * 0.2)

    return values