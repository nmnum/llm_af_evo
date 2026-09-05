def score_pool(context):
    """Exploits predicted means with uncertainty-weighted ref-point distance and dynamic exploration scaling."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize candidate means and stds by the observed range
        mu_norm = np.array([gp[name]["mean"] / front_range[name] for name in names])
        sigma_norm = np.array([gp[name]["std"] / front_range[name] for name in names])

        # Compute distance to reference point (normalized)
        ref_dist = np.linalg.norm(ref_point - mu_norm)

        # Dynamic uncertainty scaling: increase impact of std as progress increases
        u_scale = 1.0 + 2.5 * progress

        # Combine mean, scaled uncertainty and ref-point proximity into a score  
        combined_score = (
            np.sum(mu_norm) 
          + np.sum(u_scale * sigma_norm)
          - (ref_dist / len(names))
        )

        scores.append(combined_score)

    return scores