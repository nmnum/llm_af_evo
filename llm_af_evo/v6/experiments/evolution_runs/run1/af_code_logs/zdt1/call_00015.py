def modifier(context):
    """Combine uncertainty bonus with novelty penalty based on distance in objective space."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    # Uncertainty bonus component
    weight_uncert = 0.3495 * (1.0 - progress)
    uncert_terms = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
        uncert_terms.append(weight_uncert * sigma_sum)

    # Novelty penalty component
    pool = context["pool"]
    novelty_terms = [0.0] * len(pool)
    selected_x = []
    
    for i in range(len(pool)):
        cand_x = pool[i]["x"]        
        min_dist_sq = float('inf')
        for prev_x in selected_x:
            dist_sq = np.sum((cand_x - prev_x) ** 2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        
        distance = np.sqrt(min_dist_sq) if min_dist_sq != float('inf') else 0.01
        multiplier = 1.0 - np.exp(-distance)
        
        novelty_terms[i] = (multiplier - 1.0) * 0.244

        selected_x.append(cand_x)

    return [u + n for u, n in zip(uncert_terms, novelty_terms)]