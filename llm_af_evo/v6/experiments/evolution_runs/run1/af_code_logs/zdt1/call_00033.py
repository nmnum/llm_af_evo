def modifier(context):
    """Combine uncertainty bonus with novelty reward, scaled by stagnation level for adaptive exploration."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Compute uncertainty bonus (UCB-style)
    weight_u = min(1.0, 0.5 + 0.2 * stagnant_batches) 
    sigma_norms = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]  
        sigmas = [gp[name]["std"] / front_range[name] for name in names]
        sigma_norms.append(sum(sigmas))
    
    # Compute novelty bonus
    y_min = np.array([context["ref_point_by_name"][name] for name in names])
    y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
    ranges = y_max - y_min
    
    normalized_y_obs = (context["Y_obs"] - y_min) / ranges
    novelty_scores = []
    
    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - y_min) / ranges  
        distances = np.linalg.norm(normalized_y_obs - norm_mean, axis=1)
        min_distance = np.min(distances)
        novelty_scores.append(min_distance)

    # Normalize and combine
    max_novelty = max(novelty_scores) if any(v > 0 for v in novelty_scores) else 1.0
    
    values = []
    for i, cand in enumerate(context["pool"]):
        ucb_bonus = weight_u * sigma_norms[i]
        nov_reward = (novelty_scores[i] / max_novelty) * 0.3
        total_correction = ucb_bonus + nov_reward 
        values.append(total_correction)
    
    return values