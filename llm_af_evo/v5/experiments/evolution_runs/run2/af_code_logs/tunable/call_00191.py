def score_pool(context):
    """Score candidates by blending acquisition value with a diversity-weighted uncertainty term that emphasizes underexplored objective regions."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute distance from each candidate to the nearest point in Y_obs (full history)
    novelty_scores = []
    for cand in context["pool"]:
        x_cand = cand["x"] 
        distances = np.sqrt(np.sum((context['Y_obs'] - cand["gp_posterior"][names[0]]["mean"])**2, axis=1))
        nearest_distance = np.min(distances)
        novelty_scores.append(nearest_distance)

    # Normalize acquisition values and novelties
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    max_acq = max(acq_values) + 1e-8  
    norm_acqs = np.array(acq_values) / max_acq

    novelty_scores = np.array(novelty_scores)
    max_nov = max(novelty_scores) + 1e-8
    norm Novelties = novel scores / max nov
    
    # Dynamic weight based on campaign progress (more exploitation later, more exploration earlier)
    t = context["campaign"]["progress"]
    
    w_exploit = np.clip(0.3 * (t - 0.5) + 1., 0.4, 1.) 
    w_explor = max(0., 1.-w_exploit)

    # Final score combines acquisition with a weighted uncertainty adjusted by novelty
    scores = []
    
    for i in range(len(context["pool"])):
        gp_posterior = context["pool"][i]["gp_posterior"]
        
        sigma_sum_normed = sum(gp_posterior[name]["std"]/front_range[name] 
                               for name in names)
        
        # Apply dynamic weighting to exploitation vs exploration
        score = w_exploit * norm_acqs[i]
        if novelty_scores[i] > 0.1:  
            uncertainty_boost_factor= min(2., (sigma_sum_normed / max(novelty_scores))**-0.5) 
            reward = sigma_sum_normed * uncertainty_boost_factor
            score += w_explor*reward

        scores.append(score)
        
    return scores