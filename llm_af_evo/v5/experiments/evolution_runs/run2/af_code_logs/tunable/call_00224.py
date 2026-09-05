def score_pool(context):
    """Blend acquisition value with progress-adaptive uncertainty and inverse novelty penalty, enhanced by coverage-gap targeting sparse Pareto front regions."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distances from all previously observed points
    X_obs = context["X_obs"]
    if len(X_obs) > 0:
        novelty_distances = []
        for i, cand in enumerate(context["pool"]):
            x_cand = np.array(cand['x'])
            dists_to_observed = [np.linalg.norm(x_cand - x_obs) for x_obs in X_obs]
            min_dist = min(dists_to_observed)
            novelty_distances.append(min_dist)

        # Normalize distances to [0, 1] scale
        if max(novelty_distances) > 0:
            novel_scores = np.array(novelty_distances) / max(novelty_distances)
        else:
            novel_scores = np.zeros_like(novelty_distances)
    else: 
        novel_scores = np.zeros(len(context["pool"]))

    # Progress-aware scaling of uncertainty bonus
    progress = context['campaign']['progress']
    
    # Early in campaign, boost exploration; later focus on exploitation  
    exploit_weight = 1.0 - min(1.0, max(0., (2 * progress) - 1)) 
    explore_weight = 1.0 - exploit_weight
    
    scores = []
        
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand['gp_posterior']
                
        # Uncertainty bonus scaled by campaign phase
        sigma_sum_normed = sum(gp_posterior[name]["std"]/context["pareto_front_range"][name] 
                               for name in names)
        
        uncertainty_bonus = explore_weight * (sigma_sum_normed / len(names))
            
        # Inverse novelty penalty — penalize candidates too close to previous observations  
        novel_penalty = 0.25 * novel_scores[i]
                
        # Coverage-gap term: encourage points near sparse regions of the front
        pf = context["pareto_front"]
        use_y_obs = len(pf) < 3
        
        distances = []
        reference_points = pf if not use_y_obs else context["Y_obs"] 
        cand_point = np.array([gp_posterior[name]["mean"] for name in names])

        for ref_point in reference_points:
            dist = np.linalg.norm(cand_point - ref_point, ord=2)
            distances.append(dist)

        sorted_distances = sorted(distances)[:min(3, len(distances))]
        mean_dist_to_front = sum(sorted_distances)/len(sorted_distances) if sorted_distances else 0.0

        coverage_gap_score = (1.0 - (mean_dist_to_front / context["pareto_front_range"][names[0]])) if not use_y_obs else mean_dist_to_front
        
        # Final blend
        combined_score = acq_scores[i] + uncertainty_bonus - novel_penalty + 0.2 * coverage_gap_score

        scores.append(combined_score)
        
    return scores