def score_pool(context):
    """Blend acquisition value with uncertainty and novelty to improve exploration-exploitation balance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base scores using botorch's qLogNEHVI values (already hypervolume estimates)
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Add uncertainty bonus: sum of normalized stds
    sigma_bonus = []
    for cand in context["pool"]:
        norm_std_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                           for name in names)
        sigma_bonus.append(norm_std_sum)

    # Compute novelty scores (inverse distance to nearest observed point)  
    X_obs = context["X_obs"]
    novelties = []
    if len(X_obs) > 0:
        cand_xs = np.array([cand['x'] for cand in context["pool"]])
        distances = cdist(cand_xs, X_obs)
        min_distances = np.min(distances, axis=1)
        # Invert to get novelty: closer points have lower scores
        novelties = 1.0 / (min_distances + 1e-8) 
    else:
        novelties = [1.0] * len(context["pool"])

    # Combine acquisition value with uncertainty bonus and novelty using dynamic weights  
    progress = context['campaign']['progress']
    
    w_acq = max(0.7, 1 - progress)
    w_sigma = min(0.3, progress) 
    w_novelty = 0.2 * (1 - progress)

    scores = []
    for i in range(len(context["pool"])):
        score = (
            acq_values[i] * w_acq +
            sigma_bonus[i] * w_sigma +  
            novelties[i] * w_novelty
        )
        scores.append(score)
    
    return scores