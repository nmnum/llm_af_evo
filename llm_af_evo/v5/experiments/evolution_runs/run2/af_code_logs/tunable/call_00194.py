def score_pool(context):
    """Blend acquisition value with an entropy-based diversity incentive that rewards candidates near uncovered regions of objective space."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute candidate-wise entropic novelty scores based on distance to Pareto frontier in normalized obj-space
    pf_normalized = (context["pareto_front"] - [front_range[name] for name in names]) / \
                    np.array([front_range[name] for name in names])
    novelties = []
    
    # For each candidate, compute the inverse of squared distance to nearest front point,
    # normalized by objective ranges
    for cand in context["pool"]:
        x_norm = (cand['x'] - [0.]*len(names)) / np.array([1.] * len(names))
        
        distances_to_front = []
        if len(context["pareto_front"]) > 0:
            dists_sq = ((pf_normalized[:, None] - x_norm[None, :])**2).sum(axis=2)
            min_dist_squared = np.min(dists_sq) + 1e-8
        else:
            # If no front yet (early stage), use a fixed reference distance to encourage exploration.
            min_dist_squared = 0.5**2
            
        novelty_score = 1 / min_dist_squared 
        novelties.append(novelty_score)

    scores = []
    
    for i, cand in enumerate(context["pool"]):
        # Use the provided acquisition value as primary signal
        acq_value_norm = cand['acq_value_norm']
        
        # Combine with novelty (entropy-based incentive)
        score = 0.8 * acq_value_norm + 0.2 * novelties[i]
    
        scores.append(score)

    return scores