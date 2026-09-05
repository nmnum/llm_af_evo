def score_pool(context):
    """Blend acquisition value with an entropy-based diversity signal that rewards candidates near uncovered regions of objective space."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute a rough estimate of candidate density in feature space using observed points  
    if len(context['X_obs']) < 2:
        densities = np.ones(len(context['pool']))
    else: 
        from scipy.spatial.distance import cdist
        X_pool = np.array([cand["x"] for cand in context["pool"]])
        distances = cdist(X_pool, context['X_obs'], metric='euclidean')
        nearest_distances = np.min(distances, axis=1)
        # Invert to get density (smaller distance => higher density)  
        densities = 1.0 / (nearest_distances + 1e-8)

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        acq_val = cand['acq_value_norm']
                
        # Entropy-based diversity: candidates with lower density get higher score  
        entropy_bonus = np.log(densities[i] + 1.0) 

        scores.append(acq_val * (1.0 + 0.5 * entropy_bonus))

    return scores