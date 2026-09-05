def modifier(context):
    """Diversity bonus: penalizes candidates similar to high-acquisition ones in feature space."""
    if len(context["pool"]) < 2:
        return [0.0] * len(context["pool"])
    
    names = context['objective_names']
    X_obs = context["X_obs"]
    values = []
    acq_threshold = np.percentile([cand["acq_value_norm"] for cand in context["pool"]], 75)
    high_acq_indices = [i for i, cand in enumerate(context["pool"]) if cand["acq_value_norm"] >= acq_threshold]
    
    # Compute similarity kernel between candidates
    similarities = np.zeros((len(context["pool"]), len(context["pool"])))
    X_pool_features = np.array([cand['x'] for cand in context["pool"]])
    
    distances_sq = cdist(X_pool_features, X_pool_features, 'sqeuclidean')
    sigma2 = 0.5 * (np.median(distances_sq) + 1e-8)
    similarities = np.exp(-distances_sq / (2*sigma2))
    # Zero out diagonal
    np.fill_diagonal(similarities, 0.)
    
    for i in range(len(context["pool"])):
        sim_sum = sum(similarities[i][j] * context["pool"][j]["acq_value_norm"]
                      for j in high_acq_indices if i != j)
        
        # Normalize by number of similar candidates
        num_similar_high = len([j for j in high_acq_indices if i!=j and similarities[i,j]>0.1])
        norm_factor = max(1., num_similar_high / 3.)
            
        values.append(-sim_sum/norm_factor)
    
    # Scale to [0, 0.3]
    scale_factor = np.max(values) - np.min(values) if (np.max(values)-np.min(values)) > 1e-8 else 1
    return [(v / scale_factor)*0.3 for v in values]