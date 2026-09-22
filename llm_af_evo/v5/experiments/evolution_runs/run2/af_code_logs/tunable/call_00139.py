def score_pool(context):
    """Blend acquisition value with an entropy-based uncertainty bonus that encourages exploring objective-space regions where posterior distributions are most informative."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]        
        # Use the precomputed acquisition value as baseline
        score = cand["acq_value_norm"]

        # Add an entropy-based uncertainty bonus: higher when multiple objectives have high variance 
        entropies = [0.5 * np.log(2*np.pi*np.e*gp[name]["std"]**2) for name in names]  # Differential entropy of Gaussian
        total_entropy = sum(entropies)
        
        # Normalize by number of objectives to avoid bias from dimensionality  
        normalized_entropy = total_entropy / len(names)

        # Scale the bonus based on campaign progress: more exploration early, less later 
        progress = context["campaign"]["progress"]
        weight = 1.0 - progress
        
        score += weight * normalized_entropy
    
    return scores