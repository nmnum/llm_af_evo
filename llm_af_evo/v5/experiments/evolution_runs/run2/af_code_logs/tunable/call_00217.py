def score_pool(context):
    """Rank pool greedily by selecting candidates one at a time; each remaining candidate's base score is multiplied by a 'multiplier' factor inversely proportional to its proximity (in feature space) to already-selected candidates."""
    
    names = context["objective_names"]
    scores = []
    selected_indices = set()
    batch_size = len(context["pool"])
    
    # Compute initial acquisition values
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        mu_sum = sum(gp_posterior[name]["mean"] for name in names)
        sigma_norm = sum(gp_posterior[name]["std"] / (context["pareto_front_range"][name]) for name in names) if all(name in context["pareto_front_range"] for name in names) else 0.0
        scores.append(mu_sum + 2.0 * sigma_norm)

    final_scores = [0.] * len(context["pool"])
    
    # Greedily select candidates and apply multiplier penalties based on feature space proximity  
    for i in range(batch_size):
        
        best_idx = -1 
        max_score = float('-inf')
        
        for j, score in enumerate(scores):
            if j not in selected_indices:
                if score > max_score:    
                    max_score = score
                    best_idx = j
                    
        # Add the chosen candidate to picked set  
        assert(best_idx != -1)
        selected_indices.add(best_idx) 
        
        base_score = scores[best_idx]
        
        multiplier = 1.0
        
        for k in range(len(context["pool"])):
            if k not in selected_indices and k!= best_idx:
                # Compute distance between features of two candidates  
                
                dist_squared = np.sum((context['pool'][k]['x'] - context['pool'][best_idx]["x"]) ** 2)
                
                penalty_factor = 1.0 - np.exp(-dist_squared) 
              
                multiplier *= (1.0 - penalty_factor)

        final_scores[best_idx] = base_score * max(multiplier, 0.)
        
    return final_scores