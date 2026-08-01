def score_pool(context):
    """Rank candidates greedily by base score, then reduce scores of nearby candidates to avoid clustering."""
    names = context["objective_names"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        scores.append(mu_sum)
    
    selected = []
    final_scores = [0.0] * len(scores)
    
    while len(selected) < len(scores):
        # Pick the best remaining candidate
        best_idx = -1
        best_score = -float('inf')
        for i, score in enumerate(scores):
            if i not in selected and score > best_score:
                best_score = score
                best_idx = i
        
        selected.append(best_idx)
        final_scores[best_idx] = best_score
        
        # Reduce scores of nearby candidates
        base_x = context["pool"][best_idx]["x"]
        for i, cand in enumerate(context["pool"]):
            if i not in selected:
                x = cand["x"]
                dist = np.linalg.norm(x - base_x)
                scores[i] *= (1.0 - 0.5 * dist)  # Reduce score by proximity
    
    return final_scores