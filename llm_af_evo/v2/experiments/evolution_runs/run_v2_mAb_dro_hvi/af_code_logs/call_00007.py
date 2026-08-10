def score_pool(context):
    """Rank candidates greedily by base score, then suppress remaining scores based on proximity to already-picked candidates."""
    names = context["objective_names"]
    
    # Base scoring: sum of means (higher is better)
    base_scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        base_scores.append(mu_sum)

    picked_indices = []  # indices already selected
    final_scores = [0.0] * len(context["pool"])  # to be filled

    while len(picked_indices) < len(context["pool"]):
        best_idx = -1
        best_score = float('-inf')
        
        for i in range(len(context["pool"])):
            if i not in picked_indices:
                score = base_scores[i]
                if score > best_score:
                    best_score = score
                    best_idx = i

        # Pick the best remaining candidate and add to list of picks  
        picked_indices.append(best_idx)
        
        # Compute multiplier for this pick (1.0 at dist=inf, near 0 at dist=0) 
        cand_x = context["pool"][best_idx]["x"]
        multipliers = []
        for i in range(len(context["pool"])):
            if i not in picked_indices:
                other_cand = context["pool"][i]
                
                # Distance between candidates (L2 norm)
                dist_sq = np.sum((cand_x - other_cand["x"]) ** 2) 
                multiplier = 1.0 - np.exp(-dist_sq)

                multipliers.append(multiplier)
            else:
                multipliers.append(1.0) # no suppression for already picked

        final_scores[best_idx] = base_scores[best_idx]

    return [s * m if i not in picked_indices or (i == best_idx and len(picked_indices)==1) 
           else s
            for i, (s,m) in enumerate(zip(base_scores, multipliers))]