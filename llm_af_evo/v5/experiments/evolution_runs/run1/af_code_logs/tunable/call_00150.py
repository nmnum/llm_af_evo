def score_pool(context):
    """Score candidates by acquisition value enhanced with a dynamic exploitation-uncertainty tradeoff that shifts based on campaign progress and penalizes overconfidence in dominated predictions."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base scores using normalized acq_value_norm
    raw_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Normalize the acquisition values to be between 0 and 1, if not already done,
    # though they are supposed to be min-max normalised by definition.
    max_acq = max(raw_scores)
    min_acq = min(raw_scores)

    normalized_acqs = [(s - min_acq) / (max_acq - min_acq + 1e-9) for s in raw_scores]
    
    # Progress-aware exploitation factor: decrease uncertainty weight as campaign progresses
    progress_factor = context["campaign"]["progress"]
  
    scores = []
    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]

        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Uncertainty term: normalized standard deviation across objectives 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        # Dynamic balance between exploitation and uncertainty
        exploit_weight = 1.0 - progress_factor * 0.5   # Early more uncertain, later more exploitative
        
        ucb_score = mu_sum + (exploit_weight) * sigma_norm

        # Adjust score based on how much the candidate's prediction is dominated by current front.
        dominates_front = False
        pred_vector = [gp[name]["mean"] for name in names]
        
        if len(context["pareto_front"]) > 0:
            for pf_point in context["pareto_front"]:
                dom_check = all(pf_point[i] >= pred_vector[i] for i in range(len(names))) and \
                            any(pf_point[i] > pred_vector[i] for i in range(len(names)))
                
                if not dom_check: continue
                dominates_front = True
                
        # Penalize overconfident predictions that are dominated by the front.
        penalty_factor = 1.0 + (dominates_front * -0.2) 
          
        final_score = normalized_acqs[i] * ucb_score * penalty_factor

        scores.append(final_score)

    return scores