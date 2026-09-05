def score_pool(context):
    """Exploits progress-aware mean prediction while dynamically scaling uncertainty to balance exploration and exploitation based on front diversity."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Progress-sensitive weight for balancing exploitation vs. uncertainty
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.4)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Scale uncertainty dynamically based on progress and how much the current Pareto
        # frontier spans each objective (i.e., less range implies more pressure to explore).
        sigma_scaled = 0.0
        total_front_span = sum(front_range.values())
        
        if total_front_span > 1e-8: 
            for name in names:
                normalized_std = gp[name]["std"] / front_range[name]
                
                # If objective range is small, we're more sensitive to uncertainty — high variance means "more exploration needed"
                span_weight = (front_range[name] / total_front_span)
                
                sigma_scaled += 0.5 * np.exp(-3*progress) + normalized_std
        else:
            # Fallback if ranges are all zero or nearly so, treat as uniform scaling.
            sigma_scaled = sum(gp[name]["std"] for name in names)

        score = w_exploit * mu_sum_norm - (1.0-w_exploit)*sigma_scaled
        
        scores.append(score)
    
    return scores