def score_pool(context):
    """Combine pure exploitation with hypervolume improvement estimation, favoring candidates that are both highly predicted and help expand the dominated region."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    pareto_front = context["pareto_front"]
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        cand_obj = np.array([gp[name]["mean"] for name in names])
        
        # Exploitation: sum of predicted means
        exploitation_score = sum(cand_obj)
        
        # Hypervolume improvement estimation
        if len(pareto_front) == 0:
            hv_improvement = np.prod(ref_point - cand_obj)
        else:
            # Check if candidate is dominated by current Pareto front
            is_dominated = False
            for front_point in pareto_front:
                if all(front_point[i] >= cand_obj[i] for i in range(len(names))):
                    is_dominated = True
                    break

            if is_dominated:
                hv_improvement = -1e10
            else:
                hv_improvement = np.prod(ref_point - cand_obj)
        
        # Combine exploitation and HV improvement scores
        scores.append(exploitation_score + hv_improvement)
    
    return scores