def score_pool(context):
    """Score by predicted objective sum adjusted for uncertainty-based pareto dominance probability."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Estimate how likely this candidate is to be Pareto-optimal
        # by computing the probability that it beats all current front points  
        prob_pareto = 1.0
        
        if len(context["pareto_front"]) > 0:
            for i, obj_name in enumerate(names):
                cand_mean = gp[obj_name]["mean"]
                cand_std = gp[obj_name]["std"] 
                
                # Probability this candidate dominates the current front point
                prob_dominate_i = 1.0
                
                if len(context["pareto_front"]) > 0:
                    for pf_point in context["pareto_front"]:
                        z_score = (pf_point[i] - cand_mean) / max(cand_std, 1e-8)
                        
                        # Standard normal CDF
                        import scipy.stats as stats  
                        prob_dominate_i *= (1.0 - stats.norm.cdf(z_score))
                
                if i == 0:
                    prob_pareto = prob_dominate_i 
                else:   
                    prob_pareto *= prob_dominate_i
        
        # If candidate is likely not Pareto-optimal, reduce its score
        dominance_bonus = max(1.0 - (1.0 / (prob_pareto + 1e-8)), 0) if prob_pareto < 1 else 0

        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        scores.append(mu_sum * (1.0 + dominance_bonus) - 2.5*sigma_norm)

    return scores