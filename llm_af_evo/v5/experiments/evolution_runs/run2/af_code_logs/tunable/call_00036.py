def score_pool(context):
    """Blend acquisition value with uncertainty-aware Pareto probability for adaptive exploration-exploitation."""
    names = context["objective_names"]
    
    # Base scores from normalized hypervolume improvement estimates  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Estimate how likely each candidate is to be Pareto-optimal
    pareto_probs = []
    front_range = context["pareto_front_range"]
        
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]

        n_better_than_pf = 0
        
        # Sample from the GP posterior to estimate how many PF points it beats
        total_evals = 100

        for _ in range(total_evals):  
            sample_means = []
            
            for name in names:
                mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"]
                
                # Sample from the GP posterior (assuming normality)
                sampled_val = np.random.normal(mu, sigma) 
                    
            if len(context["pareto_front"]) > 0:  
                pf_vals = context["pareto_front"][:, names.index(name)]
            
                n_better_than_point_in_pf = sum(1 for val in pf_vals if sampled_val >= val)
                
                # Only count as better than a PF point (not just any objective value) 
                if len(context['objective_names']) == 2:
                    obj_0, obj_1 = context["pareto_front"][:, names.index('obj_0')], \
                                   context["pareto_front"][:, names.index('obj_1')]
                    
                    # Check for dominance in both objectives  
                    dominates_pf_point = all(sampled_val >= pf_vals[i] 
                                             if i < len(pf_vals) else False
                                            for i, _ in enumerate(obj_0))
                elif not isinstance(context["pareto_front"], np.ndarray):
                    n_better_than_point_in_pf = 1.0 # fallback to high prob when no PF available

            pareto_probs.append(n_better_than_point_in_pf / len(context["pareto_front"]) 
                                if len(context["pareto_front"]) > 0 else 1.0)
    
    final_scores = []
        
    for i in range(len(acq_scores)):
        # Use uncertainty-aware Pareto probability
        prob_pareto = pareto_probs[i]
            
        combined_score = (acq_scores[i] 
                          + 0.3 * np.clip(prob_pareto, 0., 1.)  
                         )
                
        final_scores.append(combined_score)
        
    return final_scores