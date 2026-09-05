def score_pool(context):
    """Blend acquisition value with uncertainty-aware Pareto front coverage to balance exploration and exploitation."""
    names = context["objective_names"]
    
    # Use normalized acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Estimate how much each candidate extends the dominated hypervolume
    front_range = context["pareto_front_range"]

    coverage_gap_terms = []
        
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
            
        # Compute distance from current Pareto frontier (normalized by range)
        if len(context["pareto_front"]) > 0:
            pf_vals = context["pareto_front"]

            min_distances_to_pf = []
                
            for name in names: 
                mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"]
                    
                # Sample from the GP posterior (assuming normality)
                sampled_val = np.random.normal(mu, sigma) 

                pf_vals_for_obj = pf_vals[:, names.index(name)]
                
                distances_to_pf_points = [sampled_val - val for val in pf_vals_for_obj]
                        
                min_distance = np.min(distances_to_pf_points)

                # Normalize by the range of observed values
                normalized_min_dist = (min_distance / front_range[name]) if front_range[name] > 0 else 1.0

                min_distances_to_pf.append(normalized_min_dist)
                
            coverage_gap_term = -np.mean(min_distances_to_pf) 
        else:
            # No frontier yet, assume it's likely to be good
            coverage_gap_term = 1.0
            
        coverage_gap_terms.append(coverage_gap_term)

    final_scores = []
        
    for i in range(len(acq_scores)):
            
        combined_score = (acq_scores[i] 
                          + 0.5 * np.clip(coverage_gap_terms[i], -2., 2.) # weight the gap term
                         )
                
        final_scores.append(combined_score)
        
    return final_scores