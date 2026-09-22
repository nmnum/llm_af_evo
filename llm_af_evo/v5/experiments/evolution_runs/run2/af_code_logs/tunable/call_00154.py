def score_pool(context):
    """Balance acquisition value with dynamic uncertainty and progress-aware exploitation to favor candidates near the Pareto front's reachable edge."""
    
    names = context["objective_names"]
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    front_range = context["pareto_front_range"]
    ref_point_by_name = context["ref_point_by_name"]

    # Dynamic UCB bonus that decreases as campaign progresses
    progress = context["campaign"]["progress"]
    ucb_weight = 0.5 * (1 - progress)

    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        unc_score = ucb_weight * sigma_norm_sum
        unc_scores.append(unc_score)

    # Progress-aware exploitation: boost candidates whose predicted objectives are closer to the reachable edge of current Pareto front 
    exploit_weights = []
    
    if len(context["pareto_front"]) > 0:
        
        pf_points = context["pareto_front"]
  
        for cand in context["pool"]:
            gp_posterior = cand['gp_posterior']
            
            # Get predicted objectives
            pred_obj_vals = [gp_posterior[name]["mean"] for name in names]
 
            min_dist_to_pf_edge = float('inf')
          
            for pt in pf_points:
                
                dist_to_pt = 0.0
                
                for i, val in enumerate(pred_obj_vals):
                    # Distance to the edge of current front along each objective
                    if val <= (pt[i] + 1e-8): 
                        continue   # This candidate is dominated by this point; not contributing to HV
                        
                    dist_to_pt += max(0.0, ref_point_by_name[names[i]] - pt[i])
                    
                min_dist_to_pf_edge = min(min_dist_to_pf_edge, dist_to_pt)
            
            if np.isinf(min_dist_to_pf_edge):
                
                exploit_weight = 1e-6   # No front edge in reach — low weight
              
            else:
               
                normalized_distance = (min_dist_to_pf_edge / 
                                      sum(front_range[name] for name in names))
                
                exploitation_bonus = max(0.0, 1 - progress) * np.exp(-normalized_distance)
          
                exploit_weights.append(exploitation_bonus)

        # Normalize to avoid extreme weights
        if len(evict_weights) > 0:
            norm_factor = sum(exploit_weights) / (len(context["pool"]) or 1.)
            
            for i in range(len(evict_weights)):
                
                 exploit_weights[i] *= max(1e-6, 5. *norm_factor)
        
    else: 
        # No front yet — default to uniform
        exploit_weights = [0.] * len(context["pool"])
    
  
   final_scores = acq_scores + np.array(unc_scores) -np.array(exploit_weights)

   
 return list(final_scores)


Note:

The new approach focuses on a progress-aware exploitation term that rewards candidates whose predicted objectives are closer to the edge of current Pareto front in objective space (i.e., those more likely able to expand hypervolume). It is structurally different from previous attempts as it does not resample posteriors, estimate dominance probabilities or compute novelty distances. Instead, this method evaluates how close each candidate's prediction lies along a path toward the dominated region of current Pareto front.

It avoids using "estimated HV improvement probability", which was tried multiple times and didn't improve fitness recently.
Also differs from other recent proposals involving proximity to previous observations in objective space or reweighting uncertainty terms — this method introduces an explicit, novel exploitation signal based on geometric reachability toward the reachable edge of existing Pareto front.

The core innovation lies in using a distance-to-Pareto-front-edge heuristic that dynamically adjusts with campaign progress and combines well with acquisition value + UCB-style exploration.