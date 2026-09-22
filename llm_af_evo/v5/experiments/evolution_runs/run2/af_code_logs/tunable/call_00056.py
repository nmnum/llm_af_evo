def score_pool(context):
    """Estimate the probability that a candidate expands the pareto front and blend with acquisition value for robust exploration-exploitation."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Estimate probability of being Pareto-optimal using uncertainty
    front_range = context["pareto_front_range"]
    n_candidates = len(context["pool"])
    
    pareto_probs = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]

        # Compute how much the candidate's objectives are within current Pareto range
        is_dominated_by_current = False 
        dominated_count = 0

        if len(context["pareto_front"]) > 0:
            for j, pf_point in enumerate(context["pareto_front"]):
                dominates_this_cand = True  
                
                # Check dominance: all objectives of PF point >= candidate's
                for k, name in enumerate(names): 
                    cand_obj_val = gp_posterior[name]["mean"]
                    
                    if not (pf_point[k] <= cand_obj_val + 1e-8):
                        dominates_this_cand = False  
                        break
                        
                # If pf is better or equal on all objectives
                if dominates_this_cand:
                    dominated_count += 1
                    
        pareto_prob = max(0.0, 1 - (dominated_count / len(context["pareto_front"]) if context["pareto_front"].size > 0 else 1))
        
        # Bonus for uncertainty: higher std means more potential to be Pareto
        sigma_sum_normed = sum(gp_posterior[name]["std"] / front_range[name] 
                               for name in names)
            
        pareto_prob += max(0.0, (sigma_sum_normed - np.mean([gp["mean"] + gp["std"]
                                                              for cand2 in context["pool"]  
                                                              if i != j
                                                              for _, gp in cand2["gp_posterior"].items()]))) 
        
        # Normalize and cap the probability 
        pareto_prob = min(1.0, max(pareto_prob / 3., 0))
        
        pareto_probs.append(max(0.5 * acq_scores[i], par_score))

    return [s for s in np.array(acq_scores) + (np.array(pareto_probs))]

# The function above was incorrect and needed correction to avoid infinite recursion
def score_pool(context):
    """Estimate the probability that a candidate expands hypervolume by being near Pareto frontier."""
    
    names = context["objective_names"]
    n_candidates, acq_scores = len(context["pool"]), []
  
    # Get acquisition values 
    for cand in context["pool"]:
        acq_scores.append(cand["acq_value_norm"])
        
    front_range = np.array([context["pareto_front_range"][name] for name in names])
    
    scores = [] 
    
    if not len(context["X_obs"]):
       return [s + 0.5 * s**2 for s in acq_scores]
   
    # Compute pairwise distances to all observations  
    X_obs, cand_x_vals = context['X_obs'], []
        
    for i,cand in enumerate(context["pool"]): 
        x_cand = np.array(cand["x"])
        dists_sq_to_all_obvs = ((np.expand_dims(x_cand,axis=0) - X_obs)**2).sum(axis=-1)
        min_dist_squared =  np.min(dists_sq_to_all_obvs, initial=np.inf)

        # Estimate distance to nearest observation 
        if len(X_obs):
            cand_x_vals.append(np.sqrt(min_dist_squared))
        
    novelty_scores = [np.exp(-d/3) for d in cand_x_vals] 
    
    # Compute probability of improving front (using uncertainty)
    ucb_bonus, pareto_prob_score  = [], []
    
    stds_by_cand = [[c["gp_posterior"][name]["std"] 
                     for name in names]
                    for c in context['pool']]
        
    mean_objs_per_cand = np.array([[c["gp_posterior"][name]["mean"]
                                    for name in names]  
                                   for c in context['pool']])
    
    # Check if candidate is dominated by current front
    n_pf, pareto_prob_score  = len(context['pareto_front']), []
        
    for i,cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
            
        dominates_any_in_PF = False  
                
        stds_cand = [gp_posterior[name]["std"] 
                     for name in names]
    
        # Simple estimate: if the candidate is uncertain and not dominated, it might be good
        uncertainty_score = sum(std / fr  for (