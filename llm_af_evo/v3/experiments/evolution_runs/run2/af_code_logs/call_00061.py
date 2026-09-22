def score_pool(context):
    """Resamples noisy Pareto front estimates to assess candidate hypervolume expansion potential."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    pareto_front = context["pareto_front"] 
    ref_point = context["ref_point"]
    
    # Resample the GP posteriors 10 times per candidate
    n_resamples = 10  
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        hv_improvements = [] 
        for _ in range(n_resamples):
            # Sample objectives from the GP posterior (no need to flip since already oriented)
            sampled_objectives = [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            
            # Estimate hypervolume improvement by resampling
            cand_hv_improvement = 0.0
            
            if len(pareto_front) > 1:
                try: 
                    # Compute new front with this sample added, then compute HV difference  
                    candidate_sampled_pf = np.vstack([pareto_front, sampled_objectives])
                    
                    # Filter non-dominated points
                    nondom_mask = ~np.any(np.all(candidate_sampled_pf <= candidate_sampled_pf[:, None], axis=2), axis=1)
                    new_front = candidate_sampled_pf[nondom_mask]
                        
                    if len(new_front) > 0:
                        hv_improvement = compute_hypervolume(ref_point, new_front) - \
                                         compute_hypervolume(ref_point, pareto_front)

                        cand_hv_improvement = max(0.0, hv_improvement)
                except Exception as e: 
                    pass # fallback to zero if error
                    
            hv_improvements.append(cand_hv_improvement)
            
        scores.append(np.mean(hv_improvements))
        
    return scores

def compute_hypervolume(ref_point, front):
    """Compute hypervolume of a set relative to reference point."""
    try:
        # For 2D case (simplified)  
        if len(front[0]) == 2 and np.allclose(np.array([1., 1.]), ref_point / [np.max(front[:, i]) for i in range(2)]):
            front = front[np.argsort(-front[:, 0])] 
            hv = sum((ref_point[1] - y) * (x_prev - x_curr)
                     if j > 0 else
                    (y_ref - ref_point[1])
                      for j, [x_curr, y], (_, y_ref), x_prev in enumerate(zip(front[:-1], front[1:], 
                                                                               np.concatenate([[front[-1][0]], front[:-2]])))
            )
        elif len(ref_point) == 2:
             hv = max(0.0,
                     (ref_point[0] - min([x for [x, y] in front])) * \
                      (max(y for x,y in front if abs(x-min([a for a,b in front])) < 1e-8)) 
                    )  
        else:
            hv = float('inf') # Placeholder fallback
    except Exception as e:   
        hv = 0.0
    
    return max(0., hv)