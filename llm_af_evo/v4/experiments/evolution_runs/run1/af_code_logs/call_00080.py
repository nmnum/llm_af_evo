def score_pool(context):
    """Estimate Pareto front uncertainty by bootstrapping Y_obs to compute hypervolume improvement of each candidate across resampled fronts."""
    import numpy as np
    
    # Parameters
    n_bootstrap = 15
    names = context["objective_names"]
    
    # Helper: vectorized dominance test (maximizing objectives)
    def is_dominated(points, point):
        dominated_mask = np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        return npany(dominates_mask)

    def get_non-dominated_front(points):  # Correctly vectorized
        n_points = points.shape[0]
        if n_points == 0:
            return []
        
        dominates = np.zeros((n_points, n_points), dtype=bool)
        for i in range(n_points):
            for j in range(n_points):
                if (i != j and 
                    np.all(points[j] >= points[i]) and
                    np.any(points[j] > points[i])):
                        dominates[i,j] = True
        
        is_ndominated = ~np.any(dominates, axis=1)
        return points[is_ndominated]

    # Compute hypervolume of a set with respect to ref_point (maximizing objectives)
    def hv_volume(front, reference):
        if len(front) == 0:
            return float(0.0)

        front = np.array(front).reshape(-1,2)
        
        try: 
            volume = np.prod(reference - np.min(front, axis=0))
            # Handle edge cases (e.g., dominated points in some direction),
            # but since we're computing HV from non-dominated set,
            # this should work directly if all objectives are maximized.
            
            return float(volume)
        except:
             return 0.0

    def hypervolume_improvement(front, candidate_point):
        """Compute the difference between adding a point to front and not."""
        
        current_hv = hv_volume(front, context["ref_point"])
        new_front_with_candidate = np.vstack([front,candidate_point])
        if len(new_front_with_candidate) <= 1:
            return float(0.0)
            
        added_hv = hv_volume(get_non-dominated_front(new_front_with_candidate), 
                            context['ref_point'])
        
        # Return the difference (improvement from adding candidate to front).
        try:  
             diff_improve = max(float(added_hv - current_hv) , 0.0)
             return float(diff_improve)
        except:
            return float(0.)

    scores = []
    
    for cand in context["pool"]:
        
        # Get the posterior mean of candidate
        gp_posterior_mean = np.array([cand['gp_posterior'][name]['mean'] 
                                     for name in names])
         
        improvements_across_resamples = []

        for _ in range(n_bootstrap):
            
            indices_sampled_with_replacement = np.random.choice(len(context["Y_obs"]), size=len(context["Y_obs"]))
            resampled_Y = context["Y_obs"][indices_sampled_with_replacement]

            # Get non-dominated front of the sample
            nd_front_resample= get_non-dominated_front(resampled_Y)
            
             if len(nd_front_resample) == 0:
                improvements_across_resamples.append(float(0.))
                
             else:  
                 improvement = hypervolume_improvement(
                     nd_front_resample, 
                      gp_posterior_mean
                  )
                 
            improvements_across_resamples.append(improvement)

        avg_imp = np.mean(np.array(improvements_across_resamples))    
        
         scores.append(avg_imp)
            
    return scores