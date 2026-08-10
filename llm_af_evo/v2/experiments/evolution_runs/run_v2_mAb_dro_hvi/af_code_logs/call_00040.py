def score_pool(context):
    """Estimate improvement in hypervolume if candidate becomes pareto optimal, using Monte Carlo samples of its GP posteriors."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_range = context["pareto_front_range"]

    # Sample each candidate's posterior to estimate hypervolume improvement
    scores = []
    n_samples = 100

    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from the joint GP distribution of all objectives 
        means = np.array([gp[name]["mean"] for name in names])
        stds = np.array([gp[name]["std"] for name in names])

        # Simplified: assume independent normal posteriors
        sampled_objs = []
        for _ in range(n_samples):
            sample_obj = [np.random.normal(mean, stdev) if stdev > 0 else mean 
                          for (mean, stdev) in zip(means, stds)]
            sampled_objs.append(sample_obj)

        # Compute hypervolume improvement from each sample
        hv_improvements = []
        
        for obj_vals in sampled_objs:
            # Check how much better this candidate would be than current pareto front 
            if not len(context["pareto_front"]):
                # No known non-dominated points yet — treat as a big win.
                hypervolume_gain = np.prod(ref_point - np.array(obj_vals))
                
            else:  
                dominates_any_pareto = False
                for pf in context["pareto_front"]:
                    if all(pf_obj >= obj_val for (pf_obj, obj_val) in zip(pf, obj_vals)):
                        # This candidate is dominated by a point on the pareto front.
                        dominates_any_pareto = True 
                        break

                if not dominates_any_pareto:
                     # Candidate could be Pareto optimal — compute HV improvement
                    hypervolume_gain = 0.0
                    
                    for pf_point in context["pareto_front"]:
                         # Compute contribution to the dominated region by this new point.
                        
                        # If candidate is better than front's current bounds, 
                        # we can't really improve on that part of space — just count it
                        if all(obj_val >= pfo  or np.isclose(pfo,obj_val) for (pfo, obj_val) in zip(pf_point, obj_vals)):
                            continue 

                         # Otherwise compute the HV contribution this candidate makes to expanding dominated region  
                        
                         # We want a volume from reference point down toward new objective
                        lower_bounds = [max(rf_pnt, pnt)
                                        for rf_pnt,pnt in zip(ref_point,obj_vals)]

                        upper_bound = obj_vals  # The current sample

                        vol_contrib = np.prod(np.array(upper_bound) - 
                                              np.maximum(lower_bounds,
                                                         ref_point))

                        hypervolume_gain += max(vol_contrib ,0.0)

                else:
                    hv_improvements.append(-1e-6)
                    
            if not dominates_any_pareto and len(context["pareto_front"]) > 0:  
               # Normalize by the total range of current front to avoid scaling issues
                   norm_factor = np.prod([front_range[name] for name in names])
                    hv_improvements.append(hypervolume_gain / (norm_factor + 1e-8))
            else:
                hypervolume_gain=0.0  
                
        # Take expected HV improvement across samples as the score 
        if len(hv_improvements) > 0:    
             scores.append(np.mean(hv_improvements))   
        elif hv_improvements == []:
              scores.append(1e-8)
              
    return [max(score, -5.0) for score in scores]