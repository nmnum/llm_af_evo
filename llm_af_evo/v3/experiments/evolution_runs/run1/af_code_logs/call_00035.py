def score_pool(context):
    """Estimates improvement potential by resampling candidates' GP posteriors to compute hypervolume expansion expectation."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the joint posterior of both objectives
        n_samples = 100
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Compute expected hypervolume improvement for each sample point
        hv_improvements = []
        for f1_s, f2_s in zip(samples_f1, samples_f2):
            cand_point = np.array([f1_s, f2_s])
            
            # Determine if this candidate would improve the current front's HV 
            hypervolume_gain = 0.0
            try:
                new_front = np.vstack((context["pareto_front"], cand_point))
                
                from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
                
                nds = NonDominatedSorting()
                fronts = nds.do(new_front, n_stop_if_ranked=len(new_front))

                # Simple estimate of HV change using reference point
                if len(fronts) > 1:
                    front_pareto = new_front[fronts[0]]
                    
                    hv_new = hypervolume_calculator(
                        points=front_pareto,
                        ref_point=ref_point)
                        
                   hv_old = hypervolume_calculator(
                        points=context["pareto_front"],
                        ref_point=ref_point)

                else:
                    # No change in front
                    hv_improvements.append(0.0) 
                    continue
                    
            except Exception as e:
                 hv_improvements.append(-1e6)
                 
        if len(hv_improvements):
             scores.append(np.mean(hv_improvements))
        
    return [s for s in scores]  # Fill missing with zero

# Note: A helper function `hypervolume_calculator` must be defined or imported to compute HV.
def hypervolume_calculator(points, ref_point):  
     from pymoo.util.hypervolume import Hypervolume
     hv = Hypervolume(ref=ref_point)
     return hv.do(points)