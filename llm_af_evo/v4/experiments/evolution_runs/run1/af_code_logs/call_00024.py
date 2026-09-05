def score_pool(context):
    """Estimate each candidate's potential for improving the pareto front by sampling noisy objectives and computing expected HV gain."""
    
    if not context["pool"]:
        return []
        
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    scores = []

    # Use a fixed noise level to simulate uncertainty
    noise_level = 0.01

    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample noisy predictions from the GP posteriors (mean ± normal(0, std))
        samples_per_objective = []
        num_samples = 256
        
        for name in names:
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            
            # Generate noisy sample
            noise_sample = np.random.normal(mean_val, std_val + noise_level)
            samples_per_objective.append(noise_sample)

        predicted_obj_values = np.array(samples_per_objective) 

        # Compute hypervolume contribution of this candidate (if added to the current front).
        hv_contribution_estimate = 0.0
        for _ in range(num_samples):
            sample_point = [np.random.normal(mean_val, std_val + noise_level)
                            for mean_val, std_val in zip([gp_posterior[name]["mean"] 
                                                         for name in names],
                                                        [gp_posterior[name]["std"]
                                                         for name in names])]
            
            # Create a new front by adding this sample point to the existing pareto_front
            extended_pf = np.vstack((context["pareto_front"], sample_point))
            
            try:
                hv_improvement = hypervolume(extended_pf, ref_point) - hypervolume(context["pareto_front"], ref_point)
                
                if not (hv_improvement < 0): # Only consider positive contributions
                    hv_contribution_estimate += max(hv_improvement / num_samples, 1e-8)

            except:
                 pass

        scores.append(max(0.5 * cand["acq_value_norm"] + 0.5 * (hv_contribution_estimate), 
                          np.finfo(np.float32).eps))

    return [float(score) for score in scores]

def hypervolume(front, ref_point):
    """Compute the hypervolume of a front relative to reference point."""
    
    # Simple implementation assuming two objectives and non-dominated points
    if len(front.shape) == 1:
        front = np.expand_dims(front, axis=0)
        
    n_points = len(front)

    vol_total = float(0.0)

    for i in range(n_points):
         x_i,y_i = front[i][0],front[i][1]
         
         # For a reference point r=(r_x,r_y), hypervolume contribution is:
         # min(r_x - p[0] , 0) * (min(ry, y_j) for all j in PF such that x_j >=x_i)
    
    return vol_total