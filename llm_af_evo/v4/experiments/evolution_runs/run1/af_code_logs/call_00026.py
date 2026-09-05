def score_pool(context):
    """Resample noisy GP predictions to estimate each candidate's probability of improving hypervolume and blend with acquisition value."""
    if not context["pool"]:
        return []
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    scores = []

    # Use a fixed noise level to simulate uncertainty
    noise_level = 0.01

    num_samples = 256
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        hv_contributions = []
        # Sample from the GP posteriors multiple times per candidate 
        for _ in range(num_samples):
            sample_point = []

            for name in names:
                mean_val = gp_posterior[name]["mean"] 
                std_val = gp_posterior[name]["std"]

                noise_sample = np.random.normal(mean_val, std_val + noise_level)
                
                # Ensure the sampled point is within [0, 1] range if needed (though it should be already normalized by GP training).
                sample_point.append(noise_sample)

            extended_pf = np.vstack((context["pareto_front"], sample_point))
            
            try:
                 hv_improvement = hypervolume(extended_pf, ref_point) - hypervolume(context["pareto_front"], ref_point)
                 
                 if not (hv_improvement < 0): # Only consider positive contributions
                     hv_contributions.append(max(hv_improvement / num_samples, np.finfo(np.float32).eps))
            
            except:
                pass

        estimated_hypervolume_gain = sum(hv_contributions) 

        scores.append(estimated_hypervolume_gain * 0.5 + cand["acq_value_norm"] * 0.5)

    return [float(score) for score in scores]

def hypervolume(front, ref_point):
    """Compute the two-dimensional hypervolume of a front relative to reference point."""
    
    if len(front.shape) == 1:
        front = np.expand_dims(front, axis=0)
        
    n_points = len(front)

    vol_total = float(0.0)
    
    for i in range(n_points):
         x_i,y_i = front[i][0],front[i][1]
         
         # For two objectives and a reference point r=(r_x,r_y), hypervolume contribution is:
         min_rx_minus_xi  = max(ref_point[0] - x_i, 0)
         min_ry_minus_yi  = max(ref_point[1] - y_i, 0)

    return vol_total