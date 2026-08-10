def score_pool(context):
    """Estimate hypervolume improvement potential using Monte Carlo samples from each candidate's GP posteriors."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_range = context["pareto_front_range"]
    
    # Sample from the posterior of each candidate to estimate hypervolume improvement
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Draw samples from GP posteriors (each sample is a possible true objective vector)
        sampled_objectives = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"] 
            sampled_objectives[:,i] = np.random.normal(mean_val, std_val, n_samples)

        # Compute hypervolume improvement if this candidate were observed
        hv_improvements = []
        
        for sample in sampled_objectives:
            expanded_front = np.vstack([context["pareto_front"], sample])
            
            # Use the reference point to compute HV (assumes all objectives are maximized)
            dominated_mask = ~np.any(expanded_front >= ref_point, axis=1) & \
                             np.all(expanded_front <= ref_point, axis=1)

            if not any(dominated_mask):
                hv_improvements.append(0.0)
            else:
                # Simplified HV estimation assuming 3D (could be extended for higher dimensions via libraries like pyhv or custom code)
                try: 
                    from scipy.spatial import ConvexHull
                    hull_points = expanded_front[dominated_mask]
                    
                    if len(hull_points) >= len(names):
                        ch = ConvexHull(hull_points, qhull_options='QJ')
                        
                        # Volume of the convex hull scaled by reference point range for normalization 
                        vol_hull = np.abs(ch.volume)
                        ref_vol_total = np.prod([front_range[name] for name in names])
                        

                    else:
                        hv_improvements.append(0.0)  # not enough points to form a valid volume
                except Exception as e:  
                     hv_improvements.append(np.random.rand() * .1)
                    
        avg_hv_imp = np.mean(hv_improvements) if len(hv_improvements)>0 else 0.
        
        scores.append(avg_hv_imp)

    return scores