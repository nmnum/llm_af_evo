def score_pool(context):
    """Estimates improvement in hypervolume if candidate were to be observed, using sampled Pareto fronts under uncertainty."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Sample from each objective's GP posterior
    n_samples = 50
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Draw samples of objectives (each sample is a candidate)
        sampled_objectives = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"] 
            sampled_objectives[:,i] = np.random.normal(mean_val, std_val, n_samples)

        # Estimate hypervolume improvement from each sample
        hv_improvements = []
        
        for obj_sample in sampled_objectives:
            
            hypothetical_front = np.vstack([context["pareto_front"], obj_sample])
                
            # Compute dominated hypervolumes with ref_point 
            dominates_ref = (hypothetical_front <= ref_point).all(axis=1)
            if not any(dominates_ref):
                hv_improvement = 0.0
            else:
                front_dominated_by_ref = hypothetical_front[dominates_ref]
                
                # Compute hypervolume for this sample's contribution to the pareto set 
                try:  
                    from scipy.spatial import ConvexHull

                    hull_points = np.vstack([front_dominated_by_ref, ref_point])
                    
                    if len(hull_points) < 2:
                        hv_improvement = 0.0
                    else:
                        
                        # Use the volume of simplex formed by points (if dimension is small enough)
                        try:  
                            h = ConvexHull(hull_points)
                            
                            vol_sum = sum(np.abs(np.linalg.det(
                                np.diff([h.points[i] for i in face]+[ref_point], axis=0
                            ))) /np.math.factorial(len(ref_point)) 
                                        ) if len(face) == len(ref_point) else 1.0  
                                        
                        except:
                            
                            hv_improvement = float(np.prod((obj_sample - ref_point)[(obj_sample > ref_point)]))
                
                # Fallback to simpler hypervolume calculation for edge cases
                except Exception as e: 
                    try:

                         min_vals = np.minimum.reduce(hypothetical_front, axis=0)  
                         
                         hv_improvement = float(np.prod((min_vals - ref_point)[(min_vals > ref_point)]))

                    except:
                        hv_improvement = 1.0
                        
            hv_improvements.append(max(0.,hv_improvement))
        
        # Average improvement across all samples
        scores.append(float(np.mean(hv_improvements)))
    
    return scores