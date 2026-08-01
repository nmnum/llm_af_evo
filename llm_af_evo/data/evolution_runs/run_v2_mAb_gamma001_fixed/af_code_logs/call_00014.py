def score_pool(context):
    """Estimate improvement potential by resampling predicted objectives from each candidate's GP posterior and computing expected hypervolume gain."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Sample predictions from each candidate's GP posteriors
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        samples = []
        
        # Draw samples per objective, assuming Gaussian posterior (mean ± std * N(0,1))
        for name in names:
            mean = gp[name]["mean"]
            std  = gp[name]["std"]
            sample_vals = np.random.normal(mean, std, n_samples)
            samples.append(sample_vals)

        # Transpose to get shape [n_samples, n_objectives]
        sampled_objs = np.column_stack(samples) 

        # For each sample (i.e. predicted objective vector), compute hypervolume 
        # improvement relative to current Pareto front
        hv_improvements = []
        
        for i in range(n_samples):
            obj_vector = sampled_objs[i]  # shape: [n_objectives]
            
            # Check if this candidate's sample dominates any existing point on the PF  
            is_dominated = False 
            for pf_point in context["pareto_front"]:
                if all(obj_vector[j] >= pf_point[j] and
                       (obj_vector[j] > pf_point[j] or j == 0)) \
                        for j in range(len(names)): # at least one strict improvement  
                    is_dominated = True 
                    break
                    
            if not is_dominated:
                hv_improvement = hypervolume_contribution(obj_vector, context["pareto_front"], ref_point)
                hv_improvements.append(hv_improvement) 
            
        expected_hvi = np.mean(np.maximum(0.0, hv_improvements)) 
        scores.append(expected_hvi)

    return scores


def hypervolume_contribution(point, pf_points, reference):
    """Compute the contribution of a single point to the total hypervolume."""
    
    # If no points in PF (first batch), HV is simply volume from ref down
    if len(pf_points) == 0:
        vol = np.prod(reference - point)
        return max(0.0, vol)

    # Use efficient algorithm for computing contribution of one new point to existing front  
    n_obj = len(point)
    
    contributions = []
   
    for i in range(len(pf_points)):
        
        pf_point = pf_points[i]
                
        if all(p >= p_f and (p > p_f or j == 0) 
               for j, (p,p_f) in enumerate(zip(point, pf_point))): # dominates PF point
            continue
            
    new_front_with_p = [pf_point.copy() for pf_point in pf_points]
    
    # Remove dominated points by this one  
    to_remove = []
    for k in range(len(new_front_with_p)):
        if all(pf_point[j] >= p_f and (p_f > pf_point[j] or j == 0) 
               for j, (_,pf_point,j,p_f) in enumerate(zip(point,new_front_with_p[k], point))) \
                : # PF dominates new one
            to_remove.append(k)
            
    if not all(p <= r for p,r in zip(new_front_with_p[i],point)) or i == 0:
        continue

    contributions = []

    def dominated_by_any(pf_point, current):
        return any(all(c >= pf[j] and (c > pf[j] or j==i) 
                       for j,(pf,c) in enumerate(zip(point,pf_points)))  
                   for point_idx,point_ in enumerate(current)
                   if not i == point_idx)

    # Recompute hypervolume with this new candidate added
    extended_front = [p.copy() for p in pf_points] + [list(point)]
    
    vol_contribution  = compute_hv(extended_front, reference) - \
                        compute_hv(pf_points,reference)
                        
    contributions.append(vol_contribution)

    return np.mean(contributions)


def compute_hv(front, ref):
    """Compute hypervolume of front with respect to a given reference point."""
    
    if len(front) == 0:
        vol = np.prod(ref - [np.inf]*len(ref))
        return max(0.0,vol)
        
    # Use recursive definition for simplicity; better implementations exist 
    n_obj = len(ref)

    def hv_recursive(points):
        points.sort(key=lambda x: (-x[0],) + tuple(x[i] for i in range(n_obj-1)))  # sort by first objective desc
        if not points:
            return 0.0

        p_last, others = points[-1], points[:-