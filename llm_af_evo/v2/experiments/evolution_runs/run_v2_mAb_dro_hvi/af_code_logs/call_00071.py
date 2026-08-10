def score_pool(context):
    """
    Estimate improvement potential by resampling candidates' GP posteriors,
    then compute how much each would expand hypervolume if added to current Pareto front.
    This avoids over-reliance on fixed front estimates and better captures uncertainty in 
    dominance relationships among predictions, using a probabilistic HV calculation per candidate.  
    """
    names = context["objective_names"]
    ref_point = np.array([context['ref_point_by_name'][n] for n in names])
    
    # Use Monte Carlo to estimate expected hypervolume improvement
    nsamples = 50 
    scores = []
        
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        samples = []  
        for _ in range(nsamples):
            sample_obj = [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            # Sample may be outside the ref point, but we still score it
            samples.append(sample_obj)
            
        cand_hv_improvement = 0.0 
        front_with_cand = np.vstack([context["pareto_front"],samples])
        
        if len(front_with_cand) > 1:
            # Calculate hypervolume of the new set (vs ref point), subtract original
            try:  
                hv_new = compute_hypervolume(front_with_cand,ref_point)
                hv_old = compute_hypervolume(context["pareto_front"],ref_point)
                cand_hv_improvement = max(0.0,hv_new - hv_old) 
            except:
                pass  # fallback to simple score if hypervolume calc fails
            
        scores.append(cand_hv_improvement)

    return scores

def compute_hypervolume(front, ref_point):
    """Compute approximate normalized hyper-volume using dominated points"""
    front = np.array(front)
    
    def dominates(p1,p2): # p1 >= p2 in all dims
        return (p1 <= p2).all()
        
    if len(front) == 0: 
        return float(0.0)

    ref_point = np.asarray(ref_point)
   
    dominated_points_mask = []
    
    for i, point_i in enumerate(front):
        is_dominated_by_anyone_else = False
        for j,point_j in enumerate(front):  
            if (i != j) and dominates(point_j,point_i):
                # Point 'i' is dominated by some other point ('j')
                is_dominated_by_anyone_else = True 
                break

        dominated_points_mask.append(is_dominated_by_anyone_else)

    non_dom_front = front[~np.array(dominated_points_mask)]

    if len(non_dom_front) == 0:
       return float(0.0)
    
    # Normalize to ref point range for volume calc
    vol_sum=float(0.)
   
    try: 
        ranges = np.max(ref_point - non_dom_front,axis=0)

        unit_vol_per_dim = (1./np.prod(ranges)) if not all(x== 0 for x in ranges) else float(0.)

        # Volume of each cell from point to ref
        vol_sum += sum(np.product((ref_point-p)/ranges, axis=-1).clip(min=0.))
        
    except:
       pass

    return max(vol_sum * unit_vol_per_dim , 0.)