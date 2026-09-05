def score_pool(context):
    """Incentivize exploration by amplifying acquisition scores for candidates whose predicted objectives lie outside the current Pareto front's convex hull."""
    from scipy.spatial import ConvexHull
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # If we have fewer than 3 points, use all observations to define the hull
    if len(context["pareto_front"]) < 3:
        front_points_for_hull = context["Y_obs"]
    else:
        front_points_for_hull = context["pareto_front"]

    scores = []
    
    # Compute convex hull of Pareto front for reference (in objective space)
    try: 
        if len(front_points_for_hull) >= 3:
            hull = ConvexHull(front_points_for_hull)
            infront = False
        else:
            infront = True   # treat all as outside when not enough points to form a convex set
            
    except Exception:
        infront = True      # fallback: assume candidates are mostly outside
        
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        if len(front_points_for_hull) >= 3 and not infront:
            dist_to_front, _ = scipy.spatial.distance.cdist(
                [gp_mean], front_points_for_hull[hull.vertices]
            ).min(0)
            
            # Encourage candidates outside the convex hull (more diverse expansion potential)
            if np.all(gp_mean > ref_point):  # check dominance
                scores.append(cand["acq_value_norm"] * max(
                    [1.5, 
                     2 - dist_to_front / (
                         front_points_for_hull.max(axis=0) -  
                         front_points_for_hull.min(axis=0)
                     ).mean()
                 ]))
            else:
                # For candidates inside the convex hull (less promising), reduce score
                scores.append(cand["acq_value_norm"] * 0.5)

        elif len(front_points_for_hull) < 3 or infront: 
             # Less certainty, just use raw acquisition value with small boost for uncertainty if needed  
            sigma_sum = sum(
               cand["gp_posterior"][name]["std"]
                / context["pareto_front_range"][name] for name in names
           )
            
            scores.append(cand["acq_value_norm"] * (1 + 0.25*sigma_sum))

    return scores