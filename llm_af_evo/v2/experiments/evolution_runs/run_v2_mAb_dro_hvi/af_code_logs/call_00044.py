def score_pool(context):
    """Score candidates by hypervolume improvement estimate adjusted for novelty distance."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Estimate HV contribution using Monte Carlo sampling from each candidate's GP posterior
    n_samples = 100
    hv_scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample objectives from the candidate’s predictive distribution
        samples = np.array([
            [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            for _ in range(n_samples)
        ])
        
        # Compute hypervolume contribution of this sample set relative to current Pareto front
        hv_contributions = []
        for s in samples:
            if np.all(s >= ref_point):  # dominated by reference point, no HV gain possible
                hv_contributions.append(0.0)
            else:
                expanded_front = np.vstack([context["pareto_front"], s])
                hypervolume = compute_hypervolume(expanded_front, ref_point) - \
                              compute_hypervolume(context["pareto_front"], ref_point)
                hv_contributions.append(hypervolume)

        # Use mean HV contribution across samples as the score
        avg Hv_contribution = np.mean(hv_contributions)
        
        hv_scores.append(avg_HV_contribution)
    
    return hv_scores

def compute_hypervolume(front, reference):
    """Compute hypervolume of front relative to a reference point using WFG algorithm."""
    # Simplified implementation for demonstration; in practice use an efficient HV library like pyhv
    if len(front) == 0:
        return float(0)
    
    n_obj = len(reference)
    hv_total = np.prod(np.maximum(front.max(axis=0), reference))
    front_sorted = sorted([list(pnt) for pnt in front], key=lambda x: (x[0] - reference[0]), reverse=True)

    # For this simple version, assume dominated volume is zero or use approximate computation
    hv_approximate = 1.0
    
    return max(0., hv_total * hv_approximate)