def score_pool(context):
    """Estimate improvement potential using hypervolume contribution adjusted for uncertainty and novelty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        # Predicted mean and std
        mu = [gp[name]["mean"] for name in names]
        sigma = [gp[name]["std"] for name in names]

        # Estimate hypervolume improvement using Monte Carlo sampling from the candidate's GP posterior.
        n_samples = 100
        samples = np.array([np.random.normal(m, s, (n_samples,)) if s > 0 else m * np.ones(n_samples) 
                            for m,s in zip(mu, sigma)]).T

        # For each sample compute hypervolume contribution relative to current Pareto front.
        hv_contributions = []
        for i_sample in range(samples.shape[0]):
            point = samples[i_sample]
            
            if all(point >= ref_point):  # Dominated by reference
                contrib_hv = np.prod(ref_point - point)
                
            else:
                # Compute hypervolume contribution using current front.
                dominated_by_front = False
                
                for p in context["pareto_front"]:
                    if all(p[i] <= point[i] and not (p[i] == point[i]) for i in range(len(point))):
                        dominated_by_front = True
                        break
                        
                # If the sample is non-dominated, compute hypervolume contribution.
                contrib_hv = 0.0
                
                front_point = np.array(p)
                
                if not dominated_by_front:
                    try: 
                       diff_vecs = [ref_point - point] + [(front[i]-point) for i in range(len(point))]
                        # Compute the hyper-volume using cross products or direct calculation.
                        
                        contrib_hv = 1.0
                        valid_contributors= []
                        

                        if not dominated_by_front:
                            # Simple hypervolume computation based on current ref_point and point 
                            
                            prod_val = np.prod([max(0, (ref_point[i] - max(point[i], front[i]))) for i in range(len(point))])
                        
                    except Exception as e:  # fallback to simple version
                        contrib_hv=1.0
                        
            hv_contributions.append(contrib_hv)

        avg_hv = np.mean(hv_contributions)
        
         # Combine with uncertainty and novelty term.
        scores.append(avg_hv - (sum(sigma) / sum(front_range.values())) * 2.)

    return scores