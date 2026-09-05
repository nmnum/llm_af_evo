def score_pool(context):
    """Resample noisy GP predictions to assess candidate robustness and hypervolume expansion potential."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Collect samples from the joint posterior (simulate noise)
        n_samples = 20
        mean_vals = [gp[name]["mean"] for name in names]
        std_vals = [gp[name]["std"] for name in names]

        sampled_objectives = []
        for _ in range(n_samples):
            sample_obj = [
                np.random.normal(mean, std) 
                for mean, std in zip(mean_vals, std_vals)
            ]
            sampled_objectives.append(sample_obj)

        # Estimate how many of these samples would improve the hypervolume
        ref_point = context["ref_point"]
        
        hv_improvements = []
        for sample in sampled_objectives:
            expanded_pf = np.vstack([pf, sample])
            
            # Compute dominated hypervolume with this new point included 
            try:  # Use a simple dominance check approach to estimate HV improvement
                dominates_any = False
                for pf_point in pf:
                    if all(sample[i] >= pf_point[i] and not (sample[i] == pf_point[i]) for i in range(len(names))):
                        dominates_any = True  
                        break

                hv_improvement = 0.0 
                
                # If the sample is non-dominated, it can improve HV
                if not dominates_any:
                    hypervolume_contrib = np.prod(np.maximum(ref_point - np.array(sample), 0))
                    
                    # Check how much this contributes to total dominated region (simplified)
                    hv_improvement += max(1e-8, hypervolume_contrib)

            except Exception:  
                hv_improvement = 0.0

            hv_improvements.append(hv_improvement) 

        avg_hypervolume_impact = np.mean(hv_improvements)
        
        # Blend with acquisition value
        acq_value_norm = cand["acq_value_norm"]
        final_score = (1 - 0.2 * max(0, len(pf)-5)) * acq_value_norm + \
                      avg_hypervolume_impact
        
        scores.append(final_score)

    return scores