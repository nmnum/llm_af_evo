def score_pool(context):
    """Balances exploitation and uncertainty via dynamic weighting conditioned on front diversity and stagnation."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]

    # Compute how much the current Pareto front spans each objective
    pf_range = np.array([context["pareto_front_range"][name] for name in names])
    
    # Normalize candidate means by pareto range (for exploitation)
    mu_norms = []
    sigma_sums = []

    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum_norm = sum(gp[name]["mean"] / pf_range[i] for i, name in enumerate(names))
       sigma_sum_scaled = sum(
            gp[name]["std"] * (1.0 + 2.0 * min(stagnant_batches/5., 1.)) 
            for name in names
        )
        
        mu_norms.append(mu_sum_norm)
        sigma_sums.append(sigma_sum_scaled)

    # Use front diversity to modulate exploration vs exploitation strength  
    if len(context["pareto_front"]) > 2:
        pf_spread = np.max(context["pareto_front"], axis=0) - np.min(context["pareto_front"], axis=0)
        avg spread_ratio = np.mean(pf_spread / pf_range)
        
        # When front is diverse, trust exploitation more; when narrow (stagnant), explore
        w_exploit_base = 1.0/(1 + np.exp(-5 * (avgspread_ratio - 0.3)))
    else:
        w_exploit_base = max(0.2, progress) 

    # Blend based on current campaign state 
    w_exploit = min(w_exploit_base, 0.9)

    scores = []
    for i in range(len(context["pool"])):
        score = (w_exploit * mu_norms[i] +  
                (1 - w_exploit) * sigma_sums[i])
        
        scores.append(score)
    
    return scores