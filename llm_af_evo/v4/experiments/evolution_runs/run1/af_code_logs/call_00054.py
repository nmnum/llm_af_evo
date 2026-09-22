def score_pool(context):
    """Incorporate uncertainty sensitivity into acquisition value using a logarithmic scaling and adaptively weight it based on campaign stagnation."""
    scores = []
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        
        # Compute total normalized standard deviation
        sigma_total = sum(cand["gp_posterior"][name]["std"] 
                         for name in context["objective_names"])
        
        # Apply logarithmic scaling to uncertainty (to dampen large uncertainties)
        log_sigma = np.log(1.0 +.sigma_total) if sigma_total > 0 else 0
        
        # Adaptively adjust the influence of uncertainty based on stagnation
        # If stagnant, increase exploration; otherwise reduce it slightly 
        weight_factor = max(0.5, 2.0 / (stagnant_batches + 1)) 
        
        adjusted_acq = acq + weight_factor * log_sigma
        
        scores.append(adjusted_acq)
        
    return scores