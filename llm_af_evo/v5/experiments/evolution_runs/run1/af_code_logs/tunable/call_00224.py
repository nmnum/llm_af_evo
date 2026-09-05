def score_pool(context):
    """Score candidates by normalized acquisition value enhanced with a trend-aware uncertainty bonus that dynamically weights exploration vs exploitation based on recent progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base reward from acq_value_norm and normalize it
    acqs = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    if len(acqs) > 1:
        acq_mean, acq_std = np.mean(acqs), np.std(acqs)
        normalized_acq = (acqs - acq_mean) / max(1e-8, acq_std)
    else:
        normalized_acq = acqs

    # Trend-aware uncertainty bonus
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names) / len(names)

        sigma_norm = np.mean([gp[name]["std"] / front_range[name] 
                              for name in names if not np.isnan(front_range.get(name, 1))])

        # Trend-aware weight: increase exploration as progress stalls
        stagnation_factor = min(2.0, max(1.0, context["campaign"]["stagnant_batches"]))
        
        uncertainty_bonus = sigma_norm * (1 + (context['campaign']['progress'] < 0.3) * 
                                          np.log(stagnation_factor))
            
        # Combine acquisition and trend-aware bonus
        score = normalized_acq[None] if len(normalized_acq.shape)==0 else \
                ((normalized_acq - np.min(normalized_acq)) / max(1e-8, (np.max(normalized_acq) - np.min(normalized_acq)))) + \ 
               2.5 * uncertainty_bonus
        scores.append(score)
        
    return [float(s) for s in scores]