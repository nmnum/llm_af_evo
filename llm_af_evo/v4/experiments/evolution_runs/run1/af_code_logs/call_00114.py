def score_pool(context):
    """Incorporate uncertainty into acquisition scores with a dynamic weighting based on front diversity and stagnation."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Compute the average distance from each candidate to the nearest observed point
    X_obs = context["X_obs"]
    scores = []
    
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]

        # Add uncertainty bonus, scaled by how stagnant we've been and inversely related to front diversity 
        ucb_bonus = 0.1 * (1 - progress) * sum(cand["gp_posterior"][name]["std"] for name in names)
        
        # If stagnation is high or the candidate's predicted objectives are near current pareto frontier,
        # reduce exploitation and increase uncertainty bonus to encourage exploration
        if stagnant_batches > 2:
            ucb_bonus *= (1 + min(0.5, max(cand["gp_posterior"][name]["mean"] for name in names) / context["pareto_front_range"]["f1"]))
        
        scores.append(acq + ucb_bonus)

    return scores