def score_pool(context):
    """Score candidates by their predicted objective vector's projection onto the direction of maximum recent improvement, weighted by acquisition value and regularized with a distance-based novelty bonus."""
    names = context["objective_names"]
    y_obs = context["Y_obs"]
    pareto_front = context["pareto_front"]
    
    # Early exit if not enough observations for momentum
    if len(y_obs) < 4:
        return [cand['acq_value_norm'] for cand in context['pool']]
        
    n_half = len(y_obs) // 2
    older_y = y_obs[:n_half]
    newer_y = y_obs[n_half:]
    
    # Compute momentum direction (change in mean outcome over time)
    old_mean = np.mean(older_y, axis=0)
    new_mean = np.mean(newer_y, axis=0)
    momentum_direction = new_mean - old_mean
    
    # Normalize the momentum vector
    mom_norm = np.linalg.norm(momentum_direction) + 1e-9
    if mom_norm == 0:
        return [cand['acq_value_norm'] for cand in context['pool']]
    
    norm_mom_dir = momentum_direction / mom_norm

    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Get predicted means
        pred_means = np.array([gp_posterior[name]["mean"] for name in names])
        
        # Compute dot product with direction of recent improvement (positive is good)
        proj_score = max(0, np.dot(pred_means - new_mean, norm_mom_dir))
        
        acq_value_norm = cand['acq_value_norm']
        
        if pareto_front.size == 0:
            novelty_bonus = 1.0
        else:
            # Compute distance to nearest front point in objective space  
            distances_to_pf = np.linalg.norm(pred_means - pareto_front, axis=1)
            min_dist_to_pf = np.min(distances_to_pf) + 1e-9
            
            # Novelty bonus: higher when farther from existing non-dominated points
            novelty_bonus = max(0.5, 2.0 / (min_dist_to_pf ** 0.5))
        
        scores.append(acq_value_norm * proj_score * novelty_bonus)
    
    return scores