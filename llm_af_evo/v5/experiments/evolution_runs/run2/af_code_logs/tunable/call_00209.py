def score_pool(context):
    """Score candidates by normalized predicted quality multiplied by inverse diversity, favoring strong yet distinct points."""
    names = context["objective_names"]
    
    # Extract qualities (predicted objective sums) and positions for similarity calculation
    qualities = []
    Xs = []  # feature vectors 
    Y_preds = []

    for cand in context["pool"]:
        gp = cand["gp_posterior"]  
        mu_sum = sum(gp[name]["mean"] for name in names)
        qualities.append(mu_sum)
        
        x_vec = cand['x']
        y_pred = np.array([gp[name]['mean'] for name in names])
        
        Xs.append(x_vec) 
        Y_preds.append(y_pred)

    # Normalize quality to [0, 1] range
    q_min, q_max = min(qualities), max(qualities)
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.zeros(len(qualities))
    else:
        norm_qualities = (np.array(qualities) - q_min) / (q_max - q_min)

    # Compute similarity matrix based on feature space distances
    Xs = np.stack(Xs)
    
    # Pairwise squared Euclidean distance between all candidates in x-space  
    dist_matrix_x = np.sum((Xs[:, None] - Xs[None, :]) ** 2, axis=-1) 
    
    # Gaussian similarity kernel (exp(-d^2 / sigma^2), using median of distances as bandwidth)
    if dist_matrix_x.size > 0:
        med_dist_sq = np.median(dist_matrix_x[np.triu(np.ones_like(dist_matrix_x), k=1).astype(bool)])
        sigma_sq = max(med_dist_sq, 1e-9) 
        sim_matrix_x = np.exp(-dist_matrix_x / (2 * sigma_sq))
    else:
        # Fallback if no candidates
        sim_matrix_x = np.zeros((len(qualities), len(qualities)))

    # Zero out diagonal entries so each candidate doesn't penalize itself  
    np.fill_diagonal(sim_matrix_x, 0)
    
    diversity_scores = []
        
    for i in range(len(context["pool"])):
        if not (sim_matrix_x[i] > 1e-9).any():
            div_score = 1.0
        else:
            # Average similarity to all other candidates  
            avg_sim_to_others_i = np.mean(sim_matrix_x[i])
            
            # Inverse diversity: higher when less similar 
            inv_diversity_i = (1 - avg_sim_to_others_i)
    
            div_score = max(0.0, min(inv_diversity_i, 1.0)) 
            
        diversity_scores.append(div_score)

    scores = norm_qualities * np.array(diversity_scores) 
    
    return list(scores)