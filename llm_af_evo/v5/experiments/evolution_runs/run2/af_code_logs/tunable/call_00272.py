def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus to improve exploration."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    scores = []
    
    for cand in context["pool"]:
        acq_norm = cand["acq_value_norm"]  # already properly computed by botorch
        
        # Compute novelty as inverse of distance to nearest observed point
        x_cand = cand["x"]
        if len(X_obs) == 0:
            novel_score = 1.0
        else:
            distances = np.linalg.norm(X_obs - x_cand, axis=1)
            min_distance = np.min(distances)
            # Invert and normalize the distance to get novelty score in [0, 1]
            if min_distance == 0:
                novel_score = 1.0
            else:
                max_dist = np.max(np.linalg.norm(X_obs[:, None] - X_obs[None], axis=2))
                if max_dist > 0:
                    novel_score = (max_dist - min_distance) / max_dist
                else:
                    novel_score = 1.0
        
        # Blend acquisition value with novelty bonus, favoring acq_norm by a factor of 5:1  
        score = 0.8 * acq_norm + 0.2 * novel_score 
        scores.append(score)
    
    return scores