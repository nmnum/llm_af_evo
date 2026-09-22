def modifier(context):
    """Adaptive entropy bonus based on posterior variance covariance structure to encourage diverse exploration paths."""
    import numpy as np
    
    pool = context["pool"]
    names = context['objective_names']
    
    # Compute the determinant of the posterior's covariance matrix for each candidate  
    values = []
    for cand in pool:
        gp_posterior = cand["gp_posterior"] 
        cov_matrix = np.zeros((len(names), len(names)))
        
        for i, name_i in enumerate(names):
            std_i = gp_posterior[name_i]["std"]
            
            # Fill diagonal
            cov_matrix[i,i] = std_i ** 2
            
            # Off-diagonal terms (assume zero correlation to simplify)
                
        det_cov = np.linalg.det(cov_matrix) 
        entropy_bonus = -np.log(det_cov + 1e-8) if det_cov > 0 else 0.0
        
        values.append(entropy_bonus * 0.2)

    return values