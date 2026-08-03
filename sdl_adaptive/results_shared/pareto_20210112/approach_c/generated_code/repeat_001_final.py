"""
optimiser_template.py — Default optimiser for Approach C.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def suggest(X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Suggest the next experiment point using a GP-based acquisition function.
    """
    d = bounds.shape[0]
    n = len(X_obs)
    
    # 1. ALWAYS initialize a fallback random point to prevent returning None
    x_next = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])

    try:
        # 2. Only attempt GP if we have enough data (n > d)
        if n > d:
            # Instantiate the model correctly as an object, not a class alias[cite: 12]
            # normalize_y=True is critical for datasets with large values like 'coatings'
            model = GaussianProcessRegressor(
                kernel=Matern(nu=2.5), 
                normalize_y=True, 
                n_restarts_optimizer=3
            )
            model.fit(X_obs, y_obs)
            
            # 3. Use the required candidate grid pattern[cite: 9]
            n_candidates = 300
            candidates = np.column_stack([
                np.random.uniform(bounds[i, 0], bounds[i, 1], n_candidates) 
                for i in range(d)
            ])
            
            # 4. Predict and score (Example: Upper Confidence Bound)
            mu, sigma = model.predict(candidates, return_std=True)
            beta = 2.0
            acquisition_scores = mu + beta * sigma
            
            # Update suggestion if GP logic succeeds
            x_next = candidates[np.argmax(acquisition_scores)]

    except Exception:
        # If any internal error occurs, the pre-defined random x_next is returned[cite: 10]
        pass
        
    return x_next.flatten() # Ensures a (d,) shape[cite: 9]