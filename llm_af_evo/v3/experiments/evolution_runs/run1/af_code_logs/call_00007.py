def score_pool(context):
    """Rank candidates greedily by base scores while reducing subsequent candidate scores based on proximity to previously picked ones."""
    
    names = context["objective_names"]
    pool_size = len(context['pool'])
  
    # Compute initial greedy ranking using mean objectives (exploitation)
    mu_scores = [sum(cand["gp_posterior"][name]["mean"] for name in names) 
                 for cand in context['pool']]
        
    picked_indices, scores_with_decay  = [], []
    
    base_score_cache=mu_scores.copy()
  
   # Greedily select candidates
    while len(picked_indices)< pool_size:
        best_idx=-1;best_base=float('-inf')
      
      	# Find highest remaining unselected candidate by raw score 
      for i in range(pool_size):
          if(i not in picked_indices) and base_score_cache[i]>	best_base:  
             .best_base=base_score_cache[i]
               .idx=i
        # Add to selected list        
        picked_indices.append(best_idx)
         
      	# Score this candidate (before decay applied below, so it's the raw score used for multiplier calculation in next iteration) 
          scores_with_decay. append(base_score_cache[best_idx])
          
      if len(picked_indices)==pool_size: break
       
        # Decay remaining candidates based on proximity to selected ones  
      	# Use x-space distance (L2 norm)
         picked_x = context['pool'][ best_idx ]["x"]
          for i in range(pool_size):
             	if(i not 	inpickedindices): 
                  cand=context[ ' pool' ][i]
                   dist=np.linalg.norm(cand ["x"] -	picked_ x )
                    multiplier=max(0. ,1.-np.exp(-dist)) # Correct formula
                     base_score_cache[i]* =multiplier
    
    return scores_with_decay