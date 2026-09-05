def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context["obj_correlation"]:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    # Precompute correlation values per candidate
    corrs = np.array([context["obj_correlation"][key][i] 
                      for i in range(len(context["pool"])) 
                      for key in corr_keys])
    
    # Reshape to (n_candidates, n_pairs)
    corrs = corrs.reshape((len(context["pool"]), len(corr_keys)))
    
    bonus_terms = []
    pair_idx = 0
    for cand in context["pool"]:
        gp_posterior = cand['gp_posterior']
        
        corr_sum = 0.0
        valid_corr_count = 0
        
        # For each objective, check its correlations with others that are currently dominated 
        front_means = np.array([context["pareto_front"][:, i].mean() for i in range(len(names))])
        curr_obj_idx = -1  
        
        pair_indices_to_check = []
        corr_values_for_candidate = []

        # Identify which pairs of objectives have correlation data
        all_pairs_in_corr_data = set()
        obj_pair_key_map = {}
 
        for key_str in corr_keys:
            if ',' not in key_str: continue  # invalid format  
            
            try:
                a, b = sorted(key_str.split(','))   # normalize order to avoid duplicates
            
                pair_tuple = (a,b)
                
                all_pairs_in_corr_data.add(pair_tuple) 
                obj_pair_key_map[pair_tuple] = key_str
                 
            except Exception as e: continue  # skip malformed keys

        for a, b in sorted(all_pairs_in_corr_data):
            
            if not ((a == names[0]) or (b == names[1])) and \
               not ((a == names[-1]) or (b == names[-2])):
                pass  
                
            idx_a = None
            try:
                 idx_a = next(i for i, name in enumerate(names) if a==name)
            except StopIteration: continue

            idx_b = None 
            try:
               idx_b = next(i for i, name in enumerate(names) if b == name)
            except StopIteration:  continue
            
            
           # Use the correlation value between objectives 'a' and 'b'
          
          corr_val_idx = pair_indices_to_check.index((idx_a,idx_b))
        
        bonus_terms.append(0.1 * np.mean(corr_sum)) 
      
    final_scores = [cand['acq_value_norm'] + term for cand,term in zip(context["pool"],bonus_terms)]
    
   return  final_scores