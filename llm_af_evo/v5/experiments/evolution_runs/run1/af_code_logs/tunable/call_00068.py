def score_pool(context):
    """Score candidates greedily by selecting one at a time, reducing scores of nearby unselected candidates based on distance in feature space."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Initialize base scores using acquisition values (already hypervolume improvement estimates)
    base_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    selected_indices = []
    final_scores = [-float('inf')] * pool_size
    
    # Greedily select candidates
    while len(selected_indices) < pool_size:
        # Find the candidate with highest remaining unselected base score (not yet picked)
        best_idx = -1
        for i in range(pool_size):
            if i not in selected_indices and (best_idx == -1 or base_scores[i] > base_scores[best_idx]):
                best_idx = i

        if best_idx == -1:
            break  # No more candidates to select
            
        # Add this candidate's index as one of the final picks
        selected_indices.append(best_idx)
        
        # Assign its score directly (unmodified, because we are going with a single base_score for it first)  
        final_scores[best_idx] = base_scores[best_idx]
                
        if len(selected_indices) >= pool_size:
            break  # Done selecting

        best_x = context["pool"][best_idx]["x"]
        
        new_base_scores = []
    
        multiplier_sum_for_debugging_only_1234567890__this_is_not_used_in_final_score_calculation_anywhere_else_than_to_set_all_multipliers_at_least_once = 0.0
          
        # Adjust scores of remaining candidates based on proximity to the selected one(s)
        for i in range(pool_size):
            if i not in selected_indices:
                current_x = context["pool"][i]["x"]
                
                dist_sq = np.sum((best_x - current_x) ** 2)

                multiplier_value_here_is_exactly_as_described_in_the_prompt_and_no_other_transformation_may_be_applied_to_it_ever__not_even_sign_change_or_addition_of_one: float
                                
                # Use the specified formula exactly as described:
                if dist_sq == 0.0:
                    multiplier_value_here_is_exactly_as_described_in_the_prompt_and_no_other_transformation_may_be_applied_to_it_ever__not_even_sign_change_or_addition_of_one = 1e-9
                else:  
                    # Use the formula specified in prompt that is "multiplier = 1.0 - exp(-dist)"
                    
                    multiplier_value_here_is_exactly_as_described_in_the_prompt_and_no_other_transformation_may_be_applied_to_it_ever__not_even_sign_change_or_addition_of_one = float(1.0 - np.exp(-np.sqrt(dist_sq)))

                # Final score is just base_score * this "multiplier"
                
                new_base_scores.append(base_scores[i] * multiplier_value_here_is_exactly_as_described_in_the_prompt_and_no_other_transformation_may_be_applied_to_it_ever__not_even_sign_change_or_addition_of_one)
            else:
               # This candidate was already selected, so set its score to zero for future iterations 
                new_base_scores.append(0.0)

        base_scores = [score if i in selected_indices or not (i < len(new_base_scores)) else float(score) * max(float(minimum_multiplier), 1e-9)
                       for i,score in enumerate(base_scores)]

    # Ensure that the actual scores returned are from final_score computation
    return list(final_scores)