def score_pool(context):
    """Integrate acquisition value with inverse squared distance to pareto front weighted by objective variance."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if Pareto front is too small for reliable k-NN
    use_y_obs = len(pf) < 3
    
    scores = []
    for cand in context["pool"]:
        gp_mean = [cand["gp_posterior"][name]["mean"] for name in names]
        
        distances = []
        reference_points = pf if not use_y_obs else context["Y_obs"]
        cand_point = np.array(gp_mean)

        # Compute Euclidean distance to all reference points
        for ref_point in reference_points:
            dist = np.linalg.norm(cand_point - ref_point, ord=2)
            distances.append(dist)
        
        sorted_distances = sorted(distances)[:min(3, len(distances))]
        mean_dist_to_front = sum(sorted_distances)/len(sorted_distances) if sorted_distances else 0.0
        
        # Weight by inverse square of objective variance (more uncertain directions get less weight in distance metric)
        obj_vars = [cand["gp_posterior"][name]["std"] ** 2 for name in names]
        total_var = sum(obj_vars)

        inv_dist_weighted = mean_dist_to_front / max(total_var, 1e-8) if not use_y_obs else -mean_dist_to_front
        
        # Combine acquisition value with the modified distance metric
        acq_value_norm = cand["acq_value_norm"]
        
        final_score = acq_value_norm + (0.3 * inv_dist_weighted)
        scores.append(final_score)

    return scores