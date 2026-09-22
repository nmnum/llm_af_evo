def score_pool(context):
    """Blend acquisition value with uncertainty-weighted distance to the Pareto front's convex hull for progressive exploration."""
    from scipy.spatial import ConvexHull
    
    names = context["objective_names"]
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    if len(context["pareto_front"]) == 0:
        # No front yet, fall back to acquisition only
        return list(acq_values)
        
    # Compute distance from each candidate's mean prediction to the convex hull of Pareto points  
    pf_points = np.array([list(cand.values()) for cand in context["pool"] if "gp_posterior" not in cand or len(context["pareto_front"]) == 0])
    
    try:
        # Get all objective means from candidates
        candidate_means_list = []
        
        for i, cand in enumerate(context["pool"]):
            mean_vals = [cand['gp_posterior'][name]['mean'] for name in names]
            candidate_means_list.append(mean_vals)
            
        X_candidates = np.array(candidate_means_list) 
        
        # Build convex hull from current Pareto front
        pf_obj_values = context["pareto_front"]
        
        if len(pf_obj_values) < 2:
            distances_to_hull = [1.0] * len(context["pool"])
        else:  
            try:
                hull = ConvexHull(pf_obj_values)
                
                # For each candidate mean, compute distance to the convex hull
                from scipy.spatial.distance import cdist
                
                if X_candidates.shape[0]> 0 and pf_obj_values.shape[0] > 0 :
                    distances_to_hull = []
                    
                    for i in range(X_candidates.shape[0]):
                        point = np.array([X_candidates[i]])
                        
                        # Compute distance to the convex hull
                        dists = cdist(point, pf_obj_values)
                        min_dist = float(np.min(dists))
                            
                        if len(hull.simplices) > 1:
                            distances_to_hull.append(min_dist * (len(context["objective_names"]) - 1)) 
                        else:  
                            # Fallback for small number of points
                            distances_to_hull.append(float(0.5 + min_dist/2))
                elif X_candidates.shape[0] == 0 or pf_obj_values.shape[0]==0:
                    raise ValueError("Empty arrays")
                    
            except Exception as e1:
                
                print(f"ConvexHull error: {e1}")
            
    except (Exception,ValueError) :
        # Fallback in case of failure
        distances_to_hull = [np.random.rand() for _ in range(len(context["pool"]))]

    
     if len(distances_to_hull)==0:
         try : 
             X_candidates= np.array([[cand['gp_posterior'][name]['mean']  \
                                      for name in names]   \  
                                     for cand in context["pool"]]) 
            
              # If no front, just use acquisition score
               if len(context["pareto_front"]) <1:
                  return list(acq_values)
                  
             distances_to_hull = []
              
         except Exception as e :
            print(f"Error computing distance to hull: {e}")
            
     finally : 
        try :  
              # Fallback for any computation errors
               if not 'distances_to_hull' or len(distances_to_hull) == 0:
                  distances_to_hull = [1.0] *len(context["pool"])
                  
            except Exception as e :
                print(f"Error in fallback: {e}")
                
     # Normalize and weight distance term
    max_dist= np.max(distances_to_hull)
    
    if max_dist == 0:
        normalized_distances=[float(0.) for _ in distances_to_hull]
        
    else : 
         try :
             normalized_distances = [dist/max_dist   \
                                    * (len(context["objective_names"])/2. )  
                                     + float(np.random.rand()/4)   
                                      for dist in  distances_to_hull ]
              
        except Exception as e:
            print(f"Error normalizing distance: {e}")
            
    # Combine with acquisition value
     scores = []
     
      try : 
          for i, cand in enumerate(context["pool"]):
               w1,w2=0.75 , 0.25  
                score=w1*acq_values[i] + \
                       (w2 * normalized_distances [i]) 
            
              # Add some randomness to break ties and add robustness
                 noise = np.random.normal(0, float(.03))
                  final_score=score+noise 
                   scores.append(final_score)
                   
         except Exception as e :
            print(f"Error combining score: {e}")
            
      finally :  
          if len(scores)== 0:
              # Last resort fallback
               try: