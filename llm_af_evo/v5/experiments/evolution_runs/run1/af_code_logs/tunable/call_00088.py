def score_pool(context):
    """Rank candidates by a combination of normalized predicted quality and mutual diversity; higher is better."""
    names = context["objective_names"]
    
    # Compute raw qualities (sum of means) for each candidate
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)

    min_qual, max_qual = np.min(qualities), np.max(qualities)
    
    # Normalize quality to [0, 1] range
    if max_qual - min_qual < 1e-9:
        normed_quals = np.array([0.5]*len(qualities))
    else:
        normed_quals = (np.array(qualities) - min_qual)/(max_qual-min_qual)
    
    # Build kernel matrix of similarities between candidates in feature space
    X_pool = np.stack([cand["x"] for cand in context["pool"]])
    dists = np.sum((X_pool[:, None, :] - X_pool[None, :, :])**2, axis=2)**0.5
    
    # Avoid self-similarity by setting diagonal to large value
    np.fill_diagonal(dists, 1e9)
    
    similarities = np.exp(-dists) 

    scores = []
    for i in range(len(context["pool"])):
        qual_score = normed_quals[i]
        
        # Compute diversity score as mean similarity with all other candidates (excluding self)
        div_score = np.mean(similarities[i, :])
    
        final_score = qual_score * (1.0 - div_score)  # Higher quality and lower similarity
        scores.append(final_score)

    return scores