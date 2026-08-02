def score_pool(context):
    """Estimate each candidate's potential hypervolume improvement using Monte Carlo samples from their posteriors (sign-fixed from run_v2_mAb_gamma001_fixed's call_00021: a sample counts as improving only when it is NOT dominated by the current Pareto front, not when it happens to dominate a front point -- the original check was inverted). Note: the nearest-neighbour diversity bonus promised by the original docstring was never implemented in call_00021's code and is not added here -- this fix corrects only the dominance-direction bug, nothing else."""
    names = context["objective_names"]

    n_samples = 100
    scores = []

    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"]
            samples[:, i] = np.random.normal(mean_val, std_val, n_samples)

        hv_improvement = 0.0

        for sample in samples:
            is_dominated = False

            # FIXED: original checked "all(sample[i] >= pf_point[i])" -- i.e.
            # whether the SAMPLE dominates a front point -- and excluded such
            # samples from the improvement count, backwards from what
            # estimating HV improvement requires. A sample should count as an
            # improvement when NO existing front point dominates it.
            for pf_point in context["pareto_front"]:
                if (all(pf_point[i] >= sample[i] for i in range(len(names))) and
                        any(pf_point[i] > sample[i] for i in range(len(names)))):
                    is_dominated = True
                    break

            hv_improvement += 1.0 if not is_dominated else 0.0

        avg_hv_impact = hv_improvement / n_samples
        scores.append(avg_hv_impact)

    return scores
