# excipient_adapter.py — save in same directory as posthoc_llm_comparison.py
from excipient_oracle import ExcipientOracle, PROTEIN_PROFILES

def make_excipient_nn_oracle(protein="mAb_aggregation", n_samples=500, seed=42):
    """Wrap ExcipientOracle to match NNOracle interface."""
    oracle_cont = ExcipientOracle(protein=protein, seed=seed)
    disc = oracle_cont.make_discrete_oracle(n_samples=n_samples, seed=seed)
    
    # Patch to match NNOracle interface exactly
    disc._scaler = disc._scaler          # already exists
    disc.global_best = disc.global_best  # already exists
    disc.bounds = disc.bounds            # already exists
    
    # NNOracle stores raw y separately from query tracking
    disc._y_raw = disc._y_raw            # already exists
    disc._X_raw = disc._X_raw            # already exists
    
    # Add dataset name for logging
    disc._protein = protein
    return disc