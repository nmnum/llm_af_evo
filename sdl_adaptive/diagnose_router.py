"""Run from sdl_adaptive/ directory: python diagnose_router.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(".").resolve()))

# ── Test 1: Context signals ──────────────────────────────────────────────────
print("=== Test 1: Context signals ===")
try:
    from oracle import NNOracle
    from controllers.rule_router import RuleRouterController
    import numpy as np

    oracle = NNOracle.from_dataset("pareto_campaign 2021-01-12_16-26-56", "data/")
    ctrl = RuleRouterController()
    context_seen = {}
    orig = ctrl.decide
    def patched(ctx):
        context_seen.update({k: v for k, v in ctx.items()
                             if k not in ('X_obs','y_obs','bounds')})
        return orig(ctx)
    ctrl.decide = patched

    from shared_seed_experiment import generate_shared_inits, run_sdl_campaign
    inits = generate_shared_inits(oracle, 1, 5, 42)
    X_init, y_init = inits[0]
    res = run_sdl_campaign(oracle, "pareto_20210112", X_init, y_init,
                           ctrl, budget=20, controller_interval=5)
    print(f"  lengthscale_norm: {context_seen.get('lengthscale_norm', 'MISSING'):.3f}")
    print(f"  n_dims: {context_seen.get('n_dims', 'MISSING')}")
    print(f"  obs_per_dim at first call: "
          f"{context_seen.get('n_obs',0)/max(context_seen.get('n_dims',1),1):.1f}")
    strategies = [s for _,s,_ in res['decisions']]
    print(f"  Decisions: {strategies}")
    print(f"  ✓ PASS" if len(set(strategies)) > 1 else "  ✗ Still stuck on one strategy")
except Exception as e:
    import traceback; traceback.print_exc()

# ── Test 2: Diagnose ollama qwen3 response format ────────────────────────────
print("\n=== Test 2: qwen3 response format ===")
try:
    import ollama

    # Check what models are available
    models = ollama.list()
    model_names = [m.model for m in models.models]
    print(f"  Available models: {model_names}")

    # Find a qwen3 model
    qwen3_models = [m for m in model_names if 'qwen3' in m.lower()]
    other_models = [m for m in model_names if 'qwen' in m.lower() and 'qwen3' not in m.lower()]
    print(f"  qwen3 models: {qwen3_models}")
    print(f"  other qwen models: {other_models}")

    # Test raw response structure
    for model in (qwen3_models + other_models)[:2]:
        print(f"\n  Testing model: {model}")
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": 'Say {"strategy": "lhs"} and nothing else'}],
                options={"temperature": 0, "num_predict": 128},
            )
            msg = resp["message"]
            print(f"    message keys: {list(msg.keys())}")
            print(f"    content: {repr(msg.get('content',''))[:200]}")
            # qwen3 may have thinking in a separate key
            if 'thinking' in msg:
                print(f"    thinking: {repr(str(msg['thinking'])[:100])}")
            # Try with think=False
            resp2 = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": 'Say {"strategy": "lhs"} and nothing else'}],
                options={"temperature": 0, "num_predict": 128, "think": False},
            )
            msg2 = resp2["message"]
            print(f"    with think=False content: {repr(msg2.get('content',''))[:200]}")
        except Exception as e:
            print(f"    FAILED: {e}")

except Exception as e:
    import traceback; traceback.print_exc()

print("\n=== Test 3: ollama Python client version ===")
try:
    import ollama
    print(f"  ollama version: {ollama.__version__ if hasattr(ollama,'__version__') else 'unknown'}")
    import importlib.metadata
    print(f"  package version: {importlib.metadata.version('ollama')}")
except Exception as e:
    print(f"  {e}")
