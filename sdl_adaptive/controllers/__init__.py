"""
controllers/ — LLM-based strategy controllers.

Three levels of autonomy:
  ApproachAController : LLM tunes UCB β only (scalar output)
  ApproachBController : LLM selects from 6 named strategies
  ApproachCController : LLM rewrites the optimiser Python file entirely

Mock versions (no Ollama required):
  MockApproachAController
  MockApproachBController
  MockApproachCController
"""
