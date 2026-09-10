"""Public roster projected from actual role/tool boundaries, not independent workers."""

ROSTER = [
    {"id": "framer", "name": "Problem framer", "model_role": "heavy", "entry": "solve", "input": "Problem + supplied context + steering", "output": "ProblemDNA"},
    {"id": "complexity_architect", "name": "Complexity architect", "model_role": "heavy", "entry": "represent_system", "input": "DNA, evidence, current representation, capabilities", "output": "ComplexSystemSpec"},
    {"id": "hypothesis_critic", "name": "Hypothesis critic", "model_role": "validator", "entry": "review_hypotheses", "input": "Current system + scoped evidence", "output": "HypothesisReview"},
    {"id": "evidence_synthesist", "name": "Evidence synthesist", "model_role": "heavy", "entry": "synthesize_evidence", "input": "Source artifacts + hypotheses", "output": "EvidenceSynthesis"},
    {"id": "solution_designer", "name": "Exploratory intervention designer", "model_role": "heavy", "entry": "develop_candidate", "input": "System alternatives + evidence + parent design", "output": "CandidateDesign"},
    {"id": "experiment_designer", "name": "Experiment designer", "model_role": "heavy", "entry": "plan_experiment", "input": "System experiment + validation checks", "output": "ExperimentProtocol"},
    {"id": "simulation_engineer", "name": "Simulation and inference engineer", "model_role": "heavy", "entry": "run_model_code", "input": "Files, scientific contracts, explicit predecessor", "output": "Executed code + result/visual files"},
    {"id": "checker", "name": "Output checker", "model_role": "validator", "entry": "review_model", "input": "Actual code/results, evidence and contract excerpts", "output": "DeliveryReview; no rerun implied"},
    {"id": "decision_editor", "name": "Decision editor", "model_role": "heavy", "entry": "build_outcome", "input": "Artifact-linked interpretations and comparisons", "output": "DecisionBrief"},
    {"id": "method_optimizer", "name": "Method revision proposer", "model_role": "heavy", "entry": "propose_method", "input": "Recorded critiques and evaluation feedback", "output": "Uninstalled MethodCandidate"},
    {"id": "metareasoner", "name": "Metareasoning controller", "model_role": "heavy", "entry": "solve", "input": "Current state, uncertainty, feedback and remaining resources", "output": "NextStep"},
]
