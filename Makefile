PYTHON ?= python3

.PHONY: setup check run-r1 analyze-r1 run-r2 run-r2b analyze-r2b normalize-runs analyze-runs

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e .

check:
	$(PYTHON) -m py_compile \
		run_experiment_r2.py \
		analyze_r2.py \
		experiments/prompt-injection-boundary-tags/run_experiment.py \
		experiments/prompt-injection-boundary-tags/analyze.py \
		experiments/prompt-injection-boundary-tags/run_experiment_round2.py \
		experiments/prompt-injection-boundary-tags/run_experiment_r2.py \
		experiments/prompt-injection-boundary-tags/analyze_r2.py \
		experiments/prompt-injection-boundary-tags/rounds/round1/harness/run_experiment.py \
		experiments/prompt-injection-boundary-tags/rounds/round1/analysis/analyze.py \
		experiments/prompt-injection-boundary-tags/rounds/round2/harness/run_experiment.py \
		experiments/prompt-injection-boundary-tags/rounds/round2b/harness/run_experiment.py \
		experiments/prompt-injection-boundary-tags/rounds/round2b/analysis/analyze.py \
		tools/normalize_prompt_injection_runs.py \
		tools/analyze_prompt_injection_runs.py

run-r1:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round1/harness/run_experiment.py

analyze-r1:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round1/analysis/analyze.py

run-r2:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round2/harness/run_experiment.py

run-r2b:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round2b/harness/run_experiment.py

analyze-r2b:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round2b/analysis/analyze.py

normalize-runs:
	$(PYTHON) tools/normalize_prompt_injection_runs.py

analyze-runs:
	$(PYTHON) tools/analyze_prompt_injection_runs.py
