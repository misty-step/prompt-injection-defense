PYTHON ?= python3

.PHONY: setup check check-wrappers smoke-analyze ci-smoke test run-r1 analyze-r1 run-r2 run-r2b analyze-r2b calibrate-r2b run-r3 analyze-r3 run-r4 analyze-r4 run-r5 analyze-r5 run-r6 analyze-r6 run-r7 analyze-r7 normalize-runs analyze-runs run-opencode analyze-opencode

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
		experiments/prompt-injection-boundary-tags/shared/scoring/scorer.py \
		experiments/prompt-injection-boundary-tags/shared/budget/controller.py \
		experiments/prompt-injection-boundary-tags/rounds/round3/harness/run_experiment.py \
		experiments/prompt-injection-boundary-tags/rounds/round3/analysis/analyze.py \
		experiments/prompt-injection-boundary-tags/rounds/round4/harness/run_experiment.py \
		experiments/prompt-injection-boundary-tags/rounds/round4/analysis/analyze.py \
		experiments/prompt-injection-boundary-tags/rounds/round5/harness/run_experiment.py \
		experiments/prompt-injection-boundary-tags/rounds/round5/analysis/analyze.py \
		experiments/prompt-injection-boundary-tags/rounds/round6/harness/run_experiment.py \
		experiments/prompt-injection-boundary-tags/rounds/round6/analysis/analyze.py \
		experiments/prompt-injection-boundary-tags/rounds/round7/harness/run_experiment.py \
		experiments/prompt-injection-boundary-tags/rounds/round7/analysis/analyze.py \
		tools/check_compat_wrappers.py \
		tools/normalize_prompt_injection_runs.py \
		tools/analyze_prompt_injection_runs.py \
		tools/calibrate_round2b_scorer.py

check-wrappers:
	$(PYTHON) tools/check_compat_wrappers.py

smoke-analyze:
	$(PYTHON) analyze_r2.py
	$(PYTHON) experiments/prompt-injection-boundary-tags/analyze.py
	$(PYTHON) tools/analyze_prompt_injection_runs.py

ci-smoke: check check-wrappers test smoke-analyze

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

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

calibrate-r2b:
	$(PYTHON) tools/calibrate_round2b_scorer.py

run-r3:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round3/harness/run_experiment.py

analyze-r3:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round3/analysis/analyze.py

run-r4:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round4/harness/run_experiment.py

analyze-r4:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round4/analysis/analyze.py

run-r5:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round5/harness/run_experiment.py

analyze-r5:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round5/analysis/analyze.py

run-r6:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round6/harness/run_experiment.py

analyze-r6:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round6/analysis/analyze.py

run-r7:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round7/harness/run_experiment.py

analyze-r7:
	$(PYTHON) experiments/prompt-injection-boundary-tags/rounds/round7/analysis/analyze.py

normalize-runs:
	$(PYTHON) tools/normalize_prompt_injection_runs.py

analyze-runs:
	$(PYTHON) tools/analyze_prompt_injection_runs.py

run-opencode:
	cd experiments/opencode-agent-models && while IFS= read -r model; do \
		[ -z "$$model" ] || [ "$${model#\#}" != "$$model" ] && continue; \
		./run-test.sh "$$model"; \
	done < models.txt

analyze-opencode:
	cd experiments/opencode-agent-models && ./analyze.sh
