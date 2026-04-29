.PHONY: install survey generate prepare noise train evaluate inference api test eda diagnose gold calibration crf diversity talkbank all clean


install:
	pip install -r requirements.txt
	pip install -e .
survey:
	python scripts/run_survey.py

# --- Data ---
generate:
	python scripts/run_generate.py

gemini:
	python scripts/run_generate_gemini.py

talkbank:
	python scripts/run_talkbank_analysis.py

# --- Pre-processing ---
prepare:
	python scripts/run_prepare.py

noise:
	python scripts/run_noise.py

# --- Training ---
train:
	python scripts/run_train.py

# --- Evaluation ---
evaluate:
	python scripts/run_evaluate.py

crf:
	python scripts/run_crf.py

calibration:
	python scripts/run_calibration.py

ablation:
	python scripts/run_ablation.py

diversity:
	python scripts/run_diversity.py

# --- Inference ---
inference:
	python scripts/run_inference.py --text "um hi my name is sarah chen and my order is ORD dash 2024 dash 5591"

api:
	python scripts/run_api.py

# --- Testing & Diagnostics ---
test:
	python -m pytest tests/ -v

eda:
	python scripts/run_eda.py

diagnose:
	python scripts/run_diagnose.py

gold:
	python scripts/offset_helper.py

# --- Full pipeline ---
all: generate prepare noise train evaluate
	@echo "Full pipeline completed."

clean:
	rm -rf results/models/
	rm -rf results/figures/*.png
	rm -rf results/eda/
	rm -f results/eval_report.md
	rm -f results/error_analysis.json
	@echo "Cleaned generated artifacts."
