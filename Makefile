.PHONY: install data train monitor serve all test clean

install:
	pip install -r requirements.txt && pip install -e .

data:
	python -m credit_risk.cli generate

train:
	python -m credit_risk.cli train

monitor:
	python -m credit_risk.cli monitor

serve:
	uvicorn credit_risk.serving.api:app --reload --port 8000

all: data train monitor

test:
	pytest -q

clean:
	rm -rf data/raw model_registry/credit_pd reports/*.json reports/*.md
