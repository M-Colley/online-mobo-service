# Common dev/deploy tasks for the optimizer service.  Usage: make <target>
# Deploy needs PROJECT (your GCP project id):  make deploy PROJECT=my-project
REGION  ?= us-central1
PROJECT ?=
SERVICE ?= vam-optimizer
PY      ?= python3
MODULES  = vam_space.py optimizer_core.py main.py simulate.py test_service.py test_vam_space.py inspect_db.py

.PHONY: help describe unit test inspect deploy

help:
	@echo "Targets: describe | unit | test | inspect | deploy"
	@echo "deploy needs PROJECT=<gcp-project-id> (REGION defaults to $(REGION))"

describe:            ## print the resolved JND grids
	$(PY) vam_space.py

unit:               ## fast search-space unit tests (sub-second)
	$(PY) test_vam_space.py

test:               ## compile + unit + simulate + full service test
	$(PY) -m py_compile $(MODULES)
	$(PY) test_vam_space.py
	$(PY) simulate.py
	$(PY) test_service.py

inspect:            ## read-only Firestore inspector (needs auth + GOOGLE_CLOUD_PROJECT)
	$(PY) inspect_db.py

deploy:             ## gcloud run deploy to $(PROJECT) / $(REGION)
	@test -n "$(PROJECT)" || { echo "Set PROJECT, e.g.: make deploy PROJECT=my-gcp-project"; exit 1; }
	gcloud run deploy $(SERVICE) --source . --project $(PROJECT) \
	  --region $(REGION) --allow-unauthenticated \
	  --memory 2Gi --cpu 2 --timeout 300
