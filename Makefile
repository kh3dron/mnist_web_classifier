.PHONY: run-local run-local-blind

help:
	@echo ''
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Default:'
	@echo '  make run-local'
	@echo ''
	@echo 'Targets:'
	@echo '  run-local: runs with new relic observability agent'
	@echo '  run-local-blind: runs without new relic'

run-local:
	NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program uvicorn app.main:app --reload

run-local-blind:
	uvicorn app.main:app --reload
