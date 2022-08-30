build:
	docker-compose up -d --build

down:
	docker-compose down

start-ui:
	docker exec `docker ps | grep people-counter_people-counter | awk '{ print $$1 }'` ./scripts/start_ui.sh

download-models:
	cd ./people-counter && ./scripts/download_models.sh models.txt

run-default:
	echo "should run the app on the demo video"
	docker exec `docker ps | grep people-counter_people-counter | awk '{ print $$1 }'` ./scripts/run.sh

run-model-1:
	echo "should run the app on the demo video"
	docker exec `docker ps | grep people-counter_people-counter | awk '{ print $$1 }'` ./scripts/run_model_1.sh

run-model-2:
	echo "should run the app on the demo video"
	docker exec `docker ps | grep people-counter_people-counter | awk '{ print $$1 }'` ./scripts/run_model_2.sh
