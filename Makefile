build:
	docker-compose up -d --build

down:
	docker-compose down

start-ui:
	docker exec `docker ps | grep people-counter_people-counter | awk '{ print $$1 }'` ./scripts/start_ui.sh

download-models-host:
	cd ./people-counter && ./scripts/download_models.sh models.txt

downlad-models:
	docker exec `docker ps | grep people-counter_people-counter | awk '{ print $$1 }'` ./scripts/download_models_from_container.sh models.txt

run-default:
	echo "un the app on the demo video"
	docker exec `docker ps | grep people-counter_people-counter | awk '{ print $$1 }'` ./scripts/run.sh

run-model-1:
	echo "run the app on the demo video using pedestrian-detection-adas-0002 model"
	docker exec `docker ps | grep people-counter_people-counter | awk '{ print $$1 }'` ./scripts/run_model_1.sh

run-model-2:
	echo "run the app on the demo video using person-detection-0201 model"
	docker exec `docker ps | grep people-counter_people-counter | awk '{ print $$1 }'` ./scripts/run_model_2.sh

run-converted:
	echo "run the app on the demo video using converted ssd_mobilenet_v2_coco_2018_03_29"
	docker exec `docker ps | grep people-counter_people-counter | awk '{ print $$1 }'` ./scripts/run_converted_ssd_mobilenet_v2.sh