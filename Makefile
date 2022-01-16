CONTAINER_NAME_BASE=jupyter_base
CONTAINER_NAME=jupyter
IMAGE_NAME=$(CONTAINER_NAME)

install-docker: ## install docker
	curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
	sh /tmp/get-docker.sh

run-base-image-only: ## run tf jupyter container only
	docker run --gpus all -v $(HOME)/:/tf/home -v /var/run/docker.sock:/var/run/docker.sock -p 8888:8888 -it --rm --name $(CONTAINER_NAME_BASE) tensorflow/tensorflow:latest-gpu-jupyter

build: ## build image
	docker build -t $(IMAGE_NAME) .

run: build ## run container, and build if needed
	docker run --gpus all -v $(HOME)/:tf/home -v /var/run/docker.sock:/var/run/docker.sock -p 8888:8888 -it --rm --name $(CONTAINER_NAME) $(IMAGE_NAME)

bash: ## get bash prompt inside the container
	docker exec -it $(CONTAINER_NAME) "/bin/bash"