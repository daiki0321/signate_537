TARGET ?= amd64

ifeq ("$(TARGET"),"arm64")
    WAMR_ARCH = AARCH64
else
    WAMR_ARCH = X86_64
endif

EXTRA_CFLAGS =
EXTRA_LIBS =

.PHONY: all
all: wasm_module wamr_main copy_build

.PHONY: wamr_main
wamr_main:
	docker buildx build --load --platform linux/$(TARGET) --build-arg ARCH=$(WAMR_ARCH) -t wamr_$(TARGET):latest .

.PHONY: wasm_module
wasm_module:
	docker build -t yolo_wasm:latest -f Dockerfile.wasm_module .

.PHONY: clean
clean:
	rm -rf output

define copy_out_from_docker
if [ ! -d $(3) ]; then mkdir -p $(3); fi
$(eval CONTAINER_ID := $(shell docker create $(1)))
docker cp $(CONTAINER_ID):$(2) $(3)
docker rm $(CONTAINER_ID)
endef

.PHONY: copy_build
copy_build:
	$(call copy_out_from_docker, yolo_wasm:latest,/root/src/wasm_module/tracking.wasm, ./output/$(TARGET))
	$(call copy_out_from_docker, wamr_$(TARGET):latest,/root/src/main/yolo_main, ./output/$(TARGET))
	$(call copy_out_from_docker, wamr_$(TARGET):latest,/root/src/main/yolov3-tiny.cfg, ./output/$(TARGET))
	$(call copy_out_from_docker, wamr_$(TARGET):latest,/root/src/main/yolov3-tiny.weights, ./output/$(TARGET))
	$(call copy_out_from_docker, wamr_$(TARGET):latest,/root/src/main/coco.data, ./output/$(TARGET))
	$(call copy_out_from_docker, wamr_$(TARGET):latest,/root/src/main/coco.names, ./output/$(TARGET))
	$(call copy_out_from_docker, wamr_$(TARGET):latest,/root/src/main/dog.jpg, ./output/$(TARGET))
	$(call copy_out_from_docker, wamr_$(TARGET):latest,/root/src/main/data, ./output/$(TARGET))
