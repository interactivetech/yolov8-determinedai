DOCKER_IMAGE_NAME=mendeza/yolov8det:0.0.3
echo "Building $DOCKER_IMAGE_NAME..."
docker build . -t $DOCKER_IMAGE_NAME --no-cache && docker push $DOCKER_IMAGE_NAME
echo "Done!"