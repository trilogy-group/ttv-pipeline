#!/usr/bin/env bash
# deploy.sh: build and push x86_64 Docker image for FramePack
set -euo pipefail

IMAGE_NAME="framepack-demo"
REGISTRY="cr.eu-north1.nebius.cloud/e00kwmrf0879q1806h"
TAG="latest"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "Building ${IMAGE_NAME} for x86_64..."
docker build --platform linux/amd64 -t "${IMAGE_NAME}" .

echo "Tagging ${IMAGE_NAME} -> ${FULL_IMAGE}..."
docker tag "${IMAGE_NAME}" "${FULL_IMAGE}"

echo "Pushing ${FULL_IMAGE}..."
docker push "${FULL_IMAGE}"

echo "Deployment complete."
