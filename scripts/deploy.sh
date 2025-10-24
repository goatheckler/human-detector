#!/bin/bash

set -e

VERSION="${1:-latest}"

echo "Deploying human-detector stack with version: $VERSION"

VERSION="$VERSION" docker stack deploy -c docker-stack.yml human-detector

echo ""
echo "Deployment initiated. Waiting for services to start..."
sleep 5

echo ""
echo "Stack services:"
docker stack services human-detector

echo ""
echo "To check service logs:"
echo "  docker service logs human-detector_backend"
echo "  docker service logs human-detector_frontend"
