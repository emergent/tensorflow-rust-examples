#!/usr/bin/env bash

cd "$(dirname $0)" || exit
mkdir -p models
curl -L https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1?tf-hub-format=compressed | tar zx -C models
