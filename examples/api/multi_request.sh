#!/bin/bash

for i in `seq 1 10`;
do
curl -s -X POST -H "Content-Type: application/json" -d "$(echo -e "import numpy as np; print(np.random.standard_normal([1,16]).tolist())" | python)" http://localhost:5000/predict && echo "done"$i &
done

wait


# Polling
# curl -s -X POST -H "Content-Type: application/json" -d '"c6fa7396-ea9f-4df4-b989-84ddc57c7c3d"' http://localhost:5000/poll