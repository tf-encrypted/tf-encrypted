#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
for i in `seq 1 10`;
do
  python $DIR/client.py --shape 1,16
done

wait

# Polling
# python client.py --poll y --request_id my-request-id