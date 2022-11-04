SCRIPT_PATH="${BASH_SOURCE:-$0}"
ABS_DIRECTORY="$(dirname "${SCRIPT_PATH}")"

kill -9 `ps -ef | grep tf_encrypted.player | grep -v grep | awk '{print $2}'`
kill -9 `ps -ef | grep ${ABS_DIRECTORY}/convert.py | grep -v grep | awk '{print $2}'`

echo "Starting server0"
(python -m tf_encrypted.player server0 --config ${ABS_DIRECTORY}/config.json > log_server_0.txt 2>&1 &)
sleep 1
echo "Starting server1"
(python -m tf_encrypted.player server1 --config ${ABS_DIRECTORY}/config.json > log_server_1.txt 2>&1 &)
sleep 1
echo "Starting server2"
(python -m tf_encrypted.player server2 --config ${ABS_DIRECTORY}/config.json > log_server_2.txt 2>&1 &)
sleep 1
echo "Starting weights-provider"
(python -m tf_encrypted.player weights-provider --config ${ABS_DIRECTORY}/config.json > log_weights_provider.txt 2>&1 &)
sleep 1
echo "Starting prediction-client"
(python -m tf_encrypted.player prediction-client --config ${ABS_DIRECTORY}/config.json > log_predict_client.txt 2>&1 &)
sleep 1

echo "Run private inference example..."
(time python ${ABS_DIRECTORY}/convert.py $* > log_master.txt 2>&1 &)