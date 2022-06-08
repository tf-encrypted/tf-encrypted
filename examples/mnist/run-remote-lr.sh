SCRIPT_PATH="${BASH_SOURCE:-$0}"
ABS_DIRECTORY="$(dirname "${SCRIPT_PATH}")"

kill -9 `ps -ef | grep ${ABS_DIRECTORY}/config.json | grep -v grep | awk '{print $2}'`

echo "Starting server0"
(python -m tf_encrypted.player server0 --config ${ABS_DIRECTORY}/config.json > log0.txt 2>&1 &)
sleep 1
echo "Starting server1"
(python -m tf_encrypted.player server1 --config ${ABS_DIRECTORY}/config.json > log1.txt 2>&1 &)
sleep 1
echo "Starting server2"
(python -m tf_encrypted.player server2 --config ${ABS_DIRECTORY}/config.json > log2.txt 2>&1 &)
sleep 1
echo "Starting training-client"
(python -m tf_encrypted.player training-client --config ${ABS_DIRECTORY}/config.json > log3.txt 2>&1 &)
sleep 1
echo "Starting prediction-client"
(python -m tf_encrypted.player prediction-client --config ${ABS_DIRECTORY}/config.json > log4.txt 2>&1 &)
sleep 1

echo "Run private training example..."
(time python ${ABS_DIRECTORY}/private_lr_training.py ${ABS_DIRECTORY}/config.json > log.txt 2>&1 &)
