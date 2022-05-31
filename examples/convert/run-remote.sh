SCRIPT_PATH="${BASH_SOURCE:-$0}"
ABS_DIRECTORY="$(dirname "${SCRIPT_PATH}")"

echo "Starting server0"
(python -m tf_encrypted.player server0 --config ${ABS_DIRECTORY}/config.json > log0.txt 2>&1 &)
sleep 1
echo "Starting server1"
(python -m tf_encrypted.player server1 --config ${ABS_DIRECTORY}/config.json > log1.txt 2>&1 &)
sleep 1
echo "Starting server2"
(python -m tf_encrypted.player server2 --config ${ABS_DIRECTORY}/config.json > log2.txt 2>&1 &)
sleep 1
echo "Starting weights-provider"
(python -m tf_encrypted.player weights-provider --config ${ABS_DIRECTORY}/config.json > log3.txt 2>&1 &)
sleep 1
echo "Starting prediction-client"
(python -m tf_encrypted.player prediction-client --config ${ABS_DIRECTORY}/config.json > log4.txt 2>&1 &)
sleep 1

echo "Run private model conversion and prediction example..."
(time python ${ABS_DIRECTORY}/convert2.py ${ABS_DIRECTORY}/config.json > log.txt 2>&1 &)
