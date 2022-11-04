SCRIPT_PATH="${BASH_SOURCE:-$0}"
ABS_DIRECTORY="$(dirname "${SCRIPT_PATH}")"

kill -9 `ps -ef | grep tf_encrypted.player | grep -v grep | awk '{print $2}'`
kill -9 `ps -ef | grep ${ABS_DIRECTORY}/vertical-training.py | grep -v grep | awk '{print $2}'`

echo "Starting train-data-owner-0"
(python -m tf_encrypted.player train-data-owner-0 --config ${ABS_DIRECTORY}/config.json > log_train_data_owner_0.txt 2>&1 &)
sleep 1
echo "Starting train-data-owner-1"
(python -m tf_encrypted.player train-data-owner-1 --config ${ABS_DIRECTORY}/config.json > log_train_data_owner_1.txt 2>&1 &)
sleep 1
echo "Starting train-data-owner-2"
(python -m tf_encrypted.player train-data-owner-2 --config ${ABS_DIRECTORY}/config.json > log_train_data_owner_2.txt 2>&1 &)
sleep 1
echo "Starting test-data-owner"
(python -m tf_encrypted.player test-data-owner --config ${ABS_DIRECTORY}/config.json > log_test_data_owner.txt 2>&1 &)
sleep 1
echo "Starting server0"
(python -m tf_encrypted.player server0 --config ${ABS_DIRECTORY}/config.json > log_server_0.txt 2>&1 &)
sleep 1
echo "Starting server1"
(python -m tf_encrypted.player server1 --config ${ABS_DIRECTORY}/config.json > log_server_1.txt 2>&1 &)
sleep 1
echo "Starting server2"
(python -m tf_encrypted.player server2 --config ${ABS_DIRECTORY}/config.json > log_server_2.txt 2>&1 &)
sleep 1

echo "Run vertically splited data training example..."
(time python ${ABS_DIRECTORY}/vertical-training.py $* > log_master.txt 2>&1 &)
