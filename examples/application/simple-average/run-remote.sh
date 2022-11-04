SCRIPT_PATH="${BASH_SOURCE:-$0}"
ABS_DIRECTORY="$(dirname "${SCRIPT_PATH}")"

kill -9 `ps -ef | grep tf_encrypted.player | grep -v grep | awk '{print $2}'`
kill -9 `ps -ef | grep ${ABS_DIRECTORY}/run.py | grep -v grep | awk '{print $2}'`

echo "Starting server0"
(python -m tf_encrypted.player server0 --config ${ABS_DIRECTORY}/config.json > log_server_0.txt 2>&1 &)
sleep 1
echo "Starting server1"
(python -m tf_encrypted.player server1 --config ${ABS_DIRECTORY}/config.json > log_server_1.txt 2>&1 &)
sleep 1
echo "Starting server2"
(python -m tf_encrypted.player server2 --config ${ABS_DIRECTORY}/config.json > log_server_2.txt 2>&1 &)
sleep 1
echo "Starting inputter-0"
(python -m tf_encrypted.player inputter-0 --config ${ABS_DIRECTORY}/config.json > log_inputer_0.txt 2>&1 &)
sleep 1
echo "Starting inputter-1"
(python -m tf_encrypted.player inputter-1 --config ${ABS_DIRECTORY}/config.json > log_inputer_1.txt 2>&1 &)
sleep 1
echo "Starting inputter-2"
(python -m tf_encrypted.player inputter-2 --config ${ABS_DIRECTORY}/config.json > log_inputer_2.txt 2>&1 &)
sleep 1
echo "Starting inputter-3"
(python -m tf_encrypted.player inputter-3 --config ${ABS_DIRECTORY}/config.json > log_inputer_3.txt 2>&1 &)
sleep 1
echo "Starting inputter-4"
(python -m tf_encrypted.player inputter-4 --config ${ABS_DIRECTORY}/config.json > log_inputer_4.txt 2>&1 &)
sleep 1
echo "Starting result-receiver"
(python -m tf_encrypted.player result-receiver --config ${ABS_DIRECTORY}/config.json > log_receiver.txt 2>&1 &)
sleep 1

echo "Run private average example..."
(time python ${ABS_DIRECTORY}/run.py $* > log_master.txt 2>&1 &)
