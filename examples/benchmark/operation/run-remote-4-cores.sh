# Usage: ./run-remote.sh test_sort_performance 

SCRIPT_PATH="${BASH_SOURCE:-$0}"
ABS_DIRECTORY="$(dirname "${SCRIPT_PATH}")"

kill -9 `ps -ef | grep tf_encrypted.player | grep -v grep | awk '{print $2}'`
kill -9 `ps -ef | grep ${ABS_DIRECTORY}/op_profile.py | grep -v grep | awk '{print $2}'`

echo "Starting server0"
(taskset -c 0,1,2,3 python -m tf_encrypted.player server0 --config ${ABS_DIRECTORY}/config.json > log_server_0.txt 2>&1 &)
sleep 1
echo "Starting server1"
(taskset -c 4,5,6,7 python -m tf_encrypted.player server1 --config ${ABS_DIRECTORY}/config.json > log_server_1.txt 2>&1 &)
sleep 1
echo "Starting server2"
(taskset -c 8,9,10,11 python -m tf_encrypted.player server2 --config ${ABS_DIRECTORY}/config.json > log_server_2.txt 2>&1 &)
sleep 1

echo "Run operation profiling of function: $1"
(time taskset -c 12,13,14,15 python ${ABS_DIRECTORY}/op_profile.py $* > log_master.txt 2>&1 &)

