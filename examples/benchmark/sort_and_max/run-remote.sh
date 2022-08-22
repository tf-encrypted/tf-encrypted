# Usage: ./run-remote.sh test_sort_performance 

SCRIPT_PATH="${BASH_SOURCE:-$0}"
ABS_DIRECTORY="$(dirname "${SCRIPT_PATH}")"

kill -9 `ps -ef | grep tf_encrypted.player | grep -v grep | awk '{print $2}'`
kill -9 `ps -ef | grep ${ABS_DIRECTORY}/aby3_profile.py | grep -v grep | awk '{print $2}'`

echo "Starting server0"
(python -m tf_encrypted.player server0 --config ${ABS_DIRECTORY}/config.json > log0.txt 2>&1 &)
sleep 1
echo "Starting server1"
(python -m tf_encrypted.player server1 --config ${ABS_DIRECTORY}/config.json > log1.txt 2>&1 &)
sleep 1
echo "Starting server2"
(python -m tf_encrypted.player server2 --config ${ABS_DIRECTORY}/config.json > log2.txt 2>&1 &)
sleep 1

echo "Run ABY3 profiling of function: $1"
(time python ${ABS_DIRECTORY}/aby3_profile.py ${ABS_DIRECTORY}/config.json $1 > log.txt 2>&1 &)

