echo "Starting server0"
(python -m tf_encrypted.player server0 --config examples/mnist/config.json > log0.txt 2>&1 &)
sleep 1
echo "Starting server1"
(python -m tf_encrypted.player server1 --config examples/mnist/config.json > log1.txt 2>&1 &)
sleep 1
echo "Starting server2"
(python -m tf_encrypted.player server2 --config examples/mnist/config.json > log2.txt 2>&1 &)
sleep 1
echo "Starting training-client"
(python -m tf_encrypted.player training-client --config examples/mnist/config.json > log3.txt 2>&1 &)
sleep 1
echo "Starting prediction-client"
(python -m tf_encrypted.player prediction-client --config examples/mnist/config.json > log4.txt 2>&1 &)
sleep 1

echo "Run private training example..."
(time python examples/mnist/private_network_training.py examples/mnist/config.json > log.txt 2>&1 &)
