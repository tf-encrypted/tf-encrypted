python servers.py server0 &
A=$!
python servers.py server1 &
B=$!
python servers.py crypto_producer &
C=$!
python servers.py weights_provider &
D=$!
python servers.py prediction_client &
E=$!
echo "TO STOP THE CLUSTER: kill $A $B $C $D $E"