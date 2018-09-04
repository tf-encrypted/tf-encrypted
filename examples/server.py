import sys
import tensorflow_encrypted as tfe

player_name = str(sys.argv[1])

config = tfe.RemoteConfig({
    'server0': '0.0.0.0:4440',
    'server1': '0.0.0.0:4441',
    'crypto_producer': '0.0.0.0:4442'
})

server = config.server(player_name)
server.start()
server.join()
