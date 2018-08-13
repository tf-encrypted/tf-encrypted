import sys
import tensorflow_encrypted as tfe

task_index = int(sys.argv[1])

config = tfe.RemoteConfig(
    player_hosts=[
        'localhost:4440',
        'localhost:4441',
        'localhost:4442'
    ]
)

server = config.server(task_index)
server.start()
server.join()