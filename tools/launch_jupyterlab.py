import subprocess
from jupyter_server.auth import passwd

subprocess.run(['jupyter',
                'lab',
                '--ip=0.0.0.0',
                '--port=8888',
                f'--ServerApp.password={passwd()}',
                '--ServerApp.token=""',
                '--no-browser'
                ], shell=False)