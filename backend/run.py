import os
from celery import Celery
from keys import UPLOAD_PATH

# see src/demo.m for definitions
#run_inst = '(run/run_demo.sh MCR/v84/ static/{0}.png run/ 0 50 1 1 {1} {4} {2} {4} {3} {4} >> log/{0}.txt 2>&1 &)'
run_inst = 'backend/run_demo.sh backend/MCR/v84/ static/{0}.png backend/networks/ 0 50 1 1 {1} {4} {2} {4} {3} {4} >> backend/log/{0}.txt 2>&1'

backend = Celery('run', broker='redis://localhost:6379/1'); backend.conf.update(CELERYD_PREFETCH_MULTIPLIER=1)

@backend.task
def run_backend(taskpar):
	os.system('wget -q -P static http://localhost:8080/static/{0}.png'.format(taskpar[0]))
	os.system(run_inst.format(*taskpar))
	os.system('curl -s -X POST -F data=@static/{0}-out.png "http://localhost:8080/{1}"'.format(taskpar[0],UPLOAD_PATH))
	os.system('curl -s -X POST -F data=@static/{0}-sal.png "http://localhost:8080/{1}"'.format(taskpar[0],UPLOAD_PATH))
	os.system('curl -s -X POST -F data=@static/{0}-dff.png "http://localhost:8080/{1}"'.format(taskpar[0],UPLOAD_PATH))
	os.system('curl -s -X POST -F data=@backend/log/{0}.txt "http://localhost:8080/{1}"'.format(taskpar[0],UPLOAD_PATH))
	return None
