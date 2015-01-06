import os
import pandas

from uuid import UUID, uuid4
import urllib2, io
from PIL import Image as PILImage
import numpy

from flask import Flask, request, session, redirect, url_for, render_template, flash
from flask_wtf import Form, RecaptchaField
from wtforms import SelectMultipleField, widgets, SelectField
from flask_wtf.file import FileField
from wtforms.fields.html5 import URLField
from wtforms.validators import DataRequired, url
from keys import *

from gevent.wsgi import WSGIServer
from gevent import monkey; monkey.patch_all()

import redis; client = redis.Redis(host="localhost", port=6379, db=1)
from backend.run import run_backend
import backend.info

app = Flask(__name__)
app.config.from_object(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

networks = [('1','Berkeley CaffeNet (ImageNet challenge 2012 winning level) [<a href="http://arxiv.org/abs/1408.5093">1</a>]'), 
            ('2','Oxford CNN-S (ImageNet challenge 2013 winning level) [<a href="http://arxiv.org/abs/1405.3531">2</a>]'), 
            ('3','Oxford VeryDeep-19 (ImageNet challenge 2014 winning level) [<a href="http://arxiv.org/abs/1409.1556">3</a>]')]

with open('synset_words.txt') as f:
	labels = pandas.DataFrame([
		{'synset_id': l.strip().split(' ')[0], 'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]} for l in f.readlines() ])

labels = labels['name'].values
labels = [(str(i+1), '('+str(i+1).zfill(4)+') '+labels[i]) for i in xrange(labels.size)]

class MultiCheckboxField(SelectMultipleField):

	widget = widgets.ListWidget(html_tag='ul', prefix_label=False)
	option_widget = widgets.CheckboxInput()

class TaskForm(Form):

	tasks = ['Random Noise', 'Uploading Image', 'Image at URL']
	network_selection = MultiCheckboxField(choices=networks, default=['1'], validators=[DataRequired()])
	label_selection = SelectField(choices=labels, default=['1'], validators=[DataRequired()])
	image_file = FileField()
	image_url = URLField(default='http://blogs.mathworks.com/images/loren/173/imdecompdemo_01.png')
	recaptcha = RecaptchaField()

def valid_uuid(uuid):

	try: val = UUID(uuid, version=4)
	except ValueError: return False
	return val.hex == uuid

def get_srv_load():
	
	def get_worker_num():
		#reload(backend.info)
		return backend.info.WORKER_NUM
	
	try: srv_load = 100 * client.llen("celery") / get_worker_num()
	except: srv_load = 100
	return srv_load

@app.route('/')
def index():

	req_taskid = request.args.get('taskid', '')
	if valid_uuid(req_taskid) or (req_taskid.find('example') == 0):
		session['taskid'] = req_taskid # DELETE OLD TASK?
		return redirect(url_for('index'))
	
	form = TaskForm()
	session.permanent = True
	
	advmode = ''; 
	task_info = {'taskid':'','taskpar':'','results':'','ori_class':'','new_class':''}	
	
	if 'advmode' in session: advmode = session['advmode']
	if 'taskid'  in session: task_info['taskid'] = session['taskid']
	
	if task_info['taskid']: # and os.path.isfile('static/' + taskid + '.png'):

		#may need to look-up job queue or retired id list later
		if os.path.isfile('log/' + task_info['taskid'] + '.txt'):
			progress = os.popen('tail -n4 log/' + task_info['taskid'] + '.txt').read().split('\n')
			task_info['taskpar'] = os.popen('head -n1 log/' + task_info['taskid'] + '.txt').read().replace('\n','')
		else:
			progress = []
		
		# RENDER RESULTS
		if len(progress) == 4 and progress[-1] == 'DONE':
			#os.path.isfile(taskid+'-out.png') and os.path.isfile(taskid+'-sal.png')
			if (progress[-2] == '1'):
				task_info['results'] = 'Your task finished sucessfully.'
				task_info['ori_class'] = '(' + ', '.join([i.zfill(4) for i in progress[0].split()]) + ')'
				task_info['new_class'] = '(' + ', '.join([i.zfill(4) for i in progress[1].split()]) + ')'
			elif (progress[-2] == '0'):
				task_info['results'] = 'Your task didn\'t finish in time limit, and here are the best results we got.'
				task_info['ori_class'] = '(' + ', '.join([i.zfill(4) for i in progress[0].split()]) + ')'
				task_info['new_class'] = '(' + ', '.join([i.zfill(4) for i in progress[1].split()]) + ')'
			else: #'-1', error # DISPLAY ERROR IMAGE?
				task_info['results'] = 'Something went wrong and we will look at it. You can come back later and resubmit your task, thanks!'
	#else: taskid = ''
	
	srv_load = '%.2f' % get_srv_load()
	return render_template("index.html", form=form, advmode=advmode, task_info=task_info, srv_load=srv_load)

@app.route('/run_task', methods=['POST'])
def run_task():

	form = TaskForm()

	if not form.validate_on_submit():
		flash('Incorrect Form!')
		return redirect(url_for('index'))

	try:
		if request.form['start'] == form.tasks[0]:
			run_image = numpy.random.randn(227,227,3)
			#run_image = numpy.zeros((227,227,3))
			run_image = (run_image*16/256 + 0.5).clip(0,1)
			run_image = PILImage.fromarray((run_image*255).astype('uint8'))
		elif request.form['start'] == form.tasks[1]:
			run_image = PILImage.open(form.image_file.data)
		elif request.form['start'] == form.tasks[2]:
			run_image = urllib2.urlopen(form.image_url.data, timeout=2) # TIMEOUT FOR SAFETY
			if int(run_image.info().getheaders("Content-Length")[0]) > app.config['MAX_CONTENT_LENGTH']:
				flash('URL File Size over Limit!')
				raise Exception
			run_image = PILImage.open(io.BytesIO(run_image.read()))
		else:
			raise Exception
	except:
			flash('Image Loading Error!')
			return redirect(url_for('index'))

	try:
		run_image = run_image.convert('RGB').resize((227,227),PILImage.ANTIALIAS)
		taskid = str(uuid4()).replace('-','') # DELETE OLD TASK?
		run_image.filename = taskid + '.png'
		#run_image.save('static/' + run_image.filename) # MOVED LATER
		session['taskid'] = taskid
	except:
		flash('Image Processing Error!')
		return redirect(url_for('index'))
	
	try:
		if (get_srv_load() >= 100.0): session.pop('taskid', None); raise Exception

		with open('log/' + taskid + '.txt', 'w') as f:
			f.write('Algorithm {0} and Class {1}\n'.format('['+', '.join(form.network_selection.data)+']', labels[int(form.label_selection.data) - 1][1]))
				
		network_selection = [str(int(str(i+1) in form.network_selection.data)) for i in xrange(len(networks))]
		run_image.save('static/' + run_image.filename) # SAVE WHEN SCHEDULED
		
		# RUN NONBLOCKING TASK
		run_backend.delay([session['taskid']] + network_selection + [str(form.label_selection.data)])
	except:
		flash('Task Initialization/Scheduling Error!')
		return redirect(url_for('index'))

	return redirect(url_for('index'))

@app.route('/del_task', methods=['POST'])
def del_task():
	
	session.pop('taskid', None) # DELETE OLD TASK?
	return redirect(url_for('index'))


# only for distributed jobs
@app.route('/{0}'.format(UPLOAD_PATH), methods=['POST'])
def upload_result():
	
	data = request.files['data']
	if os.path.splitext(data.filename)[1] == '.png':
		data.save('static/' + data.filename)
	elif os.path.splitext(data.filename)[1] == '.txt':
		with open('log/' + data.filename, 'a') as logfile: logfile.write(data.stream.read())
	return ''

# trailing slash here
@app.route('/adv/') 
def adv_mode(): 
	
	if 'advmode' in session: session.pop('advmode', None)
	else: session['advmode'] = '1'
	
	return redirect(url_for('index'))

# (python app.py &>> app.log 2>&1 &)
if __name__ == "__main__":
	#app.run(host='0.0.0.0', port=8080, debug = True)
	http_server = WSGIServer(('0.0.0.0',8080), app); http_server.serve_forever()
	
