import os
import pandas

from uuid import UUID, uuid4
import urllib2, io
from PIL import Image as PILImage
import numpy

from flask import Flask, request, session, redirect, url_for, render_template, flash
from flask_wtf import Form
from wtforms import SelectMultipleField, widgets, SelectField
from flask_wtf.file import FileField # FileAllowed, FileRequired
from wtforms.fields.html5 import URLField
from wtforms.validators import DataRequired, url
#from flask_wtf import RecaptchaField

#from htmlmin.minify import html_minify
from gevent.wsgi import WSGIServer
from gevent import monkey; monkey.patch_all()

DEBUG = True
SECRET_KEY = 'secret'

#RECAPTCHA_PUBLIC_KEY = '6LeYIbsSAAAAACRPIllxA7wvXjIE411PfdB2gt2J'
#RECAPTCHA_PRIVATE_KEY = '6LeYIbsSAAAAAJezaIq3Ft_hSTo0YtyeFG-JgRtu'

app = Flask(__name__)
app.config.from_object(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

run_inst = '(run/run_demo.sh MCR/v84/ static/{0}.png run/ 1 50 1 1 {1} {4} {2} {4} {3} {4} >> log/{0}.txt 2>&1 &)' # see run/src/demo.m for definitions
networks = [('1','Berkeley CaffeNet (ImageNet Challenge 2012 Winning Level) [1]'), ('2','Oxford CNN-S (ImageNet Challenge 2013 Winning Level) [2]'), ('3','Oxford VeryDeep-19 (ImageNet Challenge 2014 Winning Level) [3]')]

with open('synset_words.txt') as f:
	labels = pandas.DataFrame([
		{'synset_id': l.strip().split(' ')[0], 'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]} for l in f.readlines() ])

labels = labels['name'].values
#labels = labels.sort('synset_id')['name'].values
labels = [(str(i+1), '('+str(i+1).zfill(4)+') '+labels[i]) for i in xrange(labels.size)]

class MultiCheckboxField(SelectMultipleField):

	widget = widgets.ListWidget(html_tag='ul', prefix_label=False)
	option_widget = widgets.CheckboxInput()

class RunForm(Form):

	#tasks = ['Random Noise', 'Uploaded Image', 'Image at URL']
	tasks = ['Gray Image', 'Uploaded Image', 'Image at URL']
	select_network = MultiCheckboxField(choices=networks, default=['1'], validators=[DataRequired()])
	select_target = SelectField(choices=labels, default=['1'], validators=[DataRequired()])
	image_file = FileField()
	image_url = URLField(default='http://blogs.mathworks.com/images/loren/173/imdecompdemo_01.png')
	#recaptcha = RecaptchaField()

def valid_uuid(uuid):

    try:
        val = UUID(uuid, version=4)
    except ValueError:
        return False

    return val.hex == uuid

@app.route('/')
def index():
	
	req_taskid = request.args.get('taskid', '')
	if valid_uuid(req_taskid):
		session['taskid'] = req_taskid; # DELETE OLD TASK
		return redirect(url_for('index'))
	
	form = RunForm()
	session.permanent = True	
	
	advmode = taskid = taskpar = results = ori_class = out_class = '';
		
	if 'advmode' in session:
		advmode = session['advmode']
	
	if 'taskid' in session:
		taskid = session['taskid'];

		#may need to look-up job queue or retired id list later
		if os.path.isfile('log/' + taskid + '.txt'):
			taskpar = os.popen('head -n1 log/' + taskid + '.txt').read().replace('\n','')
			progress = os.popen('tail -n4 log/' + taskid + '.txt').read().split('\n')
		else:
			progress = []
		
		# RENDER RESULTS
		if len(progress) == 4 and progress[-1] == 'DONE':
			#os.path.isfile(taskid+'-out.png') and os.path.isfile(taskid+'-sal.png')
			if (progress[-2] == '1'):
				results = 'Your task finished sucessfully.'
				ori_class = '(' + ', '.join([i.zfill(4) for i in progress[0].split()]) + ')'
				out_class = '(' + ', '.join([i.zfill(4) for i in progress[1].split()]) + ')'
			elif (progress[-2] == '0'):
				results = 'Your task didn\'t finish in time limit, and here are the best results we got.'
				ori_class = '(' + ', '.join([i.zfill(4) for i in progress[0].split()]) + ')'
				out_class = '(' + ', '.join([i.zfill(4) for i in progress[1].split()]) + ')'
			else: #'-1', error
				results = 'Something went wrong and we will look at it. You can come back later and resubmit your task, thanks!'
				# SET ERROR IMAGE?
	
	return render_template("index.html", form=form, advmode=advmode, taskid=taskid, taskpar=taskpar, results=results, ori_class=ori_class, out_class=out_class)

@app.route('/run_task', methods=['POST'])
def run_task():

	form = RunForm()

	if not form.validate_on_submit():
		flash('Incorrect Form!')
		return redirect(url_for('index'))

	try:
		if request.form['start'] == form.tasks[0]:
			#run_image = numpy.random.randn(227,227,3)
			run_image = numpy.zeros((227,227,3))
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
		taskid = str(uuid4()).replace('-','') # DELETE OLD TASK
		run_image.filename = taskid + '.png'
		run_image.save('static/' + run_image.filename)
		session['taskid'] = taskid
	except:
		flash('Image Processing Error!')
		return redirect(url_for('index'))
	
	try:
		with open('log/' + taskid + '.txt', 'w') as f:
			f.write('Algorithm {0} and Class {1}\n\n'.format('['+', '.join(form.select_network.data)+']', labels[int(form.select_target.data) - 1][1]))
		# RUN NONBLOCKING TASK
		select_network = [str(int(str(i+1) in form.select_network.data)) for i in xrange(len(networks))]
		os.system(run_inst.format(session['taskid'], *(select_network + [str(form.select_target.data)])))
	except:
		flash('Task Scheduling Error!')
		return redirect(url_for('index'))

	return redirect(url_for('index'))

@app.route('/del_task', methods=['POST'])
def del_task():
	
	session.pop('taskid', None) # NEED TO REALLY DELETE OLD TASK
	return redirect(url_for('index'))

# trailing slash here
@app.route('/adv/') 
def adv_mode(): 
	
	if 'advmode' in session:
		session.pop('advmode', None)
	else:
		session['advmode'] = '1'
	
	return redirect(url_for('index'))

# (python app.py &>> log.txt 2>&1 &)
if __name__ == "__main__":
	#app.run()
	http_server = WSGIServer(('0.0.0.0',5900), app); http_server.serve_forever()
	
