# Ostrichinator
Ostrichinator aims at providing an integrated platform for generating adversarial examples and potentially collecting  human responses, which may fundamentally help with improving current deep learning algorithms. The frontend of this framework is based on Python Flask, which handles requests from users and passes jobs to the backend, and the backend is implemented in MATLAB based on MatConvNet and minConf.

## Frontend
### General Introduction
Like most of the Flask based simple web applications, the frontend of Ostrichinator is concisely implemented in two files: <code>frontend.py</code> and <code>template/index.html</code>.
User requests are indexed by UUIDs and the source (original images) and resultant files (hacked images and logs) are named correspondingly.
While source and resultant images are directly served in <code>static/</code>, logs are placed in <code>backend/log/</code>.
### Configuration
Except for the secret keys for Flask and Flask-WTF reCAPTCHA (i.e. <code>SECRET_KEY</code>, <code>RECAPTCHA_PUBLIC_KEY</code> and <code>RECAPTCHA_PRIVATE_KEY</code>, which in our case, are put in <code>keys.py</code>), to run the backend locally or distributedly (i.e. without or with a job queue) should be the only thing which needs to be configured.
<code>Frontend.py</code> by default is using a Celery plus Redis job queue for requests.
However, users can easily comment/uncomment blocks in <code>frontend.py</code> to make it run locally following commented instructions.
### Deployment
Simply executing <code>python frontend.py</code> should be able to make the Flask frontend up and running with the gvent WSGI server.
However, other combinations (e.g. Flask with uWSGI and Nginx, as used by the demo site) can be more favorable.

## Backend
### General Introduction

### Configuration
Configuring the backend of Ostrichinator would involve compiling MATLAB codes located in <code>backend/src/</code> into an executable.
We used MATLAB R2014b, while any MATLAB version above R2013a should work correctly as well.
First thing the users need to do would be installing and setting up [MatConvNet](http://www.vlfeat.org/matconvnet/), which is fairly simple and quick.
While MatConvNet v1.0 beta-7 is used and included in this project, newer versions should work as well.
The main MATLAB file is <code>backend/src/demo.m</code>, which should be ready to run after MatConvNet is correctly installed, and both MatConvNet and <code>backend/src/minConf</code> are in your MATLAB’s path setting.
Please follow <code>demo.m</code> to compile the executable, and place the generated <code>demo</code> and <code>run_demo.sh</code> under <code>backend/</code>.
If installing MATLAB is not an option, users can check out our [precompiled executables]() as well.
After obtaining the executables, users also need to specify the path to MATLAB runtime libraries in <code>backend/MCR/</code>.
If users have a full MATLAB installation, a symbolic link, e.g. <code>backend/MCR/v84 -> /usr/local/MATLAB/R2014b/</code>, can be created to do this.
If users don’t have MATALB and decide to use our precompiled executables, please download and install the [MATLAB Compiler Runtime](http://www.mathworks.com/products/compiler/mcr/) into <code>backend/MCR/</code>.
Remember to download the pretrained deep learning networks [[1](http://www.vlfeat.org/matconvnet/models/imagenet-caffe-ref.mat),[2](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-s.mat),[3](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)] into <code>backend/networks/</code> if users haven’t.
Lastly, adjust <code>backend/run.py</code> and <code>backend/info.py</code> for running locally or distributedly as well.
### Deployment
If users decided to run this project distributedly, simply remember to start Redis and Celery by e.g. <code>redis-server</code> and <code>celery worker -A backend.run --loglevel=info</code> (for Celery, under the main directory, not backend/).
Otherwise, nothing needs to be done.
