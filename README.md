# Ostrichinator
Ostrichinator aims at providing a [web platform](http://ostrichinator.net/) for discovering adversarial examples and potentially collecting human responses, which may fundamentally help with improving current deep learning algorithms.
The frontend of this project is based on Python Flask, which handles requests from users and passes jobs to the backend, and the backend is implemented in MATLAB based on MatConvNet and minConf.

## Frontend
Like most other Flask based simple web applications, the frontend of Ostrichinator is concisely implemented in two files: <code>frontend.py</code> and <code>template/index.html</code>.
User requests are indexed by UUIDs and the source (original images) and result files (hacked images and logs) are named correspondingly.
While source and resultant images are directly served in <code>static/</code>, logs are placed in <code>backend/log/</code>.
### Configuration
Except for the secret keys for Flask and Flask-WTF reCAPTCHA (i.e. <code>SECRET_KEY</code>, <code>RECAPTCHA_PUBLIC_KEY</code> and <code>RECAPTCHA_PRIVATE_KEY</code>, which in our case, are put in <code>keys.py</code>), to run the backend locally or distributedly (i.e. without or with a job queue) should be the only thing which needs to be configured here.
<code>Frontend.py</code> by default is using a Celery plus Redis job queue for requests.
However, users can easily comment/uncomment blocks in <code>frontend.py</code> to make it run locally following instructions in the file.
### Deployment
Simply executing <code>python frontend.py</code> can launch the Flask frontend with gevent WSGI server.
However, other combinations (e.g. Flask with uWSGI and Nginx, as used by the demo site) may be more favorable.

## Backend
The backend of Ostrichinator is a compiled executable which directly takes input images from the <code>static/</code> directory and parameters from the command line arguments.
When the backend is running, log files stating the execution progresses and the final results are generated inside <code>backend/log/</code>, and when it finishes, result images are written into the <code>static/</code> directory as well.
The log files are structurally defined with the first lines describing the tasks, the fourth- and third-to-the-last lines the original and final class labels, the second-to-the-last lines the exit flags, and the final lines "DONE".
For now, there's no explicit mechanism for pushing results to the users implemented yet.
### Configuration
Configuring the backend of Ostrichinator would involve compiling MATLAB codes located in <code>backend/src/</code> into an executable.
We used MATLAB R2014b, while any MATLAB version after R2013a should be fine as well.
First thing the users need to do would be installing and setting up [MatConvNet](http://www.vlfeat.org/matconvnet/), which is fairly simple and quick.
While MatConvNet v1.0 beta-7 is used and included in this project, newer versions should work as well.
The main MATLAB file is <code>backend/src/demo.m</code>, which should be ready to run after MatConvNet is correctly installed, and both MatConvNet and <code>backend/src/minConf</code> are in MATLAB’s search path.
Please follow <code>demo.m</code> to compile the executable, and place the generated <code>demo</code> and <code>run_demo.sh</code> under <code>backend/</code>.
If installing MATLAB is not an option, users can check out our [precompiled executables](https://drive.google.com/folderview?id=0B8LpM_21I0tYfmtjdHFoenByeVhnTkZaRWRDUkZneHQzWDVZUi1VdTFxcVRxaDQ2UnFzWnM&usp=sharing) as well.
After obtaining the executable, users also need to specify the path to MATLAB runtime libraries in <code>backend/MCR/</code>.
If users have a full MATLAB installation, a symbolic link, e.g. <code>backend/MCR/v84 -> /usr/local/MATLAB/R2014b/</code>, can be created to do this.
If users don’t have a full MATALB and decide to use our precompiled executable, please download and install the [MATLAB Compiler Runtime](http://www.mathworks.com/products/compiler/mcr/) into <code>backend/MCR/</code>.
Also remember to download the pretrained deep learning networks [[1](http://www.vlfeat.org/matconvnet/models/imagenet-caffe-ref.mat),[2](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-s.mat),[3](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)] into <code>backend/networks/</code>.
Lastly, adjust <code>backend/run.py</code> and <code>backend/info.py</code> for running locally or distributedly as well.
### Deployment
If users decided to run the backend distributedly, simply remember to start Redis and Celery by e.g. <code>redis-server</code> and <code>celery worker -A backend.run --loglevel=info</code> (for Celery, under the main directory, not <code>backend/</code>).
Otherwise, nothing needs to be done.

