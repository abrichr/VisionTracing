import os
import time
import json

from redis import Redis
from flask import (
    Flask, abort, Blueprint, flash, render_template, redirect, request,
    url_for, flash, jsonify, send_from_directory, Markup, session
)
from loguru import logger
from rq import Queue
from rq.job import Job
import rq_dashboard
from worker import conn, redis_url
from flask_socketio import SocketIO

q = Queue(connection=conn)

app = Flask(__name__, static_url_path='/static')
app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')
app.config['RQ_DASHBOARD_REDIS_URL'] = redis_url
app.config.from_object(rq_dashboard.default_settings)
app.register_blueprint(
    rq_dashboard.blueprint, url_prefix='/rq/'
)
app.secret_key = 'test_key'

socket = SocketIO(
            app,
            cors_allowed_origins='*',
            message_queue=os.getenv('REDIS_URL'),
            manage_session=False
         )

@app.template_filter('empty_session')
def empty_session(session):
    '''
    This function removes all killed/timed-out jobs from the session if the video path for a
    given job does not exist. Afterwards, it specifies whether or not the session object is
    empty.
    Parameters:
    - session: Flask session object
    '''

    if 'jobs' not in session:
        logger.info('Empty session')
        return True

    def clear_session():
        for idx, job_dict in enumerate(session['jobs']):
            job = q.fetch_job(job_dict['id'])
            video_source = 'static/videos/{}'.format(job_dict['tracks_filename'])

            dead_job_or_timeout = job is None or job.get_status() == 'failed'
            video_does_not_exist = not os.path.exists(video_source)

            if dead_job_or_timeout and video_does_not_exist:
                session['jobs'].pop(idx)
                # Have to set modified to True after every in-place operation
                session.modified = True
                logger.info('Popped job from session: id={}'.format(job_dict['id']))

    clear_session()

    if len(session['jobs']) == 0:
        logger.info('Empty session')
        return True

    return False

@app.template_filter('refresh_job')
def refresh_job(job_dict):
    '''
    This function updates the meta dictionary of a given job and returns the
    job's filename
    Parameters:
    - job_dict: Dictionary containing a job id and job filename
    '''
    logger.info('Refresh job, id={}'.format(job_dict['id']))
    job = q.fetch_job(job_dict['id'])
    filename = job_dict['filename']
    try:
        job.refresh()
        logger.info('Job refreshed successfully')
    except Exception as e:
        logger.info('Job did not refresh properly, exception {}'.format(e))
    return filename

@app.template_filter('display_job')
def display_job(job_dict):
    '''
    This function checks whether or not a video with the job's track filename
    exists. If it exists, then it returns HTML markup for the video source.
    Else, it returns the progress information.
    Parameters:
    - job: Dictionary containing a job id and tracks filename
    '''
    logger.info('Displaying job, id={}'.format(job_dict['id']))
    video_source = 'static/videos/{}'.format(job_dict['tracks_filename'])
    if os.path.exists(video_source):
        return Markup("""
        <video width="320" height="240" controls>
          <source src="/{}" type="video/mp4">
        </video>
        """.format(video_source))
    job = q.fetch_job(job_dict['id'])
    if job and job.meta.get('status'):
        return job.meta.get('status')
    return 'Beginning process...'

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def upload():
    logger.info(f'upload() request.files: {request.files}')
    video_file = request.files.get('file')
    fname_video = video_file.filename
    video_stream = video_file.read()
    with open(fname_video, 'wb') as f:
        f.write(video_stream)

    one_week = 60 * 60 * 24 * 7
    fname, extension = fname_video.split('.')
    output_file = '{}-tracks{}.{}'.format(fname, time.time(), extension)

    job = q.enqueue(
        'vision.get_tracking_video',
        args=(fname_video, output_file),
        timeout=one_week
    )
    job.filename = fname_video
    job.tracks_filename = output_file
    job_dict = {
                'id': job.id,
                'filename': job.filename,
                'tracks_filename': job.tracks_filename
               }

    if 'jobs' not in session:
        session['jobs'] = []
        logger.info('Creating jobs for user session')
    session['jobs'].append(job_dict)
    session.modified = True # In-place operations not picked up automatically
    logger.info(f'job: {job}')

    return {
        'status': 200,
        'mimetype': 'application/json'
    }

@app.route('/test')
def test():
    logger.info('test()')
    one_day = 60 * 60 * 24
    result = q.enqueue(
        'utils.count_words_at_url',
        'http://news.ycombinator.com',
        job_timeout=one_day
    )
    logger.info(f'q: {q}')
    logger.info(f'result: {result}')
    return 'Enqueued'


@app.route('/')
def index():
    # TODO: use websockets to update client without refresh
    return render_template(
        'index.html',
        q=q
    )
