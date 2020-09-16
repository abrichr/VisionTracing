# Have to add worker class eventlet for gunicorn to work correctly with 
# socketio
# https://flask-socketio.readthedocs.io/en/latest/#gunicorn-web-server
gunicorn -b 0.0.0.0:$PORT --worker-class eventlet -w 1 app:app --timeout 500
