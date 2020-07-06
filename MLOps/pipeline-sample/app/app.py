import os

from flask import abort, Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

from werkzeug.exceptions import BadRequest, InternalServerError


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = \
    'postgresql://{user}:{password}@{host}/{database}'.format(
        user = os.environ['DB_USERNAME'],
        password = os.environ['DB_PASSWORD'],
        host = os.environ['DB_HOST'],
        database = os.environ['DB_NAME']
    )
db = SQLAlchemy(app)
print(app.config['SQLALCHEMY_DATABASE_URI'] )

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/jobs', methods=['POST'])
def execute():
    from models import Job
    kwargs = request.get_json(silent=True)
    if 'username' not in kwargs:
        raise BadRequest('Username required')
    job = Job(username=kwargs['username'], parameters=kwargs)
    db.session.add(job)

    try:
        db.session.commit()
    except Exception as exc:
        raise InternalServerError(exc)
    return jsonify({'id': job.id}) 


@app.route('/jobs/<int:id>', methods=['GET'])
def job(id):
    from models import Job
    job = Job.query.get(id)
    return jsonify(job.json) 
