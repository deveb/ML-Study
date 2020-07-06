from app import db
from sqlalchemy.dialects.postgresql import JSON


class Job(db.Model):
    __tablename__ = 'jobs'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String())
    parameters = db.Column(JSON)
    preprocessing = db.Column(db.Integer())
    inferencing = db.Column(db.Integer())
    visualization = db.Column(db.Integer())

    def __init__(self, username, parameters):
        self.username = username
        self.parameters = parameters
        # 0: None, 1: running, 2: done
        self.preprocessing = 0
        self.inferencing = 0
        self.visualization = 0

    def __repr__(self):
        return '<Job {}>'.format(self.id)

    @property
    def json(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns} 