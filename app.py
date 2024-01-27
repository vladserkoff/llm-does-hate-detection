"""An embarrassingly primitive flask app with a database to store the requests and responses from the API calls."""

import datetime as dt
import os

import peewee as pw
import requests
from flask import Flask, request
from playhouse.flask_utils import FlaskDB

from helpers.data import postproces_prediction

DATABASE_NAME = "hate-detection.sqlite"
OLLAMA_ENDPOINT = "http://localhost:11434"

app = Flask(__name__)
app.config.from_object(__name__)

db_wrapper = FlaskDB(app, f"sqlite:///{DATABASE_NAME}", excluded_routes=("healthcheck",))


class Request(db_wrapper.Model):
    """Request model to store the API requests."""

    timestamp = pw.DateTimeField(default=dt.datetime.utcnow)
    request = pw.TextField()
    response = pw.TextField()  # techincally a JSON
    latency = pw.FloatField()
    is_hate = pw.BooleanField()

    class Meta:
        table_name = "requests"


def init_db():
    """Initialize the database."""
    if not os.path.exists(DATABASE_NAME):
        db_wrapper.database.connect()
        db_wrapper.database.create_tables([Request])
        db_wrapper.database.close()


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    """Healthcheck endpoint."""
    resp = requests.get(OLLAMA_ENDPOINT)
    try:
        resp.raise_for_status()
        return "OK", 200
    except requests.exceptions.HTTPError:
        return "ERROR", 503


@app.route("/infer", methods=["POST"])
def infer():
    """Inference endpoint."""
    req = Request()
    req.request = request.get_data()
    if len(request.data) == 0:
        return "Empty request", 400
    req.timestamp = dt.datetime.utcnow()
    ollama_body = {"prompt": request.data.decode(), "stream": False, "model": "hate-detect"}
    resp = requests.post(f"{OLLAMA_ENDPOINT}/api/generate", json=ollama_body)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        return "OLLAMA error", 500
    req.response = resp.json()
    req.latency = req.response["total_duration"]
    sanitized_response = postproces_prediction(req.response["response"])
    req.is_hate = sanitized_response == "Hate"
    with db_wrapper.database.atomic():
        req.save()
    return sanitized_response, 200


init_db()
