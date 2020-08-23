from flask import Flask, url_for, request, redirect
from flask import render_template
import json
import pandas as pd

app = Flask(__name__)

import hotel_app.views