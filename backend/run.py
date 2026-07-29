#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app as flask_app
import app.routes
from config.server_runner import ServerRunner


if __name__ == '__main__':
    runner = ServerRunner(flask_app)
    runner.run()
