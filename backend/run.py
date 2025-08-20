#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == '__main__':
    print("")
    print("API will be available at: http://localhost:5001")
    print("")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
