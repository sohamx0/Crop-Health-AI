"""WSGI entrypoint for production servers (e.g., Waitress).

Run (Windows-friendly):
  waitress-serve --listen=0.0.0.0:5000 wsgi:app
"""

from app import app

