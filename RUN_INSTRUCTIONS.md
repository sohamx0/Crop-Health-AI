# How to Run CropHealth AI Website

## Quick Start

1. **Open Command Prompt or PowerShell**
   - Navigate to the project folder: `cd C:\Users\Lenovo\EDI`

2. **Activate Virtual Environment**
   ```bash
   edi_env\Scripts\activate
   ```
   You should see `(edi_env)` in your prompt.

3. **Run the Flask Application**
   ```bash
   python app.py
   ```

4. **Open Your Browser**
   - Go to: **http://localhost:5000**
   - Or: **http://127.0.0.1:5000**

## Expected Output

When running successfully, you should see:
```
✅ Local disease detection model (ResNet50) loaded successfully!
 * Running on http://0.0.0.0:5000
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

## Pages Available

- **Homepage**: http://localhost:5000/
- **Disease Library**: http://localhost:5000/library
- **AI Scanner**: http://localhost:5000/scanner

## Health Check

- **Health endpoint**: http://localhost:5000/healthz

## Production Run (Recommended)

Flask's built-in server is for development. For a more production-like run on Windows, use Waitress:

1. Install deps:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the WSGI server:
   ```bash
   waitress-serve --listen=0.0.0.0:5000 wsgi:app
   ```

## Tailwind CSS

✅ **Tailwind CSS is working!** All pages use the Tailwind CDN, so you need an internet connection for styling to load.

## Troubleshooting

- **Port already in use?** Change port in `app.py` line 113: `app.run(host='0.0.0.0', port=5001, debug=True)`
- **Module not found?** Make sure virtual environment is activated
- **Styling not working?** Check internet connection (Tailwind loads from CDN)

## Stop the Server

Press `Ctrl+C` in the terminal where the server is running.

