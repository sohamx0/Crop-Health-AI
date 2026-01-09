# Environment Setup

## Setting Up Your API Key

1. **Create a `.env` file** in the root directory of the project:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

2. **Get your Gemini API key** from: https://makersuite.google.com/app/apikey

3. **Add the key to your `.env` file**:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

4. **The `.env` file is already in `.gitignore`**, so it won't be committed to GitHub.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

## Note

- Never commit your `.env` file to GitHub
- The `.env` file contains your personal API key
- Each developer should create their own `.env` file

