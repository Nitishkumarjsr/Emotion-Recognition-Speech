"""
VoxEmotion AI - Quick Launcher
Starts the Flask web application server and opens the browser.
"""

import sys
import os
import webbrowser
import threading
import time

def open_browser(port):
    time.sleep(1.2)
    url = f"http://127.0.0.1:{port}"
    print(f"Opening browser at {url}...")
    webbrowser.open(url)

if __name__ == "__main__":
    from app import app
    port = int(os.environ.get("PORT", 5002))
    
    # Auto open browser in a separate thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    print("\n" + "="*60)
    print("VoxEmotion AI: Speech Emotion Recognition Web App")
    print(f"Local Web Address: http://127.0.0.1:{port}")
    print("="*60 + "\n")
    
    app.run(host="127.0.0.1", port=port, debug=False)
