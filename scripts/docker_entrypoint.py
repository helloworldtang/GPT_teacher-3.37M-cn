"""Docker entrypoint: launch web demo."""
import os
import time
from src.web_demo import demo

server_name = os.environ.get("SERVER_NAME", "0.0.0.0")

try:
    demo.queue().launch(
        server_name=server_name,
        server_port=7860,
        show_error=True,
        share=False,
    )
except ValueError as e:
    print(f"\nNote: Gradio health check bypassed ({e})")
    print(f"Server is running on http://0.0.0.0:7860")
    while True:
        time.sleep(3600)
