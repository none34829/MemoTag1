#!/usr/bin/env python
"""
Startup script for Render deployment
"""
import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}")
    
    # Start Uvicorn with explicit port from environment
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")
