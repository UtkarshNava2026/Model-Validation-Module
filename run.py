#!/usr/bin/env python3
"""Launch the Nava Vision AI Model Validation UI."""

import os

import uvicorn


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
