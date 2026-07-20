#!/usr/bin/env python3
"""Launch the Nava Vision AI Model Validation UI."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )
