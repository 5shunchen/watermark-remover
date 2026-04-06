#!/usr/bin/env python3
"""
Test script to verify the video page route is working
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Test imports
try:
    from src.api.main import app, PROJECT_ROOT
    print("✓ Successfully imported app from src.api.main")

    # Check routes
    routes = []
    for route in app.routes:
        routes.append(f"  - {route.path}")

    print("\nRegistered routes:")
    for route_path in sorted(set(routes)):
        print(route_path)

    # Check if /video is registered
    video_route_exists = any("/video" in str(getattr(route, 'path', '')) for route in app.routes)
    if video_route_exists:
        print("\n✓ /video route is registered")
    else:
        print("\n✗ /video route NOT found!")

    # Check templates
    video_template = PROJECT_ROOT / "templates" / "video.html"
    if video_template.exists():
        print(f"✓ video.html exists at: {video_template}")
    else:
        print(f"✗ video.html NOT found at: {video_template}")

    # Check static files
    video_css = PROJECT_ROOT / "static" / "video.css"
    video_js = PROJECT_ROOT / "static" / "video.js"

    if video_css.exists():
        print(f"✓ video.css exists")
    else:
        print(f"✗ video.css NOT found")

    if video_js.exists():
        print(f"✓ video.js exists")
    else:
        print(f"✗ video.js NOT found")

except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
