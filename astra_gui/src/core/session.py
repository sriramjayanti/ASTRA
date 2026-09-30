"""
ASTRA Analysis Session Manager.
Allows saving and reloading recovered signal sessions and visualization snapshots to JSON.
"""

import json
import time
from typing import Dict, Any, Optional
import os


class SessionManager:
    """Serializes and restores complete ASTRA analysis sessions."""

    @staticmethod
    def save_session(filepath: str, state_data: Dict[str, Any]) -> bool:
        """Saves current capture metadata, pipeline results, and explanations to JSON."""
        try:
            session_payload = {
                "format": "ASTRA_SESSION_V1",
                "timestamp": time.time(),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "state": state_data
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session_payload, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"[SessionManager] Failed to save session: {e}")
            return False

    @staticmethod
    def load_session(filepath: str) -> Optional[Dict[str, Any]]:
        """Restores session state from JSON file."""
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("format") == "ASTRA_SESSION_V1":
                return data.get("state")
            return None
        except Exception as e:
            print(f"[SessionManager] Failed to load session: {e}")
            return None
