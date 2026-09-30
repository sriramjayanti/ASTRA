"""
Component styles and UI formatting constants for ASTRA GUI.
"""

from .theme_manager import ThemeManager


def get_badge_stylesheet(status_str: str) -> str:
    tm = ThemeManager.get_instance()
    color = tm.get_status_color(status_str)
    return f"""
    QLabel {{
        background-color: {color}22;
        color: {color};
        border: 1px solid {color};
        border-radius: 4px;
        padding: 2px 8px;
        font-weight: bold;
        font-size: 11px;
    }}
    """


def get_card_stylesheet(highlight: bool = False) -> str:
    tm = ThemeManager.get_instance()
    bg = tm.get_color_hex("background_elevated" if highlight else "background_panel")
    border = tm.get_color_hex("primary_cyan" if highlight else "border_subtle")
    return f"""
    QFrame {{
        background-color: {bg};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 8px;
    }}
    """


def get_monospace_stylesheet() -> str:
    tm = ThemeManager.get_instance()
    mono = tm.fonts.get("monospace", "Consolas, monospace")
    text_c = tm.get_color_hex("text_monospace", "#64dfdf")
    bg = tm.get_color_hex("background_deep", "#0a0c10")
    return f"""
    font-family: {mono};
    color: {text_c};
    background-color: {bg};
    font-size: 11px;
    """
