"""
═══════════════════════════════════════════════════════════════════════════════
STREAMLIT PROFESSIONAL DESIGN TEMPLATE - CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════

Centralized configuration for easy customization.
Change these values to instantly update your entire app design!

Usage:
    from config import THEME, SIDEBAR_CONFIG, FOOTER_CONFIG
"""

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR SCHEME - EASILY CUSTOMIZABLE
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    # Primary Color Scheme
    "primary_dark": "#003366",      # Dark blue (sidebar, headers)
    "primary_light": "#004d80",     # Light blue (gradient)
    "primary_lightest": "#E0F0FF",  # Very light blue (text on dark)
    
    # Secondary Colors
    "accent_gold": "#FFD700",       # Gold (highlights, borders)
    "accent_light": "#FFF9E6",      # Light gold (hover states)
    
    # Neutral Colors
    "text_dark": "#003366",         # Dark text on light background
    "text_light": "white",          # Light text on dark background
    "background_light": "#F5F5F5",  # Light background
    "border_light": "#E0E0E0",      # Light borders
    
    # Status Colors
    "success": "#2ecc71",           # Green
    "warning": "#f39c12",           # Orange
    "danger": "#e74c3c",            # Red
    "info": "#3498db",              # Blue
}

# ═══════════════════════════════════════════════════════════════════════════════
# TYPOGRAPHY
# ═══════════════════════════════════════════════════════════════════════════════

TYPOGRAPHY = {
    "font_primary": "Times New Roman",      # Primary font for headers
    "font_secondary": "Arial",              # Secondary font for body
    
    # Font sizes
    "h1_size": "40px",
    "h2_size": "32px",
    "h3_size": "24px",
    "body_size": "16px",
    "small_size": "12px",
    
    # Font weights
    "light": "300",
    "normal": "400",
    "semibold": "600",
    "bold": "700",
    "extra_bold": "900",
}

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SIDEBAR_CONFIG = {
    "background_gradient": f"linear-gradient(135deg, {COLORS['primary_dark']} 0%, {COLORS['primary_light']} 50%, {COLORS['primary_dark']} 100%)",
    "text_color": COLORS["text_light"],
    "header_text_color": COLORS["text_light"],
    "link_color": COLORS["accent_gold"],
    "link_hover_color": COLORS["accent_light"],
    "divider_color": "rgba(255, 255, 255, 0.3)",
    
    # Layout
    "padding": "1rem",
    "width": "auto",  # Streamlit default
}

# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

HERO_HEADER = {
    "background_gradient": f"linear-gradient(135deg, {COLORS['primary_dark']} 0%, {COLORS['primary_light']} 50%, {COLORS['primary_dark']} 100%)",
    "padding": "2rem 2rem",
    "border_radius": "20px",
    "border_color": COLORS["primary_dark"],
    "border_width": "4px",
    "box_shadow": "0 12px 30px rgba(0, 51, 102, 0.4)",
    "max_width": "1200px",
    
    # Text styling
    "title_color": COLORS["text_light"],
    "title_font_size": "32px",
    "title_font_weight": "900",
    "title_letter_spacing": "2px",
    
    "subtitle_color": COLORS["primary_lightest"],
    "subtitle_font_size": "24px",
    "subtitle_font_weight": "600",
    
    "description_color": "#D0E8FF",
    "description_font_size": "14px",
    
    # Emoji styling
    "emoji_size": "100px",
    "emoji_animation": "float",  # float, bounce, pulse, etc.
}

# ═══════════════════════════════════════════════════════════════════════════════
# METRIC CARD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_CARD = {
    "background_gradient": f"linear-gradient(135deg, #003d70 0%, #005a9d 100%)",
    "text_color": COLORS["text_light"],
    "padding": "1.5rem",
    "border_radius": "15px",
    "box_shadow": "0 4px 15px rgba(0, 51, 102, 0.2)",
    
    # Highlight card (for special metrics)
    "highlight_border_color": COLORS["accent_gold"],
    "highlight_border_width": "2px",
}

# ═══════════════════════════════════════════════════════════════════════════════
# BUTTON CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BUTTONS = {
    "primary_background": f"linear-gradient(135deg, {COLORS['primary_dark']} 0%, {COLORS['primary_light']} 100%)",
    "primary_text_color": COLORS["text_light"],
    "primary_border_radius": "5px",
    "primary_padding": "0.5rem 1.5rem",
    "primary_font_weight": "600",
    
    "secondary_background": f"linear-gradient(135deg, #333 0%, #555 100%)",
    "secondary_text_color": COLORS["text_light"],
    
    "accent_background": f"linear-gradient(135deg, {COLORS['accent_gold']} 0%, #FFC700 100%)",
    "accent_text_color": COLORS["text_dark"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

FOOTER_CONFIG = {
    "text_color": "#666",
    "background_color": "transparent",
    "padding": "2rem",
    "text_align": "center",
    "border_top": None,  # Set to "1px solid #E0E0E0" if you want a divider
    
    # Social links
    "social_button_padding": "0.5rem 1.5rem",
    "social_button_margin": "0 0.5rem",
    "social_button_border_radius": "5px",
    "social_button_font_weight": "600",
}

# ═══════════════════════════════════════════════════════════════════════════════
# TABS & NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

NAVIGATION = {
    "active_color": COLORS["accent_gold"],
    "inactive_color": "#666",
    "border_color": COLORS["primary_dark"],
    "hover_color": COLORS["primary_light"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# SPACING & LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

SPACING = {
    "xs": "0.25rem",
    "sm": "0.5rem",
    "md": "1rem",
    "lg": "1.5rem",
    "xl": "2rem",
    "xxl": "3rem",
}

# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSIVE BREAKPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

BREAKPOINTS = {
    "mobile": "480px",      # Small phones
    "tablet": "768px",      # Tablets
    "desktop": "1024px",    # Desktops
    "large": "1440px",      # Large monitors
}

# ═══════════════════════════════════════════════════════════════════════════════
# APP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PAGE_CONFIG = {
    "page_title": "DXY Index Analytics | The Mountain Path",
    "page_icon": "💵",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENT ACCESS - THEME DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════════

THEME = {
    "colors": COLORS,
    "typography": TYPOGRAPHY,
    "sidebar": SIDEBAR_CONFIG,
    "hero": HERO_HEADER,
    "cards": METRIC_CARD,
    "buttons": BUTTONS,
    "footer": FOOTER_CONFIG,
    "navigation": NAVIGATION,
    "spacing": SPACING,
}

# ═══════════════════════════════════════════════════════════════════════════════
# QUICK THEME SWITCHING (FUTURE FEATURE)
# ═══════════════════════════════════════════════════════════════════════════════

THEME_PRESETS = {
    "light": {
        "primary_dark": "#F5F5F5",
        "primary_light": "#FFFFFF",
        "text_dark": "#333333",
        "text_light": "#000000",
    },
    "dark": {
        "primary_dark": "#1A1A1A",
        "primary_light": "#2D2D2D",
        "text_dark": "#FFFFFF",
        "text_light": "#FFFFFF",
    },
    "ocean": {
        "primary_dark": "#0077BE",
        "primary_light": "#0096FF",
        "accent_gold": "#00D9FF",
        "text_dark": "#0077BE",
        "text_light": "#FFFFFF",
    },
    "forest": {
        "primary_dark": "#1B4332",
        "primary_light": "#2D6A4F",
        "accent_gold": "#95D5B2",
        "text_dark": "#1B4332",
        "text_light": "#FFFFFF",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# HOW TO USE THIS FILE
# ═══════════════════════════════════════════════════════════════════════════════

"""
QUICK START:

1. Change colors:
   COLORS["primary_dark"] = "#YOUR_COLOR"
   COLORS["accent_gold"] = "#YOUR_COLOR"

2. Change fonts:
   TYPOGRAPHY["font_primary"] = "Your Font Name"
   TYPOGRAPHY["h1_size"] = "50px"

3. In your app:
   from config import COLORS, THEME
   
   st.markdown(f'''
       <style>
       h1 {{ color: {COLORS["text_dark"]}; }}
       </style>
   ''', unsafe_allow_html=True)

4. Use theme colors anywhere:
   primary_color = THEME["colors"]["primary_dark"]
   sidebar_bg = THEME["sidebar"]["background_gradient"]

Everything is customizable from ONE FILE!
"""
