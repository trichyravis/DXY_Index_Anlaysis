
"""
═══════════════════════════════════════════════════════════════════════════════
STREAMLIT PROFESSIONAL DESIGN TEMPLATE - STYLES
═══════════════════════════════════════════════════════════════════════════════

Centralized CSS styling as reusable functions.
Import and use in your app to apply consistent styling!

Usage:
    from styles import apply_main_styles, get_hero_header_html
    
    apply_main_styles()
    st.markdown(get_hero_header_html(...), unsafe_allow_html=True)
"""

import streamlit as st
from config import COLORS, TYPOGRAPHY, SIDEBAR_CONFIG, HERO_HEADER, METRIC_CARD, BUTTONS, FOOTER_CONFIG, SPACING

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN STYLES APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_main_styles():
    """
    Apply all main CSS styles to the app.
    Call this ONCE at the beginning of your app.
    
    Usage:
        st.set_page_config(...)
        apply_main_styles()
    """
    st.markdown(f"""
        <style>
        {get_main_css()}
        {get_sidebar_css()}
        {get_hero_css()}
        {get_metric_card_css()}
        {get_buttons_css()}
        {get_responsive_css()}
        {get_animations_css()}
        </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL CSS GENERATORS - CUSTOMIZE AS NEEDED
# ═══════════════════════════════════════════════════════════════════════════════

def get_main_css() -> str:
    """Main page styling"""
    return f"""
    .main {{
        padding: 0rem 1rem;
    }}
    
    h1 {{
        color: {COLORS['text_dark']};
        border-bottom: 4px solid {COLORS['primary_dark']};
        padding-bottom: 0.8rem;
        font-size: {TYPOGRAPHY['h1_size']};
        font-weight: {TYPOGRAPHY['extra_bold']};
        font-family: {TYPOGRAPHY['font_primary']};
    }}
    
    h2 {{
        color: {COLORS['text_dark']};
        margin-top: 2rem;
        font-size: {TYPOGRAPHY['h2_size']};
        font-weight: {TYPOGRAPHY['bold']};
        font-family: {TYPOGRAPHY['font_primary']};
    }}
    
    h3 {{
        color: {COLORS['primary_light']};
        font-size: {TYPOGRAPHY['h3_size']};
        font-weight: {TYPOGRAPHY['bold']};
        font-family: {TYPOGRAPHY['font_primary']};
    }}
    
    p {{
        font-family: {TYPOGRAPHY['font_secondary']};
        font-size: {TYPOGRAPHY['body_size']};
    }}
    
    a {{
        color: {COLORS['primary_dark']};
        text-decoration: none;
    }}
    
    a:hover {{
        color: {COLORS['accent_gold']};
        text-decoration: underline;
    }}
    """

def get_sidebar_css() -> str:
    """Sidebar styling - dark blue gradient"""
    return f"""
    [data-testid="stSidebar"] {{
        background: {SIDEBAR_CONFIG['background_gradient']} !important;
    }}
    
    [data-testid="stSidebar"] * {{
        color: {SIDEBAR_CONFIG['text_color']} !important;
    }}
    
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: {SIDEBAR_CONFIG['header_text_color']} !important;
        font-weight: {TYPOGRAPHY['extra_bold']} !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
    }}
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {{
        color: {SIDEBAR_CONFIG['text_color']} !important;
    }}
    
    [data-testid="stSidebar"] [role="radio"] {{
        accent-color: {COLORS['accent_gold']} !important;
    }}
    
    [data-testid="stSidebar"] .stRadio > label {{
        color: {SIDEBAR_CONFIG['text_color']} !important;
        font-weight: {TYPOGRAPHY['semibold']};
    }}
    
    [data-testid="stSidebar"] hr {{
        border-color: {SIDEBAR_CONFIG['divider_color']} !important;
    }}
    
    [data-testid="stSidebar"] a {{
        color: {SIDEBAR_CONFIG['link_color']} !important;
    }}
    
    [data-testid="stSidebar"] a:hover {{
        color: {SIDEBAR_CONFIG['link_hover_color']} !important;
        text-decoration: underline;
    }}
    """

def get_hero_css() -> str:
    """Hero header styling"""
    return f"""
    .hero-title {{
        background: {HERO_HEADER['background_gradient']};
        padding: {HERO_HEADER['padding']};
        border-radius: {HERO_HEADER['border_radius']};
        margin: 0rem auto 2rem auto;
        box-shadow: {HERO_HEADER['box_shadow']};
        border: {HERO_HEADER['border_width']} solid {HERO_HEADER['border_color']};
        display: flex;
        align-items: center;
        gap: 2rem;
        max-width: {HERO_HEADER['max_width']};
        width: 90%;
    }}
    
    .mountain-emoji {{
        font-size: {HERO_HEADER['emoji_size']};
        flex-shrink: 0;
        animation: float 3s ease-in-out infinite;
        text-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }}
    
    .hero-text-right {{
        flex: 1;
        text-align: right;
    }}
    
    .hero-text-right h1 {{
        font-size: {HERO_HEADER['title_font_size']};
        font-weight: {HERO_HEADER['title_font_weight']};
        color: {HERO_HEADER['title_color']};
        margin: 0.1rem 0;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.5);
        letter-spacing: {HERO_HEADER['title_letter_spacing']};
        line-height: 1.1;
    }}
    
    .hero-text-right p:first-of-type {{
        font-size: {HERO_HEADER['subtitle_font_size']};
        color: {HERO_HEADER['subtitle_color']};
        margin: 0.8rem 0 0.3rem 0;
        font-weight: {HERO_HEADER['subtitle_font_weight']};
        letter-spacing: 0.5px;
    }}
    
    .hero-text-right p:last-of-type {{
        font-size: {HERO_HEADER['description_font_size']};
        color: {HERO_HEADER['description_color']};
        margin: 0.3rem 0 0;
        font-weight: {TYPOGRAPHY['normal']};
    }}
    """

def get_metric_card_css() -> str:
    """Metric card styling"""
    return f"""
    .metric-card {{
        background: {METRIC_CARD['background_gradient']};
        padding: {METRIC_CARD['padding']};
        border-radius: {METRIC_CARD['border_radius']};
        color: {METRIC_CARD['text_color']};
        box-shadow: {METRIC_CARD['box_shadow']};
    }}
    
    .metric-card-highlight {{
        background: {METRIC_CARD['background_gradient']};
        padding: {METRIC_CARD['padding']};
        border-radius: {METRIC_CARD['border_radius']};
        color: {METRIC_CARD['text_color']};
        box-shadow: {METRIC_CARD['box_shadow']};
        border: {METRIC_CARD['highlight_border_width']} solid {METRIC_CARD['highlight_border_color']};
    }}
    """

def get_buttons_css() -> str:
    """Button styling"""
    return f"""
    .primary-button {{
        background: {BUTTONS['primary_background']};
        color: {BUTTONS['primary_text_color']};
        padding: {BUTTONS['primary_padding']};
        border-radius: {BUTTONS['primary_border_radius']};
        font-weight: {BUTTONS['primary_font_weight']};
        border: none;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
    }}
    
    .primary-button:hover {{
        opacity: 0.9;
        transform: translateY(-2px);
    }}
    
    .secondary-button {{
        background: {BUTTONS['secondary_background']};
        color: {BUTTONS['secondary_text_color']};
        padding: {BUTTONS['primary_padding']};
        border-radius: {BUTTONS['primary_border_radius']};
        font-weight: {BUTTONS['primary_font_weight']};
        border: none;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
    }}
    
    .secondary-button:hover {{
        opacity: 0.9;
    }}
    
    .accent-button {{
        background: {BUTTONS['accent_background']};
        color: {BUTTONS['accent_text_color']};
        padding: {BUTTONS['primary_padding']};
        border-radius: {BUTTONS['primary_border_radius']};
        font-weight: {BUTTONS['primary_font_weight']};
        border: none;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
    }}
    
    .accent-button:hover {{
        opacity: 0.85;
    }}
    """

def get_responsive_css() -> str:
    """Responsive design CSS"""
    return f"""
    /* Tablet (768px) */
    @media (max-width: 768px) {{
        .hero-title {{
            flex-direction: column;
            text-align: center;
            padding: 1.5rem 1.5rem;
            gap: 1rem;
            margin: 0.5rem auto;
            max-width: 95%;
        }}
        
        .mountain-emoji {{
            font-size: 80px;
            margin: 0;
        }}
        
        .hero-text-right {{
            text-align: center;
        }}
        
        .hero-text-right h1 {{
            font-size: 24px;
            letter-spacing: 1px;
        }}
        
        .hero-text-right p:first-of-type {{
            font-size: 18px;
        }}
        
        .hero-text-right p:last-of-type {{
            font-size: 12px;
        }}
        
        h1 {{
            font-size: 32px;
        }}
        
        h2 {{
            font-size: 24px;
        }}
        
        h3 {{
            font-size: 18px;
        }}
    }}
    
    /* Mobile (480px) */
    @media (max-width: 480px) {{
        .hero-title {{
            padding: 1rem;
            max-width: 100%;
        }}
        
        .mountain-emoji {{
            font-size: 70px;
        }}
        
        .hero-text-right h1 {{
            font-size: 20px;
        }}
        
        .hero-text-right p:first-of-type {{
            font-size: 16px;
        }}
        
        h1 {{
            font-size: 24px;
        }}
        
        h2 {{
            font-size: 20px;
        }}
        
        h3 {{
            font-size: 16px;
        }}
        
        p {{
            font-size: 14px;
        }}
    }}
    """

def get_animations_css() -> str:
    """Animation definitions"""
    return f"""
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-25px); }}
    }}
    
    @keyframes bounce {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-10px); }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}
    
    @keyframes slideIn {{
        from {{ transform: translateX(-100%); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    """

# ═══════════════════════════════════════════════════════════════════════════════
# HTML COMPONENT GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def get_hero_header_html(
    title: str,
    subtitle: str,
    description: str,
    emoji: str = "🏔️"
) -> str:
    """
    Generate hero header HTML
    
    Args:
        title: Main title (bold, all caps)
        subtitle: Subtitle
        description: Description text
        emoji: Emoji to display (left side)
    
    Returns:
        HTML string
    """
    return f"""
    <div class="hero-title">
        <div class="mountain-emoji">{emoji}</div>
        <div class="hero-text-right">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <p>{description}</p>
        </div>
    </div>
    """

def get_metric_card_html(
    title: str,
    value: str,
    description: str,
    emoji: str = "📊",
    highlight: bool = False
) -> str:
    """
    Generate metric card HTML
    
    Args:
        title: Card title
        value: Main metric value
        description: Description text
        emoji: Emoji for card
        highlight: If True, adds gold border
    
    Returns:
        HTML string
    """
    css_class = "metric-card-highlight" if highlight else "metric-card"
    return f"""
    <div class="{css_class}">
        <div style="font-size: 24px; margin-bottom: 0.5rem;">{emoji}</div>
        <strong>{title}</strong>
        <div style="font-size: 18px; color: {COLORS['accent_gold']}; margin-top: 0.5rem;">{value}</div>
        <small>{description}</small>
    </div>
    """

def get_primary_button_html(
    text: str,
    url: str = "#",
    new_tab: bool = False
) -> str:
    """
    Generate primary button HTML
    
    Args:
        text: Button text
        url: Link URL
        new_tab: Open in new tab
    
    Returns:
        HTML string
    """
    target = '_blank' if new_tab else '_self'
    return f"""
    <a href="{url}" target="{target}" class="primary-button">
        {text}
    </a>
    """

def get_footer_html(
    title: str,
    description: str,
    author: str,
    social_links: dict = None
) -> str:
    """
    Generate footer HTML
    
    Args:
        title: Footer title
        description: Footer description
        author: Author name
        social_links: Dict like {"LinkedIn": "url", "GitHub": "url"}
    
    Returns:
        HTML string
    """
    social_links = social_links or {}
    
    social_html = ""
    if social_links:
        for name, url in social_links.items():
            icon_emoji = "🔗" if name == "LinkedIn" else "🐙" if name == "GitHub" else "📧"
            social_html += f"""
            <a href="{url}" target="_blank" class="primary-button" style="margin: 0 0.5rem;">
                {icon_emoji} {name}
            </a>
            """
    
    return f"""
    <div style="text-align: center; color: {FOOTER_CONFIG['text_color']}; padding: {FOOTER_CONFIG['padding']};">
        <p><strong>{title}</strong></p>
        <p>{description}</p>
        <p>{author}</p>
        <p style="margin-top: 1rem;">
            {social_html}
        </p>
    </div>
    """

# ═══════════════════════════════════════════════════════════════════════════════
# QUICK COLOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_color(color_name: str) -> str:
    """Get color by name. Usage: get_color('primary_dark')"""
    return COLORS.get(color_name, "#000000")

def get_gradient(start_color: str, end_color: str, angle: int = 135) -> str:
    """Get gradient CSS. Usage: get_gradient('primary_dark', 'primary_light')"""
    return f"linear-gradient({angle}deg, {start_color} 0%, {end_color} 100%)"

# ═══════════════════════════════════════════════════════════════════════════════
# HOW TO USE THESE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
QUICK START:

1. In your app.py:
   import streamlit as st
   from styles import apply_main_styles, get_hero_header_html
   
   st.set_page_config(...)
   apply_main_styles()
   
   st.markdown(
       get_hero_header_html(
           title="YOUR APP TITLE",
           subtitle="Your subtitle",
           description="Your description",
           emoji="📊"
       ),
       unsafe_allow_html=True
   )

2. Use metric cards:
   col1, col2, col3 = st.columns(3)
   with col1:
       st.markdown(
           get_metric_card_html(
               title="Metric 1",
               value="100/100",
               description="Description",
               emoji="📊",
               highlight=True
           ),
           unsafe_allow_html=True
       )

3. Create buttons:
   st.markdown(
       get_primary_button_html("Click Me", "https://example.com"),
       unsafe_allow_html=True
   )

4. Use footer:
   st.markdown(
       get_footer_html(
           title="YOUR APP",
           description="Description",
           author="Your Name",
           social_links={"LinkedIn": "url", "GitHub": "url"}
       ),
       unsafe_allow_html=True
   )

All styling is automatically applied from config.py!
Change colors in config.py and everything updates!
"""
