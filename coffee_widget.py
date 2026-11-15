"""Reusable Streamlit helpers for rendering a floating support button."""

from __future__ import annotations

from html import escape

import streamlit as st

DEFAULT_COFFEE_URL = "https://paypal.me/jgoncalocouto/1"

_FLOATING_BUTTON_HTML = """
<style>
  #fixed-coffee {{
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 9999;
  }}
  #fixed-coffee a {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 128px;
    height: 128px;
    border-radius: 999px;
    background: #ff813f;
    color: #ffffff !important;
    text-decoration: none;
    box-shadow: 0 8px 20px rgba(0,0,0,.25);
    font-size: 52px;
    line-height: 1;
    transition: transform .15s ease, box-shadow .15s ease, opacity .15s ease;
    opacity: 0.95;
  }}
  #fixed-coffee a:hover {{
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(0,0,0,.28);
    opacity: 1;
  }}
  @media (max-width: 640px) {{
    #fixed-coffee {{ bottom: 80px; right: 32px; }}
  }}
</style>

<div id="fixed-coffee">
  <a href="{url}" target="_blank" rel="noopener" title="{title}">☕</a>
</div>
"""


def add_buy_me_a_coffee(url: str = DEFAULT_COFFEE_URL, *, title: str = "Buy me a coffee") -> None:
    """Render a floating "Buy me a coffee" button in the current Streamlit app."""
    st.markdown(_FLOATING_BUTTON_HTML.format(url=escape(url, quote=True), title=escape(title, quote=True)),
                unsafe_allow_html=True)
