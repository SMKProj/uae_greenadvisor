"""
GreenAdvisor — UAE Plant Care & Landscaping Advisor
====================================================
Primary Engine : Groq  (llama-3.3-70b-versatile)          — text diagnosis
Vision Engine  : Groq  (meta-llama/llama-4-scout-17b)      — image analysis
Vision Fallback: Groq  (meta-llama/llama-4-maverick-17b)   — if scout rate-limited
Weather        : Open-Meteo (free, no key needed)
Built by       : Sundas Khan

No external vision API keys required — everything runs on GROQ_API_KEY.
"""

import io
import os
import time
import base64
import requests
from datetime import datetime

import streamlit as st
from groq import Groq
from PIL import Image
# Vision: Groq Llama-4 (maverick + scout fallback)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenAdvisor — UAE Plant Care",
    page_icon="🌿",
    layout="centered",          # ← centered layout as requested
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS STYLING  (centered, polished, matching artifact UI)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=DM+Sans:wght@400;500&display=swap');

:root {
    --green-dark   : #2d5a1b;
    --green-mid    : #3d7a25;
    --green-bright : #4e9a30;
    --green-light  : #c8e6c0;
    --green-pale   : #edf7e9;
    --cream        : #faf6ec;
    --text-dark    : #1a2e12;
    --text-mid     : #3d5c2e;
    --text-light   : #6b8f5e;
    --border       : #a8d090;
    --border-light : #d4e8cc;
    --white        : #ffffff;
}

html, body, [class*="css"] {
    font-family : 'DM Sans', sans-serif;
    background  : var(--cream) !important;
    color       : var(--text-dark);
}

#MainMenu, footer, header { visibility: hidden; }

/* Center & cap width */
.block-container {
    max-width   : 860px !important;
    padding-top : 1.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}
[data-testid="stAppViewContainer"] { background: var(--cream); }

/* ── Header ───────────────────────────────────────────────── */
.ga-header {
    background    : var(--green-dark);
    border-radius : 14px 14px 0 0;
    padding       : 14px 20px;
    display       : flex;
    align-items   : center;
    gap           : 14px;
    margin-bottom : 0;
}
.ga-header-icon {
    width          : 38px;
    height         : 38px;
    background     : rgba(255,255,255,0.15);
    border-radius  : 50%;
    display        : flex;
    align-items    : center;
    justify-content: center;
    font-size      : 20px;
    flex-shrink    : 0;
}
.ga-header-text { flex: 1; }
.ga-header-title {
    font-family : 'Playfair Display', serif;
    font-size   : 17px;
    font-weight : 600;
    color       : #edf7e9;
    line-height : 1.3;
    margin      : 0;
}
.ga-header-arabic {
    font-size : 17px;
    color     : #a8d5a0;
    margin    : 4px 0 0;
    font-weight: 400;
}

/* ── Weather grid ─────────────────────────────────────────── */
.weather-grid {
    display               : grid;
    grid-template-columns : repeat(4, 1fr);
    border-left           : 1px solid var(--green-mid);
    border-right          : 1px solid var(--green-mid);
}
.weather-city-header {
    background    : var(--green-mid);
    color         : #edf7e9;
    text-align    : center;
    padding       : 9px 6px;
    font-weight   : 500;
    font-size     : 13px;
    border-right  : 1px solid var(--green-dark);
}
.weather-city-header:last-child { border-right: none; }
.weather-city-body {
    background     : var(--green-light);
    text-align     : center;
    padding        : 10px 4px;
    border-right   : 1px solid #a8c8a0;
    border-bottom  : 1px solid #a8c8a0;
    min-height     : 64px;
    display        : flex;
    flex-direction : column;
    align-items    : center;
    justify-content: center;
    gap            : 2px;
}
.weather-city-body:last-child { border-right: none; }
.weather-temp {
    font-family : 'Playfair Display', serif;
    font-size   : 20px;
    font-weight : 600;
    color       : var(--green-dark);
}
.weather-desc { font-size: 11px; color: var(--text-mid); }
.weather-hum  { font-size: 10px; color: var(--text-light); }
.alert-critical { color: #c0392b; font-weight: 500; }
.alert-elevated { color: #d68910; font-weight: 500; }
.alert-normal   { color: var(--green-mid); font-weight: 500; }

/* ── Section labels ───────────────────────────────────────── */
.section-label {
    background    : var(--green-dark);
    color         : #edf7e9;
    font-weight   : 500;
    font-size     : 13px;
    padding       : 7px 13px;
    border-radius : 6px 6px 0 0;
    margin-bottom : 0;
    letter-spacing: 0.2px;
}

/* ── App body card ────────────────────────────────────────── */
.ga-body-card {
    background    : var(--cream);
    border        : 1px solid var(--green-mid);
    border-top    : none;
    border-radius : 0 0 14px 14px;
    padding       : 16px;
    margin-bottom : 0;
}

/* ── Result box ───────────────────────────────────────────── */
.result-box {
    background    : var(--white);
    border        : 1px solid var(--border);
    border-top    : none;
    border-radius : 0 0 8px 8px;
    padding       : 16px;
    min-height    : 200px;
    font-size     : 14px;
    line-height   : 1.85;
    color         : var(--text-dark);
    white-space   : pre-wrap;
    word-break    : break-word;
    position      : relative;
}
.copy-btn {
    position      : absolute;
    top           : 10px;
    right         : 10px;
    background    : var(--green-pale);
    border        : 1px solid var(--border);
    border-radius : 6px;
    padding       : 3px 10px;
    font-size     : 11px;
    color         : var(--green-dark);
    cursor        : pointer;
    font-family   : "DM Sans", sans-serif;
}
.copy-btn:hover { background: var(--green-light); }
.ga-analyzing {
    display       : flex;
    align-items   : center;
    gap           : 10px;
    color         : var(--green-mid);
    font-size     : 13px;
    padding       : 30px 16px;
    font-style    : italic;
}
.ga-dot-pulse {
    width         : 8px;
    height        : 8px;
    border-radius : 50%;
    background    : var(--green-mid);
    animation     : pulse 1.2s ease-in-out infinite;
    flex-shrink   : 0;
}
@keyframes pulse {
    0%, 100% { opacity: 0.3; transform: scale(0.8); }
    50%       { opacity: 1;   transform: scale(1.2); }
}
.result-empty {
    color       : var(--text-light);
    font-style  : italic;
    font-size   : 13px;
    text-align  : center;
    padding-top : 50px;
}

/* ── History chat bubbles ─────────────────────────────────── */
.chat-wrap { margin-bottom: 10px; }
.chat-label {
    font-size   : 10px;
    font-weight : 500;
    color       : #7ab870;
    margin-bottom: 3px;
}
.chat-msg-user {
    background    : var(--green-light);
    color         : #1a3d0f;
    border-radius : 8px;
    padding       : 9px 13px;
    font-size     : 13px;
    line-height   : 1.6;
    margin-left   : 30px;
    white-space   : pre-wrap;
    word-break    : break-word;
}
.chat-msg-advisor {
    background    : var(--white);
    border        : 1px solid var(--border-light);
    border-radius : 8px;
    padding       : 9px 13px;
    font-size     : 13px;
    line-height   : 1.75;
    margin-right  : 30px;
    white-space   : pre-wrap;
    word-break    : break-word;
}

/* ── Image preview box ────────────────────────────────────── */
.image-preview-box {
    background    : var(--white);
    border        : 1px solid var(--border);
    border-top    : none;
    border-radius : 0 0 8px 8px;
    min-height    : 140px;
    display       : flex;
    align-items   : center;
    justify-content: center;
    padding       : 10px;
}
.image-placeholder {
    color       : var(--text-light);
    font-size   : 13px;
    font-style  : italic;
    text-align  : center;
    line-height : 1.6;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    background    : var(--green-dark) !important;
    color         : #edf7e9 !important;
    border        : none !important;
    border-radius : 7px !important;
    font-weight   : 500 !important;
    font-size     : 13px !important;
    padding       : 8px 20px !important;
    width         : 100% !important;
    font-family   : 'DM Sans', sans-serif !important;
    transition    : background 0.15s !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover { background: var(--green-mid) !important; }
.stButton > button:active { transform: scale(0.97) !important; }

/* ── Textarea ─────────────────────────────────────────────── */
.stTextArea textarea {
    background    : var(--white) !important;
    border        : 1px solid var(--border) !important;
    border-radius : 0 0 7px 7px !important;
    font-size     : 13px !important;
    font-family   : 'DM Sans', sans-serif !important;
    color         : var(--text-dark) !important;
    padding       : 11px !important;
    resize        : none !important;
}
.stTextArea textarea:focus {
    border-color: var(--green-mid) !important;
    box-shadow  : 0 0 0 2px rgba(61,122,37,0.12) !important;
}

/* ── Selectbox ────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div {
    background    : var(--white) !important;
    border        : 1px solid var(--border) !important;
    border-radius : 7px !important;
    font-size     : 13px !important;
}

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background    : var(--green-light) !important;
    border        : 1px dashed var(--green-mid) !important;
    border-top    : none !important;
    border-radius : 0 0 7px 7px !important;
    padding       : 8px !important;
}

/* ── Vision status badge ──────────────────────────────────── */
.vision-badge {
    display       : inline-flex;
    align-items   : center;
    gap           : 5px;
    font-size     : 11.5px;
    border-radius : 20px;
    padding       : 3px 10px;
    margin-top    : 6px;
}
.vision-ok  { background: var(--green-pale); border: 1px solid var(--border); color: var(--green-dark); }
.vision-warn{ background: #fcebeb; border: 1px solid #f09595; color: #a32d2d; }
.vision-info{ background: #faeeda; border: 1px solid #fac775; color: #854f0b; }

/* ── Divider ──────────────────────────────────────────────── */
.divider { height: 1px; background: var(--border-light); margin: 14px 0; }

/* ── Footer ───────────────────────────────────────────────── */
.ga-footer-1 {
    background    : var(--green-pale);
    border-top    : 1px solid var(--border);
    padding       : 10px 20px;
    text-align    : center;
    font-size     : 12.5px;
    color         : var(--text-mid);
    border-radius : 0;
}
.ga-footer-1 a { color: var(--green-dark); font-weight: 500; text-decoration: underline; }
.ga-footer-2 {
    background    : var(--green-light);
    border-top    : 1px solid var(--border);
    padding       : 9px 20px;
    text-align    : center;
    font-size     : 12px;
    color         : var(--text-dark);
    font-weight   : 500;
    border-radius : 0 0 14px 14px;
}

[data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
UAE_CITIES = {
    "Dubai"    : {"lat": 25.2048, "lon": 55.2708, "owm": "Dubai,AE"},
    "Abu Dhabi": {"lat": 24.4539, "lon": 54.3773, "owm": "Abu+Dhabi,AE"},
    "Sharjah"  : {"lat": 25.3463, "lon": 55.4209, "owm": "Sharjah,AE"},
    "Ajman"    : {"lat": 25.4052, "lon": 55.5136, "owm": "Ajman,AE"},
}

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are GreenAdvisor — a professional UAE plant care and landscaping expert.
You serve homeowners, gardeners, villa managers, hotel teams, and B2B clients
across Dubai, Abu Dhabi, Sharjah, Ajman, and all UAE emirates.

━━━ LANGUAGE ━━━
- Auto-detect language. English → English. Arabic → Arabic. Mirror mixed use.
- Introduce botanical names in both languages on first mention.

━━━ PERSONALITY ━━━
- Expert, warm, conversational — like a trusted plant doctor.
- Always UAE-specific. Never give generic global advice.
- Keep responses concise and invite follow-up questions.

━━━ UAE CLIMATE CONTEXT ━━━
- Summer (Jun-Sep): 42-48°C, high humidity, intense UV, heat stress common
- Winter (Nov-Feb): 15-25°C, low humidity, ideal planting season
- Sandstorm season (Mar-May, Sep-Oct): dust blocks stomata, causes stress
- Hard water TDS 300-600 ppm causes salt buildup in soil
- Sandy, low-organic, fast-draining soil — nutrients leach quickly

━━━ CONVERSATION STYLE — TWO STAGE ━━━
STAGE 1 — QUICK ASSESSMENT (always start here):
Give a SHORT first response with only:
  🌿 Plant: [Name English (Arabic)]
  ❤️ Health: [One sentence — what you see]
  🎯 Severity: [✅ Healthy / 🟢 Mild / 🟡 Moderate / 🔴 Severe]
  💬 [One friendly sentence inviting a specific follow-up]
  Examples of follow-up invitations:
  - "Would you like the full treatment plan?"
  - "Shall I give you a UAE-specific watering schedule?"
  - "Want to know which products to buy locally?"

STAGE 2 — DETAILED RESPONSE (only when user asks for more):
Provide the full structured plan using this format:
  🌿 Plant Identified: [Name English (Arabic)]
  ⚠️ Problem: [Name]
  🎯 Severity: [✅ Healthy / 🟢 Mild / 🟡 Moderate / 🔴 Severe]
  🔍 Why in UAE: [1-2 sentences]
  💊 Treatment:
     1. [Step]
     2. [Step]
     3. [Step]
  💧 Watering: [Times + frequency for UAE]
  🛒 Buy in UAE: [Products + stores: ACE Hardware, Lulu, Garden Centre, Plantshop.ae]
  🛡️ Prevention: [One tip]
  📅 Follow-up: Check back in 3-5 days.

━━━ SEVERITY RULES — CRITICAL ━━━
- ✅ Healthy: plant looks green, firm, no spots, no wilting, normal growth.
  DO NOT assign Moderate or Severe to a healthy-looking plant.
  DO NOT invent problems that are not clearly visible.
- 🟢 Mild: very minor yellowing on 1-2 leaves, slight wilting, minor dust.
- 🟡 Moderate: clear yellowing on multiple leaves, visible pest damage, noticeable wilting.
- 🔴 Severe: widespread browning, heavy pest infestation, root rot signs, near-death.
DEFAULT TO HEALTHY unless you can see a clear, specific symptom.

━━━ IMAGE ANALYSIS — CRITICAL RULES ━━━
1. DESCRIBE FIRST before naming: state what you literally see —
   leaf colour, shape, texture, stem colour, any flowers/fruit, growth habit.
2. Name the plant based ONLY on visual evidence. NEVER assume species from
   user text. If unsure, give top 2 possibilities with confidence %.
3. State your confidence: "I am [X]% confident this is [plant]."
4. If image is blurry/unclear, say so and ask for a better photo.
5. Check for: yellowing, brown tips, spots, wilting, pests, root exposure,
   soil dryness, salt crust on pot rim — only flag what you can actually SEE.

━━━ TORONTO AGRICULTURE ━━━
Only mention when user explicitly asks for professional on-site help.
Say: "For professional landscaping, pools, and plant services across Dubai,
Abu Dhabi, and Sharjah, contact Toronto Agriculture — UAE specialists
with over a decade of experience."
"""


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
defaults = {
    "chat_history"    : [],   # list of {"role": "user"|"advisor", "text": str}
    "groq_history"    : [],   # Groq API message list
    "last_call_time"  : 0.0,
    "pending_image"   : None, # PIL Image
    "pending_img_name": None,
    "pending_img_b64" : None, # base64 JPEG string for Groq Vision
    "weather_data"    : {},
    "widget_counter"  : 0,    # incremented to force widget reset (clear/remove)
    "input_text"      : "",   # current input box value
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Snapshot the image b64 BEFORE any widgets render this rerun.
# file_uploader with key='plant_image' can clear its internal buffer on
# reruns triggered by other buttons, losing the uploaded file even if
# session_state.pending_img_b64 was set correctly on the previous run.
# Reading it here captures the value before any widget side-effects.
_img_b64_snapshot = st.session_state.pending_img_b64


# ─────────────────────────────────────────────────────────────
# API CLIENTS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not key:
        st.error("GROQ_API_KEY not found in Streamlit secrets or environment variables.")
        st.stop()
    return Groq(api_key=key)



groq_client = get_groq_client()


# ─────────────────────────────────────────────────────────────
# IMAGE HELPER — PIL → base64 JPEG
# ─────────────────────────────────────────────────────────────
def pil_to_base64_jpeg(pil_img: Image.Image, max_size: int = 1024) -> str:
    """
    Converts PIL Image to base64-encoded JPEG string.
    Resizes to max_size on the longest side to keep payload small.
    Converts PIL Image to base64 JPEG for Groq Vision API.
    """
    img = pil_img.copy()

    # 🔥 CRITICAL FIX: Always convert first (WebP often loads as RGBA/P)
    img = img.convert("RGB")

    # Resize if very large (reduces API payload)
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)

    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ─────────────────────────────────────────────────────────────
# WEATHER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def get_weather(city: str) -> dict:
    """
    Tries Open-Meteo first (free, no key needed).
    Falls back to OpenWeatherMap if OPENWEATHER_API_KEY is set.
    Falls back to realistic mock data if both fail.
    """
    coords = UAE_CITIES[city]

    # ── Open-Meteo (free, no API key) ────────────────────────
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={coords['lat']}&longitude={coords['lon']}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
            f"&wind_speed_unit=kmh&timezone=Asia%2FDubai"
        )
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            d = r.json()["current"]
            code_map = {
                0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
                45:"Foggy", 48:"Icy fog", 51:"Light drizzle", 61:"Light rain",
                63:"Moderate rain", 80:"Rain showers", 95:"Thunderstorm",
            }
            desc = code_map.get(d["weather_code"],
                                "Sunny" if d["weather_code"] <= 2 else "Hazy")
            return {
                "temp"    : round(d["temperature_2m"]),
                "humidity": round(d["relative_humidity_2m"]),
                "desc"    : desc,
                "wind"    : round(d["wind_speed_10m"]),
                "live"    : True,
                "source"  : "Open-Meteo",
            }
    except Exception:
        pass

    # ── OpenWeatherMap fallback ───────────────────────────────
    owm_key = st.secrets.get("OPENWEATHER_API_KEY",
                             os.environ.get("OPENWEATHER_API_KEY", ""))
    if owm_key:
        try:
            r = requests.get(
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?q={coords['owm']}&appid={owm_key}&units=metric",
                timeout=6,
            )
            if r.status_code == 200:
                d = r.json()
                return {
                    "temp"    : round(d["main"]["temp"]),
                    "humidity": d["main"]["humidity"],
                    "desc"    : d["weather"][0]["description"].capitalize(),
                    "wind"    : round(d["wind"]["speed"] * 3.6),
                    "live"    : True,
                    "source"  : "OpenWeatherMap",
                }
        except Exception:
            pass

    # ── Mock fallback ─────────────────────────────────────────
    month = datetime.now().month
    if   month in [6, 7, 8]:     t, h, desc = 44, 74, "Hazy sunshine"
    elif month in [12, 1, 2]:    t, h, desc = 22, 52, "Clear sky"
    elif month in [3, 4, 5]:     t, h, desc = 34, 48, "Partly cloudy"
    else:                         t, h, desc = 38, 65, "Sunny"

    offsets = {"Dubai":(0,0), "Abu Dhabi":(-1,-4), "Sharjah":(1,4), "Ajman":(0,2)}
    dt, dh  = offsets.get(city, (0, 0))
    return {"temp": t+dt, "humidity": h+dh, "desc": desc,
            "wind": 18, "live": False, "source": "estimated"}


def detect_user_city() -> str:
    """
    Silently detect user's approximate city via IP geolocation (ipapi.co).
    Returns the nearest UAE city name, defaults to 'Sharjah' on failure.
    Free tier — no API key needed, 1000 req/day.
    """
    if "detected_city" in st.session_state:
        return st.session_state.detected_city

    UAE_CITY_COORDS = {
        "Dubai"    : (25.2048, 55.2708),
        "Abu Dhabi": (24.4539, 54.3773),
        "Sharjah"  : (25.3463, 55.4209),
        "Ajman"    : (25.4052, 55.5136),
    }
    try:
        r = requests.get("https://ipapi.co/json/", timeout=4)
        if r.status_code == 200:
            d = r.json()
            lat = float(d.get("latitude",  25.3463))
            lon = float(d.get("longitude", 55.4209))
            country = d.get("country_code", "AE")
            # Find nearest UAE city by Euclidean distance
            nearest = min(
                UAE_CITY_COORDS,
                key=lambda c: (UAE_CITY_COORDS[c][0]-lat)**2 + (UAE_CITY_COORDS[c][1]-lon)**2
            )
            city = nearest if country == "AE" else "Sharjah"
            st.session_state.detected_city = city
            return city
    except Exception:
        pass
    st.session_state.detected_city = "Sharjah"
    return "Sharjah"


def heat_label(temp: int) -> tuple:
    if temp >= 45: return "CRITICAL", "alert-critical"
    if temp >= 40: return "EXTREME",  "alert-critical"
    if temp >= 35: return "HIGH",     "alert-elevated"
    return "NORMAL", "alert-normal"


def build_weather_context(city: str) -> str:
    w = st.session_state.weather_data.get(city, {})
    if not w:
        return ""
    label, _ = heat_label(w.get("temp", 30))
    month = datetime.now().month
    if   month in [6, 7, 8, 9]: season = "Peak UAE summer"
    elif month in [3, 4, 5]:    season = "UAE spring / sandstorm season"
    elif month in [10, 11]:     season = "UAE transitional season"
    else:                        season = "UAE cool / planting season"
    return (
        f"\n=== CURRENT UAE WEATHER ({city}) ===\n"
        f"Temp: {w.get('temp')}°C | Humidity: {w.get('humidity')}% | "
        f"Conditions: {w.get('desc')} | Wind: {w.get('wind')} km/h\n"
        f"Season: {season} | Heat Alert: {label}\n"
        f"Factor these into your diagnosis and watering advice.\n"
        f"=====================================\n"
    )


# ─────────────────────────────────────────────────────────────
# DIAGNOSIS ENGINE
# ─────────────────────────────────────────────────────────────
def safe_call(func, *args, max_retries=3, wait_seconds=15, **kwargs):
    """Retry wrapper for rate-limit errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                if attempt < max_retries:
                    time.sleep(wait_seconds)
                else:
                    return None
            else:
                return None
    return None


def run_diagnosis(user_message: str, image_b64: str | None, city: str) -> tuple[str, str]:
    """
    Returns (diagnosis_text, engine_used) where engine_used is one of:
    'groq-vision', 'groq-text', 'error'

    IMAGE FIX: image is passed as base64 JPEG via inline_data Part dict,
    which is the correct format for Groq Vision API.
    """
    # Rate-limit cooldown
    elapsed = time.time() - st.session_state.last_call_time
    if elapsed < 3:
        time.sleep(3 - elapsed)

    # Pass all cities weather + highlight user's detected city
    weather_ctx = "\n".join(
        build_weather_context(c) for c in ["Dubai", "Abu Dhabi", "Sharjah", "Ajman"]
        if st.session_state.weather_data.get(c)
    )
    if city:
        weather_ctx = f"User location: {city}\n" + weather_ctx

    # ── Groq Vision path (image present) ────────────────────────────────────────────
    # Uses Llama-4 via the existing Groq client — same key, no new secrets.
    # Two vision models as fallback: Scout (faster) → Maverick (higher quality).
    if image_b64 is not None:
        GROQ_VISION_MODELS = [
            "meta-llama/llama-4-maverick-17b-128e-instruct",  # higher quality — primary
            "meta-llama/llama-4-scout-17b-16e-instruct",      # faster — fallback
        ]

        _user_req = user_message.strip() or "Please identify this plant and assess its health."
        vision_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{weather_ctx}\n\n"
            "IMAGE ANALYSIS — FOLLOW THESE STEPS IN ORDER:\n"
            "STEP 1 — DESCRIBE WHAT YOU SEE (before naming anything):\n"
            "  • Leaf colour(s), shape, size, texture, edges\n"
            "  • Stem colour and structure\n"
            "  • Any flowers, fruit, or berries visible\n"
            "  • Overall growth habit (bushy, trailing, upright, rosette)\n"
            "  • Any visible symptoms: yellowing, spots, wilting, pests, salt crust\n\n"
            "STEP 2 — IDENTIFY THE PLANT:\n"
            "  • Based ONLY on your visual description above\n"
            "  • State confidence: 'I am X% confident this is [plant]'\n"
            "  • If unsure, give top 2 options with confidence %\n"
            "  • If image is blurry, say so and ask for a clearer photo\n\n"
            "STEP 3 — HEALTH ASSESSMENT:\n"
            "  • Only flag symptoms you can ACTUALLY SEE in the image\n"
            "  • If the plant looks green, firm, and normal → it IS healthy\n"
            "  • DO NOT assign Moderate/Severe unless clear damage is visible\n"
            "  • DEFAULT: if no clear symptom → ✅ Healthy\n\n"
            "STEP 4 — RESPOND in Stage 1 format (short assessment + invite follow-up)\n\n"
            f"User request: {_user_req}"
        )

        # Groq uses OpenAI-compatible format: image as data URL in image_url block
        data_url      = f"data:image/jpeg;base64,{image_b64}"
        vision_messages = [{
            "role"   : "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text",      "text": vision_prompt},
            ],
        }]

        last_err      = None
        model_results = []

        for _model in GROQ_VISION_MODELS:
            try:
                resp = groq_client.chat.completions.create(
                    model      = _model,
                    messages   = vision_messages,
                    max_tokens = 1200,
                )
                st.session_state.last_call_time = time.time()
                result_text = resp.choices[0].message.content
                if not result_text or not result_text.strip():
                    raise ValueError(f"{_model} returned empty response")
                return result_text, f"groq-vision ({_model.split('/')[-1]})"

            except Exception as _e:
                last_err  = _e
                err_str   = str(_e)
                err_lower = err_str.lower()
                is_quota  = (
                    "429"   in err_str
                    or "rate" in err_lower
                    or "quota" in err_lower
                    or "resource_exhausted" in err_lower
                )
                short = "rate limited" if is_quota else err_str[:60]
                model_results.append(f"{_model.split('/')[-1]}: {short}")
                continue   # always try next model

        # Both models failed
        err_str  = str(last_err) if last_err else "Both Groq vision models failed"
        tried    = " | ".join(model_results)
        return (
            f"Image analysis failed. Models tried: {tried}\n\n"
            f"Error: {err_str[:200]}\n\n"
            "Please try again in a moment (rate limit) or type a text description "
            "of your plant problem and Send without the image for a text-only diagnosis.",
            "error"
        )


    # ── Groq text path ────────────────────────────────────────
    if not user_message.strip():
        return "Please enter a message or upload a plant image to begin.", "none"

    enriched = f"{weather_ctx}\nUser: {user_message}"
    st.session_state.groq_history.append({"role": "user", "content": enriched})

    resp = safe_call(
        groq_client.chat.completions.create,
        model       = GROQ_MODEL,
        messages    = [{"role": "system", "content": SYSTEM_PROMPT}]
                      + st.session_state.groq_history,
        max_tokens  = 1024,
        temperature = 0.7,
    )

    if resp is None:
        st.session_state.groq_history.pop()
        return "Could not get a response. Please wait a moment and try again.", "error"

    reply = resp.choices[0].message.content
    st.session_state.groq_history.append({"role": "assistant", "content": reply})
    st.session_state.last_call_time = time.time()

    engine = "groq-text-no-vision" if image_b64 else "groq-text"
    return reply, engine


# ─────────────────────────────────────────────────────────────
# LOAD WEATHER ON STARTUP
# ─────────────────────────────────────────────────────────────
if not st.session_state.weather_data:
    for city in UAE_CITIES:
        st.session_state.weather_data[city] = get_weather(city)



# ─────────────────────────────────────────────────────────────
# ██  NEW CHAT UI CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&display=swap');

:root {
    --ga-green      : #2d5a1b;
    --ga-green-mid  : #3d7a25;
    --ga-green-light: #c8e6c0;
    --ga-green-pale : #edf7e9;
    --ga-cream      : #faf6ec;
    --ga-border     : #d4e8cc;
}

html, body, [class*="css"] {
    font-family : 'DM Sans', sans-serif !important;
    background  : var(--ga-cream) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width    : 780px !important;
    padding      : 0 !important;
    margin       : 0 auto !important;
}

[data-testid="stAppViewContainer"] { background: var(--ga-cream); }

/* ── App shell ──────────────────────────────────────────── */
.ga-shell {
    max-width     : 780px;
    margin        : 0 auto;
    border        : 0.5px solid var(--ga-border);
    border-radius : 16px;
    overflow      : hidden;
    display       : flex;
    flex-direction: column;
    background    : var(--ga-cream);
}

/* ── Header ─────────────────────────────────────────────── */
.ga-header {
    background  : var(--ga-green);
    padding     : 12px 16px;
    display     : flex;
    align-items : center;
    gap         : 10px;
}
.ga-header-icon {
    width: 34px; height: 34px;
    background: rgba(255,255,255,0.15);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.ga-header-text { flex: 1; }
.ga-header-title {
    font-size: 14px; font-weight: 500;
    color: #edf7e9; line-height: 1.3;
}
.ga-header-arabic { font-size: 16px; color: #a8d5a0; margin-top: 2px; }
.ga-loc-pill {
    background: rgba(255,255,255,0.12);
    border: 0.5px solid rgba(255,255,255,0.22);
    border-radius: 20px;
    padding: 4px 11px;
    font-size: 11px; color: #c8e6c0;
    display: flex; align-items: center; gap: 4px;
    white-space: nowrap;
}

/* ── Weather bar ─────────────────────────────────────────── */
.ga-weather {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    background: var(--ga-green-light);
    border-bottom: 0.5px solid #a8c8a0;
}
.ga-wc {
    padding: 7px 4px;
    text-align: center;
    border-right: 0.5px solid #a8c8a0;
}
.ga-wc:last-child { border-right: none; }
.ga-wc.active { background: var(--ga-green-pale); }
.ga-wc-name { font-size: 10px; font-weight: 500; color: var(--ga-green); }
.ga-wc-temp { font-size: 16px; font-weight: 500; color: #1a3d0f; }
.ga-wc-desc { font-size: 9.5px; color: #4e7a3a; }

/* ── Chat thread ─────────────────────────────────────────── */
.ga-thread {
    padding     : 16px 14px;
    display     : flex;
    flex-direction: column;
    gap         : 10px;
    min-height  : 380px;
    background  : var(--ga-cream);
}

/* ── Bubbles ─────────────────────────────────────────────── */
.ga-row { display: flex; gap: 8px; align-items: flex-end; }
.ga-row.user { flex-direction: row-reverse; }

.ga-avatar {
    width: 28px; height: 28px; border-radius: 50%;
    background: var(--ga-green-light);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
    color: var(--ga-green);
}
.ga-avatar.user-av {
    background: #e8e8e8; font-size: 11px; font-weight: 500; color: #555;
}

.ga-bubble {
    max-width: 74%;
    border-radius: 16px;
    padding: 10px 14px;
    font-size: 13.5px;
    line-height: 1.7;
    color: var(--color-text-primary, #1a1a1a);
    white-space: pre-wrap;
    word-break: break-word;
}
.ga-bubble.advisor {
    background: #ffffff;
    border: 0.5px solid var(--ga-border);
    border-bottom-left-radius: 4px;
}
.ga-bubble.user {
    background: var(--ga-green);
    color: #edf7e9;
    border-bottom-right-radius: 4px;
}
.ga-bubble.img-bubble {
    padding: 6px;
    background: #ffffff;
    border: 0.5px solid var(--ga-border);
    border-bottom-right-radius: 4px;
}
.ga-img-thumb {
    border-radius: 10px;
    overflow: hidden;
    max-width: 180px;
}
.ga-img-thumb img { width: 100%; display: block; }
.ga-img-cap {
    font-size: 11px; color: #888;
    padding: 3px 6px 1px;
}

/* ── Quick reply chips ───────────────────────────────────── */
.ga-chips {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-top: 6px; margin-left: 36px;
}
.ga-chip {
    background: #ffffff;
    border: 0.5px solid var(--ga-green-mid);
    border-radius: 16px;
    padding: 5px 13px;
    font-size: 12px;
    color: var(--ga-green);
    cursor: pointer;
    font-family: 'DM Sans', sans-serif;
    transition: background 0.15s;
    text-decoration: none;
    display: inline-block;
}
.ga-chip:hover { background: var(--ga-green-pale); }

/* ── Timestamp ───────────────────────────────────────────── */
.ga-ts {
    text-align: center;
    font-size: 10.5px;
    color: #999;
    margin: 2px 0;
}

/* ── Typing indicator ────────────────────────────────────── */
.ga-typing { display: flex; gap: 4px; align-items: center; padding: 2px 0; }
.ga-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--ga-green-mid);
    animation: gadot 1.2s ease-in-out infinite;
}
.ga-dot:nth-child(2) { animation-delay: 0.2s; }
.ga-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes gadot {
    0%,80%,100% { transform: translateY(0); opacity: 0.35; }
    40% { transform: translateY(-5px); opacity: 1; }
}

/* ── Divider before input ────────────────────────────────── */
.ga-divider { height: 0.5px; background: var(--ga-border); }

/* ── Footer ─────────────────────────────────────────────── */
.ga-footer {
    background: var(--ga-green-pale);
    border-top: 0.5px solid var(--ga-border);
    padding: 8px 16px;
    text-align: center;
    font-size: 11.5px;
    color: #4e7a3a;
}
.ga-footer a { color: var(--ga-green); font-weight: 500; }
.ga-footer-sub { font-size: 10.5px; color: #7ab870; margin-top: 2px; }

/* ── Streamlit widget resets ─────────────────────────────── */
.stButton > button {
    background    : var(--ga-green) !important;
    color         : #edf7e9 !important;
    border        : none !important;
    border-radius : 22px !important;
    font-size     : 13px !important;
    font-family   : 'DM Sans', sans-serif !important;
    padding       : 8px 20px !important;
    width         : 100% !important;
    font-weight   : 500 !important;
    transition    : background 0.15s !important;
}
.stButton > button:hover { background: var(--ga-green-mid) !important; }

.stTextArea textarea {
    background    : #ffffff !important;
    border        : 0.5px solid var(--ga-border) !important;
    border-radius : 12px !important;
    font-size     : 13px !important;
    font-family   : 'DM Sans', sans-serif !important;
    padding       : 10px 14px !important;
    resize        : none !important;
    color         : #1a1a1a !important;
}
.stTextArea textarea:focus {
    border-color: var(--ga-green-mid) !important;
    box-shadow  : 0 0 0 2px rgba(61,122,37,0.1) !important;
}

[data-testid="stFileUploader"] {
    background    : var(--ga-green-pale) !important;
    border        : 0.5px dashed var(--ga-green-mid) !important;
    border-radius : 10px !important;
    padding       : 6px !important;
}

[data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)


import html as _html
import datetime as _dt


# ─────────────────────────────────────────────────────────────
# ██  HELPERS — render chat thread HTML
# ─────────────────────────────────────────────────────────────
def _bubble_advisor(text: str) -> str:
    safe = _html.escape(text).replace("\n", "<br>")
    return f"""
    <div class='ga-row'>
      <div class='ga-avatar'>🌿</div>
      <div class='ga-bubble advisor'>{safe}</div>
    </div>"""


def _bubble_user(text: str) -> str:
    safe = _html.escape(text).replace("\n", "<br>")
    return f"""
    <div class='ga-row user'>
      <div class='ga-bubble user'>{safe}</div>
      <div class='ga-avatar user-av'>You</div>
    </div>"""


def _bubble_image(name: str, pil_img) -> str:
    import base64, io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"""
    <div class='ga-row user'>
      <div class='ga-bubble img-bubble'>
        <div class='ga-img-thumb'>
          <img src='data:image/jpeg;base64,{b64}' alt='Uploaded plant photo'>
        </div>
        <div class='ga-img-cap'>📎 {_html.escape(name)}</div>
      </div>
      <div class='ga-avatar user-av'>You</div>
    </div>"""


def _chips(options: list) -> str:
    """Store chips for rendering as Streamlit buttons below thread."""
    st.session_state["_pending_chips"] = options
    return ""





def render_thread() -> str:
    html = "<div class='ga-thread'>"
    html += "<div class='ga-ts'>Today</div>"

    # Welcome message if no history
    if not st.session_state.chat_history:
        html += _bubble_advisor(
            "Ahlan! 🌿 I'm GreenAdvisor — your UAE plant care expert.\n\n"
            "Send me a photo of your plant or describe what's happening. "
            "I'll identify it and help it thrive in the UAE climate."
        )
        html += _chips(["My leaves are yellowing", "Watering schedule help", "Identify my plant", "Best plants for UAE heat"])
    else:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            if role == "user":
                html += _bubble_user(msg["text"])
            elif role == "image":
                if msg.get("pil"):
                    html += _bubble_image(msg["name"], msg["pil"])
            elif role == "advisor":
                html += _bubble_advisor(msg["text"])
                if msg.get("chips"):
                    html += _chips(msg["chips"])

    html += "</div>"
    return html


def extract_chips(response_text: str) -> list:
    """Extract follow-up chip suggestions from advisor response."""
    chips = []
    text_lower = response_text.lower()
    if "treatment" in text_lower or "plan" in text_lower:
        chips.append("Give me the full treatment plan")
    if "water" in text_lower:
        chips.append("Watering schedule for UAE")
    if "product" in text_lower or "buy" in text_lower or "store" in text_lower:
        chips.append("Where to buy in UAE?")
    if "fungal" in text_lower or "pest" in text_lower or "disease" in text_lower:
        chips.append("How to prevent this in future?")
    if not chips:
        chips = ["Tell me more", "Is this serious?", "What products do I need?"]
    return chips[:3]


# ─────────────────────────────────────────────────────────────
# ██  HEADER
# ─────────────────────────────────────────────────────────────
detected_city = detect_user_city()

st.markdown(f"""
<div class='ga-shell'>
  <div class='ga-header'>
    <div class='ga-header-icon'>🌿</div>
    <div class='ga-header-text'>
      <div class='ga-header-title'>GreenAdvisor — UAE Plant Care</div>
      <div class='ga-header-arabic'>خبير العناية بالنباتات في الإمارات</div>
    </div>
    <div class='ga-loc-pill'>📍 {detected_city}</div>
  </div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  WEATHER BAR
# ─────────────────────────────────────────────────────────────
city_names = list(UAE_CITIES.keys())
weather_html = "<div class='ga-weather'>"
for city in city_names:
    w     = st.session_state.weather_data.get(city, {})
    temp  = w.get("temp", "--")
    desc  = w.get("desc", "…")
    hum   = w.get("humidity", "--")
    active = " active" if city == detected_city else ""
    marker = " 📍" if city == detected_city else ""
    weather_html += f"""
    <div class='ga-wc{active}'>
      <div class='ga-wc-name'>{city}{marker}</div>
      <div class='ga-wc-temp'>{temp}°C</div>
      <div class='ga-wc-desc'>{desc}</div>
    </div>"""
weather_html += "</div>"
st.markdown(weather_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  CHAT THREAD
# ─────────────────────────────────────────────────────────────
st.markdown(render_thread(), unsafe_allow_html=True)
st.markdown("<div class='ga-divider'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  QUICK REPLY CHIPS
# ─────────────────────────────────────────────────────────────
_wc = st.session_state.widget_counter
_chips_to_show = st.session_state.get("_pending_chips", [])
if _chips_to_show:
    _chip_cols = st.columns(min(len(_chips_to_show), 3))
    for _ci, (_chip_col, _chip_txt) in enumerate(zip(_chip_cols, _chips_to_show)):
        with _chip_col:
            if st.button(_chip_txt, key=f"chip_{_ci}_{_wc}"):
                st.session_state.chat_history.append({"role": "user", "text": _chip_txt})
                st.session_state["_pending_chips"] = []
                with st.spinner("🌿 Thinking…"):
                    _cd, _ce = run_diagnosis(_chip_txt, None, detected_city)
                _cchips = extract_chips(_cd)
                st.session_state.chat_history.append({"role": "advisor", "text": _cd, "chips": _cchips})
                st.rerun()


# ─────────────────────────────────────────────────────────────
# ██  INPUT AREA
# ─────────────────────────────────────────────────────────────


# Image attachment row (shown above input when image is pending)
if st.session_state.pending_image is not None:
    att_col1, att_col2 = st.columns([5, 1])
    with att_col1:
        st.markdown(
            f"<div style='background:#edf7e9;border:0.5px solid #a8c8a0;border-radius:10px;"
            f"padding:6px 12px;font-size:12px;color:#2d5a1b;margin:6px 0'>"
            f"📎 {_html.escape(st.session_state.pending_img_name or '')}</div>",
            unsafe_allow_html=True,
        )
    with att_col2:
        if st.button("✕ Remove", key=f"rm_{_wc}"):
            st.session_state.pending_image    = None
            st.session_state.pending_img_name = None
            st.session_state.pending_img_b64  = None
            st.session_state.widget_counter  += 1
            st.rerun()

# Text input + buttons
input_col, btn_col = st.columns([5, 1])
with input_col:
    user_input = st.text_area(
        "msg",
        placeholder="Ask about your plant...",
        height=80,
        label_visibility="collapsed",
        key=f"user_text_{_wc}",
    )

with btn_col:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    send_clicked  = st.button("Send 💬", key=f"send_{_wc}")
    clear_clicked = st.button("Clear ✕", key=f"clr_{_wc}")

# File uploader (compact, below input)
uploaded_file = st.file_uploader(
    "📎 Attach plant photo",
    type=["jpg", "jpeg", "png", "webp"],
    key=f"plant_image_{_wc}",
    label_visibility="visible",
)

# Process new upload (only when filename changes)
if uploaded_file is not None:
    import io as _io
    new_name = uploaded_file.name
    if new_name != st.session_state.get("pending_img_name"):
        _img_bytes = uploaded_file.read()
        _pil_img   = Image.open(_io.BytesIO(_img_bytes))
        st.session_state.pending_image    = _pil_img
        st.session_state.pending_img_name = new_name
        st.session_state.pending_img_b64  = pil_to_base64_jpeg(_pil_img)


# ─────────────────────────────────────────────────────────────
# ██  FOOTER (close ga-shell + footer bar)
# ─────────────────────────────────────────────────────────────
st.markdown("""
  <div class='ga-footer'>
    🌿 GreenAdvisor &nbsp;—&nbsp; Powered by Groq (Llama-4 Vision) &nbsp;|&nbsp;
    Built for UAE plant lovers &nbsp;|&nbsp;
    <a href='https://wa.me/971000000000' target='_blank'>
      Contact Toronto Agriculture
    </a>
    <div class='ga-footer-sub'>
      Built by Sundas Khan &nbsp;·&nbsp;
      Groq llama-3.3-70b · llama-4-maverick · Open-Meteo · Streamlit
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  BUTTON HANDLERS
# ─────────────────────────────────────────────────────────────
# Snapshot image BEFORE widgets cleared (safe to read here)
_img_b64_now = st.session_state.pending_img_b64

if send_clicked:
    msg     = (user_input or "").strip()
    img_b64 = _img_b64_now

    if not msg and img_b64 is None:
        st.warning("Please type a message or attach a plant photo.")
    else:
        # Append user text bubble
        if msg:
            st.session_state.chat_history.append({"role": "user", "text": msg})

        # Append image bubble (store PIL for inline rendering)
        if img_b64 is not None:
            st.session_state.chat_history.append({
                "role" : "image",
                "name" : st.session_state.pending_img_name or "photo.jpg",
                "pil"  : st.session_state.pending_image,
            })

        with st.spinner("🌿 Analysing your plant…"):
            diagnosis, engine = run_diagnosis(msg, img_b64, detected_city)

        # Extract smart chips from the response
        chips = extract_chips(diagnosis)

        # Clean engine tag — only show if error
        tag = " ⚠️" if engine == "error" else ""
        st.session_state.chat_history.append({
            "role" : "advisor",
            "text" : diagnosis + tag,
            "chips": chips,
        })

        # Keep image in panel (do NOT clear) — only Clear/Remove clears it
        st.session_state.groq_history  # already updated inside run_diagnosis
        st.rerun()

if clear_clicked:
    st.session_state.chat_history     = []
    st.session_state.groq_history     = []
    st.session_state.pending_image    = None
    st.session_state.pending_img_name = None
    st.session_state.pending_img_b64  = None
    st.session_state.widget_counter  += 1
    st.rerun()

if st.session_state.weather_data == {}:
    for city in UAE_CITIES:
        st.session_state.weather_data[city] = get_weather(city)
    st.rerun()
