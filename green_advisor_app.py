"""
GreenAdvisor — UAE Plant Care & Landscaping Advisor
====================================================
Primary Engine : Groq  (llama-3.3-70b-versatile)  — text diagnosis
Vision Engine  : Gemini (gemini-1.5-flash)         — image analysis  ← BUG FIXED
Weather        : Open-Meteo (free, no kefy needed)  ← upgraded from OpenWeatherMap
Built by       : Sundas Khan

IMAGE BUG FIX NOTES
-------------------
Root cause 1: PIL Image object was passed directly to Gemini SDK —
              must be converted to bytes and wrapped in a Part/blob dict.
Root cause 2: Gemini 429 quota errors caused silent fallback with no
              clear status shown to the user.
Fix applied  : Image is now converted to JPEG bytes → base64 → passed
              as an inline_data Part, which is the correct Gemini SDK
              format and works reliably on the free tier.
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
from google import genai

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
    font-size : 13px;
    color     : #a8d5a0;
    margin    : 2px 0 0;
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
You are GreenAdvisor — a professional plant care and landscaping expert
specializing in the UAE climate. You serve homeowners, gardeners, villa
managers, hotel landscaping teams, and B2B clients across Dubai,
Abu Dhabi, Sharjah, and all UAE emirates.

LANGUAGE BEHAVIOR:
- Detect user language automatically.
- Respond in English if they write English, Arabic if they write Arabic.
- Mirror mixed language use naturally.
- Introduce botanical terms in both languages on first use.

PERSONALITY & TONE:
- Professional, expert, authoritative. Warm but not casual.
- Always factor UAE-specific climate challenges into advice.
- Never give generic global advice — always UAE-specific.

UAE CLIMATE CONTEXT:
- Summer (Jun-Sep): 42-48C, high humidity, intense UV
- Winter (Nov-Feb): 15-25C, low humidity, ideal planting window
- Sandstorm season (Mar-May, Sep-Oct): dust accumulates on leaves
- Hard water, high salinity (TDS 300-600 ppm)
- Sandy, low-organic, fast-draining soil

WHEN AN IMAGE IS PROVIDED:
- You have FULL vision capability. Examine the image with care.
- Identify the exact plant species from what you SEE in the image:
  leaf shape, flower colour, stem structure, growth pattern, texture.
- CRITICAL: Do NOT guess species from the user's text. Visual analysis
  is primary and overrides any text description.
- Assess visible health: discolouration, spots, wilting, pests, dryness.
- State the plant type confidently with visual reasoning.
- Provide a full structured diagnosis even if no text is given.

DIAGNOSIS FORMAT (use emojis as shown):
🌿 Plant Identified: [Name in English (Arabic)]
📍 Location: [Emirate if known]
⚠️  Problem Diagnosed: [Problem name, or Healthy if no issues]
🎯 Severity: [🟢 Mild / 🟡 Moderate / 🔴 Severe / ✅ Healthy]
🔍 Why This Happens in UAE: [1-2 sentences specific to UAE conditions]
💊 Treatment Plan:
   1. [Step]
   2. [Step]
   3. [Step]
💧 UAE Watering Schedule: [Specific times and frequency]
🛒 Products Available in UAE: [Product names + store: ACE, Lulu, Garden Centre, Plantshop.ae]
🛡️  Prevention Tip: [One actionable tip]
📅 Follow-up: Please update me in 3-5 days with progress.

TORONTO AGRICULTURE REFERRAL:
Only mention Toronto Agriculture when user explicitly asks for
professional on-site help or a site visit. When mentioned, say:
"For professional on-site landscaping, pool, and plant services across
Dubai, Abu Dhabi, and Sharjah, I recommend Toronto Agriculture —
a specialist with over a decade of UAE experience."
DO NOT mention Toronto Agriculture unprompted.
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
    "pending_img_b64" : None, # base64 JPEG string (for Gemini)
    "weather_data"    : {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


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


@st.cache_resource
def get_gemini_model():
    """
    Returns (model, status_message) tuple.
    status: 'ok' | 'no_key' | 'error'
    """
    key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not key:
        return None, "no_key"
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model, "ok"
    except Exception as e:
        return None, f"error:{e}"


groq_client          = get_groq_client()
gemini_model, gem_status = get_gemini_model()


# ─────────────────────────────────────────────────────────────
# IMAGE HELPER — PIL → base64 JPEG
# ─────────────────────────────────────────────────────────────
def pil_to_base64_jpeg(pil_img: Image.Image, max_size: int = 1024) -> str:
    """
    Converts PIL Image to base64-encoded JPEG string.
    Resizes to max_size on the longest side to keep payload small.
    This is the CORRECT format for Gemini inline_data parts.
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
    'gemini-vision', 'groq-text', 'groq-text-no-vision', 'error'

    IMAGE FIX: image is passed as base64 JPEG via inline_data Part dict,
    which is the correct and reliable Gemini SDK format.
    """
    # Rate-limit cooldown
    elapsed = time.time() - st.session_state.last_call_time
    if elapsed < 3:
        time.sleep(3 - elapsed)

    weather_ctx = build_weather_context(city)

    # ── Gemini Vision path (image present + key available) ────
    if image_b64 is not None and gemini_model is not None:
        vision_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{weather_ctx}\n\n"
            "CRITICAL IMAGE ANALYSIS INSTRUCTION:\n"
            "1. Examine the uploaded plant image carefully.\n"
            "2. Identify the exact species from VISUAL features ONLY:\n"
            "   - Leaf shape, size, texture, colour\n"
            "   - Flower colour, shape, petal count if visible\n"
            "   - Stem structure and growth pattern\n"
            "3. Do NOT guess species from user text. Visual analysis is primary.\n"
            "4. Assess visible health: discolouration, spots, wilting, pests, damage.\n"
            "5. Follow the structured diagnosis format with emojis.\n\n"
            f"User request: {user_message if user_message.strip() else 'Please identify this plant and assess its health condition in UAE conditions.'}"
        )

        # ── FIXED: Use inline_data Part dict (correct Gemini format) ──
        image_part = {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data"     : image_b64,
            }
        }

        try:
          resp = gemini_model.generate_content(
            contents=[{
              "role": "user",
              "parts": [
                {
                  "inline_data": {"mime_type": "image/jpeg","data": image_b64}
                },
                {
                  "text": vision_prompt
                }
              ]
            }]
          )
          st.session_state.last_call_time = time.time()
          return resp.text, "gemini-vision"
        except Exception as e:
            err_str = str(e)
            # Quota exhausted — fall through to Groq with note
            fallback_note = (
                f"[Gemini Vision unavailable: {err_str[:80]}. "
                f"Proceeding with text-based diagnosis.]\n\n"
            )
            user_message = fallback_note + user_message

    # ── Gemini key missing but image uploaded ─────────────────
    elif image_b64 is not None and gemini_model is None:
        user_message = (
            "[Image uploaded but GEMINI_API_KEY is not configured — "
            "add it to Streamlit Secrets for visual diagnosis. "
            f"Diagnosing from text only.]\n\n{user_message}"
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
# ██  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='ga-header'>
    <div class='ga-header-icon'>🌿</div>
    <div class='ga-header-text'>
        <div class='ga-header-title'>
            Professional UAE Plant Care &amp; Landscaping Advisor
        </div>
        <div class='ga-header-arabic'>خبير العناية بالنباتات في الإمارات</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  WEATHER GRID
# ─────────────────────────────────────────────────────────────
city_names = list(UAE_CITIES.keys())

header_html = "<div class='weather-grid'>"
for city in city_names:
    header_html += f"<div class='weather-city-header'>{city}</div>"
header_html += "</div><div class='weather-grid'>"
for city in city_names:
    w     = st.session_state.weather_data.get(city, {})
    temp  = w.get("temp", "--")
    desc  = w.get("desc", "Loading...")
    hum   = w.get("humidity", "--")
    wind  = w.get("wind", "--")
    live  = w.get("live", False)
    label, css = heat_label(temp) if isinstance(temp, int) else ("--", "alert-normal")
    tag   = "🔴" if "critical" in css else "🟡" if "elevated" in css else "🟢"
    note  = "" if live else "<br><span style='font-size:9px;opacity:0.6'>(estimated)</span>"
    header_html += f"""
    <div class='weather-city-body'>
        <div class='weather-temp'>{temp}°C</div>
        <div class='weather-desc'>{desc}{note}</div>
        <div class='weather-hum'>
            💧{hum}% &middot; 💨{wind}km/h &middot;
            <span class='{css}'>{tag} {label}</span>
        </div>
    </div>"""
header_html += "</div>"
st.markdown(header_html, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  MAIN LAYOUT — left (message + results) | right (image)
# ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 1.4], gap="medium")


# ── LEFT PANEL ────────────────────────────────────────────────
with left_col:

    selected_city = st.selectbox(
        "City",
        options=city_names,
        index=0,
        label_visibility="collapsed",
        key="city_select",
    )

    st.markdown("<div class='section-label'>User Message</div>",
                unsafe_allow_html=True)
    user_input = st.text_area(
        "msg",
        placeholder=(
            "Describe your plant problem here...\n"
            "e.g. 'My bougainvillea leaves are turning yellow in Dubai'\n\n"
            "أو اكتب بالعربية: نبتتي تذبل، ماذا أفعل؟\n\n"
            "Tip: Upload a photo — GreenAdvisor will identify the plant automatically."
        ),
        height=130,
        label_visibility="collapsed",
        key="user_text",
    )

    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        send_clicked    = st.button("Send 💬", use_container_width=True)
    with b2:
        clear_clicked   = st.button("Clear ✕", use_container_width=True)
    with b3:
        refresh_clicked = st.button("🔄 Weather", use_container_width=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Results / Chat history ────────────────────────────────
    st.markdown("<div class='section-label'>Analysis Result / Findings</div>",
                unsafe_allow_html=True)

    if st.session_state.chat_history:
        history_html = "<div class='result-box'>"
        for msg in st.session_state.chat_history:
            role  = msg["role"]
            text  = msg["text"]
            label = "You" if role == "user" else "🌿 GreenAdvisor"
            css   = "chat-msg-user" if role == "user" else "chat-msg-advisor"
            import html as html_lib
            safe_text = html_lib.escape(text).replace("\n", "<br>")
            history_html += f"""
            <div class='chat-wrap'>
                <div class='chat-label'>{label}</div>
                <div class='{css}'>{safe_text}</div>
            </div>"""
        history_html += "</div>"
        st.markdown(history_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-box'>
            <div class='result-empty'>
                🌱 Your plant diagnosis will appear here.<br><br>
                Type a message or upload a plant photo and click Send.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── RIGHT PANEL ───────────────────────────────────────────────
with right_col:

    # Vision status badge
    if gem_status == "ok":
        badge = "<div class='vision-badge vision-ok'>✅ Vision active</div>"
    elif gem_status == "no_key":
        badge = "<div class='vision-badge vision-warn'>⚠️ Add GEMINI_API_KEY</div>"
    else:
        badge = "<div class='vision-badge vision-info'>⚠️ Gemini error</div>"
    st.markdown(badge, unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Upload Plant Image</div>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "img",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="plant_image",
    )

    # Convert uploaded file to PIL + base64 immediately
    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes))
        st.session_state.pending_image    = pil_img
        st.session_state.pending_img_name = uploaded_file.name
        st.session_state.pending_img_b64  = pil_to_base64_jpeg(pil_img)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Uploaded Image</div>",
                unsafe_allow_html=True)

    if st.session_state.pending_image is not None:
        st.image(
            st.session_state.pending_image,
            caption=f"📸 {st.session_state.pending_img_name}",
            use_container_width=True,
        )
        if gem_status == "ok":
            st.markdown(
                "<div style='font-size:11px;color:#3d7a25;text-align:center;margin-top:4px'>"
                "Image ready for visual analysis</div>",
                unsafe_allow_html=True,
            )
        if st.button("Remove Image ✕", use_container_width=True):
            st.session_state.pending_image    = None
            st.session_state.pending_img_name = None
            st.session_state.pending_img_b64  = None
            st.rerun()
    else:
        st.markdown("""
        <div class='image-preview-box'>
            <div class='image-placeholder'>
                📷 No image uploaded.<br><br>
                Upload a plant photo<br>for visual AI diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  BUTTON HANDLERS
# ─────────────────────────────────────────────────────────────
if send_clicked:
    msg      = st.session_state.get("user_text", "").strip()
    img_b64  = st.session_state.pending_img_b64
    if not msg and img_b64 is None:
        st.warning("Please type a message or upload a plant image.")
    else:
        with st.spinner("🌿 GreenAdvisor is analysing your plant..."):
            diagnosis, engine = run_diagnosis(msg, img_b64, selected_city)

        # Append to visible chat history
        if msg:
            st.session_state.chat_history.append({"role": "user", "text": msg})
        elif img_b64:
            st.session_state.chat_history.append({"role": "user", "text": "[Plant image uploaded for diagnosis]"})

        # Engine tag for transparency
        engine_tag = {
            "gemini-vision"      : " [Gemini Vision]",
            "groq-text"          : " [Groq Text]",
            "groq-text-no-vision": " [Groq Text — image not analysed]",
            "error"              : " [Error]",
        }.get(engine, "")

        full_reply = diagnosis + f"\n\n— GreenAdvisor{engine_tag}"
        st.session_state.chat_history.append({"role": "advisor", "text": full_reply})

        # Clear image after send
        st.session_state.pending_image    = None
        st.session_state.pending_img_name = None
        st.session_state.pending_img_b64  = None
        st.rerun()

if clear_clicked:
    st.session_state.chat_history     = []
    st.session_state.groq_history     = []
    st.session_state.pending_image    = None
    st.session_state.pending_img_name = None
    st.session_state.pending_img_b64  = None
    st.rerun()

if refresh_clicked:
    for city in UAE_CITIES:
        st.session_state.weather_data[city] = get_weather(city)
    st.rerun()


# ─────────────────────────────────────────────────────────────
# ██  FOOTER
# ─────────────────────────────────────────────────────────────
vision_status_str = (
    "✅ Gemini Vision active" if gem_status == "ok"
    else "⚠️ Text only — add GEMINI_API_KEY to Secrets"
)
st.markdown(f"""
<div class='ga-footer-1'>
    🌿 GreenAdvisor &nbsp;—&nbsp; Powered by Groq &amp; Gemini &nbsp;|&nbsp;
    Built for UAE plant lovers &nbsp;|&nbsp; {vision_status_str} &nbsp;|&nbsp;
    <a href='https://wa.me/971000000000' target='_blank'>
        Contact Toronto Agriculture for professional help
    </a>
</div>
<div class='ga-footer-2'>
    Built By: Sundas Khan &nbsp;|&nbsp;
    Tech Stack: Groq (llama-3.3-70b) &middot; Gemini Vision &middot;
    Open-Meteo &middot; Streamlit
</div>
""", unsafe_allow_html=True)
