"""
GreenAdvisor — UAE Plant Care & Landscaping Advisor
Streamlit Web App | Phase 4 Deployment
Primary Engine : Groq (Free, Text Diagnosis)
Vision Engine  : Gemini (Free, Image Diagnosis)
"""

import time
import os
import io
import requests
from datetime import datetime

import streamlit as st
from groq import Groq
from PIL import Image
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "GreenAdvisor — UAE Plant Care",
    page_icon   = "🌿",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ─────────────────────────── */
:root {
    --green-deep    : #1a3c2e;
    --green-mid     : #2d6a4f;
    --green-bright  : #52b788;
    --green-light   : #d8f3dc;
    --green-pale    : #f0f7f3;
    --amber         : #e9c46a;
    --coral         : #e76f51;
    --sand          : #fefae0;
    --text-dark     : #1a2e22;
    --text-mid      : #4a6358;
    --text-light    : #8aab98;
}

/* ── Global ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family : 'DM Sans', sans-serif;
    color       : var(--text-dark);
}

/* ── Hide Streamlit chrome ──────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── App background ─────────────────────────── */
.stApp {
    background: linear-gradient(160deg, #f0f7f3 0%, #fefae0 50%, #f0f7f3 100%);
}

/* ── Header banner ──────────────────────────── */
.ga-hero {
    background  : linear-gradient(135deg, var(--green-deep) 0%, var(--green-mid) 60%, var(--green-bright) 100%);
    border-radius: 16px;
    padding     : 28px 32px;
    margin-bottom: 20px;
    position    : relative;
    overflow    : hidden;
}
.ga-hero::before {
    content     : "🌴🌵🌿🌺🪴";
    position    : absolute;
    right       : 24px;
    top         : 50%;
    transform   : translateY(-50%);
    font-size   : 28px;
    letter-spacing: 6px;
    opacity     : 0.4;
}
.ga-hero h1 {
    font-family : 'DM Serif Display', serif;
    font-size   : 36px;
    color       : white;
    margin      : 0 0 6px;
    letter-spacing: -0.5px;
}
.ga-hero p {
    color       : rgba(255,255,255,0.82);
    font-size   : 15px;
    margin      : 0;
    font-weight : 300;
}

/* ── Weather card ───────────────────────────── */
.weather-card {
    background  : white;
    border      : 1px solid #c3e6cb;
    border-radius: 12px;
    padding     : 16px 20px;
    margin-bottom: 16px;
}
.weather-card.critical {
    border-color: var(--coral);
    background  : #fff5f3;
}
.weather-card.elevated {
    border-color: var(--amber);
    background  : #fffdf0;
}
.weather-title {
    font-family : 'DM Serif Display', serif;
    font-size   : 17px;
    color       : var(--green-deep);
    margin-bottom: 10px;
}
.weather-grid {
    display     : grid;
    grid-template-columns: repeat(3, 1fr);
    gap         : 8px;
}
.weather-metric {
    background  : var(--green-pale);
    border-radius: 8px;
    padding     : 8px 12px;
    text-align  : center;
}
.weather-metric .val {
    font-size   : 22px;
    font-weight : 600;
    color       : var(--green-mid);
    display     : block;
}
.weather-metric .lbl {
    font-size   : 11px;
    color       : var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.alert-pill {
    display     : inline-block;
    padding     : 4px 10px;
    border-radius: 20px;
    font-size   : 12px;
    margin      : 4px 4px 0 0;
    background  : var(--green-light);
    color       : var(--green-deep);
    font-weight : 500;
}

/* ── Chat messages ──────────────────────────── */
.msg-wrap-user {
    display         : flex;
    justify-content : flex-end;
    margin          : 10px 0;
}
.msg-wrap-agent {
    display         : flex;
    justify-content : flex-start;
    margin          : 10px 0;
}
.bubble-user {
    background      : linear-gradient(135deg, var(--green-mid), var(--green-bright));
    color           : white;
    border-radius   : 18px 18px 4px 18px;
    padding         : 12px 16px;
    max-width       : 72%;
    font-size       : 14px;
    line-height     : 1.6;
    box-shadow      : 0 2px 8px rgba(45,106,79,0.2);
}
.bubble-agent {
    background      : white;
    color           : var(--text-dark);
    border-radius   : 18px 18px 18px 4px;
    padding         : 14px 18px;
    max-width       : 78%;
    font-size       : 13.5px;
    line-height     : 1.75;
    border          : 1px solid #ddeee6;
    box-shadow      : 0 2px 8px rgba(0,0,0,0.05);
    white-space     : normal;
}
.bubble-label {
    font-size       : 11px;
    color           : var(--text-light);
    margin          : 2px 6px;
    font-weight     : 500;
}
.thinking-bubble {
    background      : var(--green-pale);
    border          : 1px dashed var(--green-bright);
    border-radius   : 18px;
    padding         : 10px 16px;
    font-size       : 13px;
    color           : var(--green-mid);
    font-style      : italic;
    display         : inline-block;
}
.empty-chat {
    text-align      : center;
    padding         : 48px 24px;
    color           : var(--text-light);
}
.empty-chat .icon { font-size: 48px; margin-bottom: 12px; }
.empty-chat p     { font-size: 15px; margin: 0; }

/* ── Sidebar ────────────────────────────────── */
[data-testid="stSidebar"] {
    background  : var(--green-deep);
}
[data-testid="stSidebar"] * {
    color       : rgba(255,255,255,0.9) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label {
    color       : rgba(255,255,255,0.7) !important;
    font-size   : 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.sidebar-logo {
    font-family : 'DM Serif Display', serif;
    font-size   : 26px;
    color       : white !important;
    text-align  : center;
    padding     : 20px 0 8px;
}
.sidebar-sub {
    font-size   : 12px;
    color       : rgba(255,255,255,0.55) !important;
    text-align  : center;
    margin-bottom: 24px;
}
.sidebar-section {
    font-size   : 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color       : rgba(255,255,255,0.45) !important;
    margin      : 20px 0 8px;
    border-top  : 1px solid rgba(255,255,255,0.1);
    padding-top : 16px;
}
.plant-badge {
    background  : rgba(255,255,255,0.08);
    border       : 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    padding     : 6px 10px;
    font-size   : 12px;
    color       : rgba(255,255,255,0.8) !important;
    margin-bottom: 4px;
}

/* ── Input area ─────────────────────────────── */
.input-container {
    background  : white;
    border      : 1.5px solid #c3e6cb;
    border-radius: 14px;
    padding     : 16px;
    margin-top  : 12px;
    box-shadow  : 0 2px 12px rgba(45,106,79,0.08);
}

/* ── Buttons ────────────────────────────────── */
.stButton > button {
    background  : linear-gradient(135deg, var(--green-mid), var(--green-bright)) !important;
    color       : white !important;
    border      : none !important;
    border-radius: 10px !important;
    font-weight : 600 !important;
    font-size   : 14px !important;
    padding     : 8px 20px !important;
    transition  : all 0.2s ease !important;
}
.stButton > button:hover {
    transform   : translateY(-1px) !important;
    box-shadow  : 0 4px 14px rgba(45,106,79,0.35) !important;
}

/* ── Scrollable chat area ───────────────────── */
.chat-scroll {
    max-height  : 460px;
    overflow-y  : auto;
    padding     : 8px 4px;
    scroll-behavior: smooth;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

UAE_CITIES = {
    "Dubai":      "Dubai,AE",
    "Abu Dhabi":  "Abu+Dhabi,AE",
    "Sharjah":    "Sharjah,AE",
    "Ajman":      "Ajman,AE",
    "RAK":        "Ras+al-Khaimah,AE",
}

GROQ_MODEL = "llama-3.3-70b-versatile"

UAE_PLANTS = {
    "Bougainvillea":     "البوغانفيليا  ⭐⭐⭐⭐⭐",
    "Desert Rose":       "وردة الصحراء  ⭐⭐⭐⭐⭐",
    "Date Palm":         "النخيل        ⭐⭐⭐⭐⭐",
    "Jasmine":           "الياسمين      ⭐⭐⭐⭐",
    "Aloe Vera":         "الصبار        ⭐⭐⭐⭐⭐",
    "Plumeria":          "الفرنجيباني   ⭐⭐⭐⭐⭐",
    "Sansevieria":       "سانسيفيريا    ⭐⭐⭐⭐⭐",
}

SYSTEM_PROMPT = """
You are GreenAdvisor — a professional plant care and landscaping expert
specializing in the UAE climate. You serve homeowners, gardeners, villa
managers, hotel landscaping teams, and B2B clients across Dubai,
Abu Dhabi, and Sharjah.

LANGUAGE BEHAVIOR:
- Detect the user's language automatically from their first message.
- If they write in Arabic → respond fully in Arabic.
- If they write in English → respond in English.
- If they mix both → mirror their mix.
- Key botanical terms should appear in both languages when first introduced.

PERSONALITY & TONE:
- Professional, expert, and authoritative.
- Warm but not casual. Confident but not condescending.
- Always acknowledge the UAE's unique climate challenges.
- Never give generic global advice — every recommendation must be UAE-specific.

UAE CLIMATE CONTEXT:
- Summer (Jun–Sep): Extreme heat 42–48°C, high humidity, intense UV
- Winter (Nov–Feb): Mild 15–25°C, low humidity, ideal planting window
- Sandstorm season (Mar–May, Sep–Oct): Dust accumulation on leaves
- Water quality: Hard water with high salinity (TDS 300–600 ppm)
- Soil: Mostly sandy, low organic matter, fast-draining

DIAGNOSIS FORMAT — always use this exact structure:
🌿 Plant Identified: [Name in English (Arabic)]
📍 Location: [Emirate]
⚠️ Problem Diagnosed: [Problem name]
🎯 Severity: [🟢 Mild / 🟡 Moderate / 🔴 Severe]
🔍 Why This Happens in UAE: [1–2 sentences]
💊 Treatment Plan:
   1. ...
   2. ...
   3. ...
💧 UAE Watering Schedule: [Specific schedule]
🛒 Products Available in UAE: [With store names]
🛡️ Prevention Tip: [One actionable tip]
📅 Follow-up: Please update me in 3–5 days.

TORONTO AGRICULTURE REFERRAL:
Only mention Toronto Agriculture when the user explicitly asks for
professional on-site help or a site visit. When triggered:
"For professional on-site treatment and landscaping services across
Dubai, Abu Dhabi, and Sharjah, I recommend Toronto Agriculture —
a specialist plant and landscaping service operating across the UAE."
DO NOT mention Toronto Agriculture unprompted.
"""


# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────

if "messages"        not in st.session_state: st.session_state.messages        = []
if "groq_history"    not in st.session_state: st.session_state.groq_history    = []
if "weather_cache"   not in st.session_state: st.session_state.weather_cache   = {}
if "last_call_time"  not in st.session_state: st.session_state.last_call_time  = 0
if "selected_city"   not in st.session_state: st.session_state.selected_city   = "Dubai"


# ─────────────────────────────────────────────────────────────
# API CLIENTS
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Add it to Streamlit secrets.")
        st.stop()
    return Groq(api_key=api_key)

@st.cache_resource
def get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None

groq_client   = get_groq_client()
gemini_model  = get_gemini_model()


# ─────────────────────────────────────────────────────────────
# WEATHER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_uae_weather(city="Dubai"):
    """Fetch live weather. Falls back to mock data if key unavailable."""
    api_key = st.secrets.get("OPENWEATHER_API_KEY",
                             os.environ.get("OPENWEATHER_API_KEY", ""))

    # Mock fallback
    if not api_key:
        return _mock_weather(city)

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={UAE_CITIES[city]}&appid={api_key}&units=metric"
        )
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            d = r.json()
            return {
                "city":        city,
                "temp":        round(d["main"]["temp"]),
                "feels_like":  round(d["main"]["feels_like"]),
                "humidity":    d["main"]["humidity"],
                "description": d["weather"][0]["description"].capitalize(),
                "wind_speed":  round(d["wind"]["speed"] * 3.6),
                "visibility":  d.get("visibility", 10000) // 1000,
            }
    except Exception:
        pass

    return _mock_weather(city)


def _mock_weather(city):
    """Realistic mock weather based on current month."""
    month = datetime.now().month
    if month in [6, 7, 8]:    temp, hum, desc = 44, 75, "Hazy sunshine"
    elif month in [12, 1, 2]: temp, hum, desc = 22, 52, "Clear sky"
    elif month in [3, 4, 5]:  temp, hum, desc = 34, 48, "Partly cloudy"
    else:                      temp, hum, desc = 38, 65, "Sunny"

    offsets = {"Dubai": (0,0), "Abu Dhabi": (-1,-4),
               "Sharjah": (1,4), "Ajman": (0,2), "RAK": (-2,-6)}
    dt, dh = offsets.get(city, (0, 0))
    return {
        "city":        city,
        "temp":        temp + dt,
        "feels_like":  temp + dt + 3,
        "humidity":    hum + dh,
        "description": desc,
        "wind_speed":  18,
        "visibility":  8,
        "mock":        True,
    }


def interpret_weather(weather):
    """Return smart plant care alerts from weather data."""
    if not weather:
        return {"alerts": [], "urgency": "normal",
                "watering": "Water as needed.", "shade_tip": ""}

    temp, hum  = weather["temp"], weather["humidity"]
    wind, vis  = weather["wind_speed"], weather["visibility"]
    alerts     = []
    urgency    = "normal"
    watering   = ""
    shade_tip  = ""

    if temp >= 45:
        alerts.append("🔴 CRITICAL HEAT — Water twice daily, full shade mandatory")
        urgency   = "critical"
        watering  = "Water at 5:30 AM and 7:00 PM. Never midday."
        shade_tip = "Move ALL non-desert plants to full shade immediately."
    elif temp >= 40:
        alerts.append("🟠 EXTREME HEAT — High stress risk for non-desert plants")
        urgency   = "elevated"
        watering  = "Water at 6:00 AM and 6:30 PM daily."
        shade_tip = "Provide afternoon shade (12 PM–5 PM) for all non-native plants."
    elif temp >= 35:
        alerts.append("🟡 HIGH HEAT — Monitor soil moisture closely")
        urgency   = "elevated"
        watering  = "Water every morning before 8 AM."
        shade_tip = "Afternoon shade recommended for herbs and flowering plants."
    else:
        alerts.append("🟢 Comfortable conditions for UAE plants")
        watering  = "Water every 1–2 days depending on plant type."
        shade_tip = "Full sun acceptable for most UAE plants."

    if hum >= 80:
        alerts.append("💧 HIGH HUMIDITY — Fungal disease risk, improve air circulation")
    elif hum <= 25:
        alerts.append("🌵 LOW HUMIDITY — Increase misting for tropical/indoor plants")

    if vis < 3:
        alerts.append("🌫️ SEVERE DUST STORM — Move sensitive plants indoors immediately")
        urgency = "critical"
    elif vis < 7 or wind > 40:
        alerts.append("💨 DUSTY CONDITIONS — Wipe leaves after conditions clear")

    if wind > 50:
        alerts.append("🌬️ HIGH WINDS — Stake tall plants and shelter potted ones")

    return {"alerts": alerts, "urgency": urgency,
            "watering": watering, "shade_tip": shade_tip}


def build_weather_context(weather):
    """Build weather string to inject into diagnosis prompt."""
    if not weather:
        return "Weather data unavailable. Apply general UAE summer care advice."

    interp  = interpret_weather(weather)
    alerts  = "\n".join(interp["alerts"])
    month   = datetime.now().month

    if month in [6,7,8,9]:    season = "Peak UAE summer"
    elif month in [3,4,5]:    season = "UAE spring/sandstorm season"
    elif month in [10,11]:    season = "UAE transitional season"
    else:                     season = "UAE cool season"

    mock_note = " (estimated)" if weather.get("mock") else " (live)"

    return f"""
=== LIVE UAE WEATHER{mock_note} — FACTOR INTO DIAGNOSIS ===
Location    : {weather['city']}
Temperature : {weather['temp']}°C (feels like {weather['feels_like']}°C)
Humidity    : {weather['humidity']}%
Wind        : {weather['wind_speed']} km/h | Visibility: {weather['visibility']} km
Conditions  : {weather['description']}
Season      : {season} | Urgency: {interp['urgency'].upper()}

ALERTS:
{alerts}

Watering    : {interp['watering']}
Shade       : {interp['shade_tip']}

CRITICAL: Reference current temperature ({weather['temp']}°C) and
humidity ({weather['humidity']}%) directly in your diagnosis.
Adjust treatment urgency to match {interp['urgency'].upper()} level.
=======================================================
"""


# ─────────────────────────────────────────────────────────────
# DIAGNOSIS ENGINE
# ─────────────────────────────────────────────────────────────

def safe_call(func, *args, max_retries=3, wait_seconds=15, **kwargs):
    """Retry wrapper for rate limit errors."""
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


def diagnose(user_message, image=None, city="Dubai"):
    """
    Run full GreenAdvisor diagnosis.
    Routes text → Groq, image → Gemini.
    """
    # Rate limit guard
    elapsed = time.time() - st.session_state.last_call_time
    if elapsed < 3:
        time.sleep(3 - elapsed)

    weather = get_uae_weather(city)
    weather_ctx = build_weather_context(weather)
    enriched = f"{weather_ctx}\n\nUser: {user_message}"

    # Image → Gemini Vision
    if image and gemini_model:
        try:
            prompt = f"{SYSTEM_PROMPT}\n\n{enriched}"
            resp   = gemini_model.generate_content([image, prompt])
            st.session_state.last_call_time = time.time()
            return resp.text, weather
        except Exception:
            pass  # fall through to Groq text

    # Text → Groq
    st.session_state.groq_history.append({
        "role": "user", "content": enriched
    })

    resp = safe_call(
        groq_client.chat.completions.create,
        model    = GROQ_MODEL,
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                   + st.session_state.groq_history,
        max_tokens  = 1024,
        temperature = 0.7,
    )

    if resp is None:
        st.session_state.groq_history.pop()
        return "⚠️ Could not get a response. Please try again in a moment.", weather

    reply = resp.choices[0].message.content
    st.session_state.groq_history.append({
        "role": "assistant", "content": reply
    })
    st.session_state.last_call_time = time.time()
    return reply, weather


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<div class='sidebar-logo'>🌿 GreenAdvisor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>UAE Plant Care Expert</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>📍 Location</div>", unsafe_allow_html=True)
    city = st.selectbox(
        "Select your city",
        options = list(UAE_CITIES.keys()),
        index   = 0,
        label_visibility = "collapsed",
    )
    st.session_state.selected_city = city

    st.markdown("<div class='sidebar-section'>📸 Plant Photo</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload plant image",
        type    = ["jpg", "jpeg", "png", "webp"],
        label_visibility = "collapsed",
    )
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded plant", use_container_width=True)

    st.markdown("<div class='sidebar-section'>🌡️ Live Weather</div>", unsafe_allow_html=True)
    if st.button("Refresh Weather 🌤", use_container_width=True):
        st.session_state.weather_cache[city] = get_uae_weather(city)

    st.markdown("<div class='sidebar-section'>💬 Session</div>", unsafe_allow_html=True)
    if st.button("New Conversation 🔄", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.groq_history  = []
        st.session_state.weather_cache = {}
        st.rerun()

    st.markdown("<div class='sidebar-section'>🌿 UAE Plant Guide</div>", unsafe_allow_html=True)
    for plant, info in UAE_PLANTS.items():
        st.markdown(f"<div class='plant-badge'><b>{plant}</b><br>{info}</div>",
                    unsafe_allow_html=True)

    # Gemini status indicator
    st.markdown("<div class='sidebar-section'>⚙️ Status</div>", unsafe_allow_html=True)
    st.markdown(
        f"**Diagnosis:** Groq ({GROQ_MODEL[:16]}...)\n\n"
        f"**Vision:** {'✅ Gemini active' if gemini_model else '⚠️ Text only'}",
    )


# ─────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class='ga-hero'>
    <h1>🌿 GreenAdvisor</h1>
    <p>Professional UAE Plant Care & Landscaping Advisor &nbsp;|&nbsp;
       خبير العناية بالنباتات في الإمارات</p>
</div>
""", unsafe_allow_html=True)


# ── Weather Card ─────────────────────────────────────────────
weather = st.session_state.weather_cache.get(city) or get_uae_weather(city)
st.session_state.weather_cache[city] = weather
interp  = interpret_weather(weather)

urgency_class = interp["urgency"] if interp["urgency"] in ["critical", "elevated"] else ""
mock_tag = " <span style='font-size:11px;opacity:0.6'>(estimated)</span>" if weather.get("mock") else ""

st.markdown(f"""
<div class='weather-card {urgency_class}'>
    <div class='weather-title'>
        📍 {weather['city']} — Current Conditions{mock_tag}
    </div>
    <div class='weather-grid'>
        <div class='weather-metric'>
            <span class='val'>{weather['temp']}°C</span>
            <span class='lbl'>Temperature</span>
        </div>
        <div class='weather-metric'>
            <span class='val'>{weather['humidity']}%</span>
            <span class='lbl'>Humidity</span>
        </div>
        <div class='weather-metric'>
            <span class='val'>{weather['wind_speed']}</span>
            <span class='lbl'>Wind km/h</span>
        </div>
    </div>
    <div style='margin-top:10px'>
        {"".join(f"<span class='alert-pill'>{a}</span>" for a in interp['alerts'])}
    </div>
    <div style='margin-top:8px; font-size:12px; color:#4a6358'>
        💧 {interp['watering']}
    </div>
</div>
""", unsafe_allow_html=True)


# ── Chat Window ──────────────────────────────────────────────
import html as html_lib

# Chat scroll container — open
st.markdown("<div class='chat-scroll'>", unsafe_allow_html=True)

if not st.session_state.messages:
    # Empty state
    st.markdown("""
    <div class='empty-chat'>
        <div class='icon'>🌱</div>
        <p>Welcome to GreenAdvisor!<br>
           Describe your plant problem below or upload a photo.<br><br>
           <span style='font-size:13px'>
           مرحباً! صف مشكلة نبتتك أدناه أو أرسل صورة
           </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # Escape user text to prevent HTML injection
            safe_text = html_lib.escape(msg["content"]).replace("\n", "<br>")
            st.markdown(f"""
            <div class='msg-wrap-user'>
                <div>
                    <div class='bubble-label' style='text-align:right'>You</div>
                    <div class='bubble-user'>{safe_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Agent reply — escape then convert newlines to <br>
            safe_reply = html_lib.escape(msg["content"]).replace("\n", "<br>")
            st.markdown(f"""
            <div class='msg-wrap-agent'>
                <div>
                    <div class='bubble-label'>🌿 GreenAdvisor</div>
                    <div class='bubble-agent'>{safe_reply}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Chat scroll container — close
st.markdown("</div>", unsafe_allow_html=True)


# ── Input Area ───────────────────────────────────────────────
st.markdown("<div class='input-container'>", unsafe_allow_html=True)

user_input = st.text_area(
    "Your message",
    placeholder = (
        "Describe your plant problem... e.g. 'My bougainvillea leaves are "
        "turning yellow in Dubai'\n\n"
        "أو اكتب بالعربية: نبتتي تذبل، ماذا أفعل؟"
    ),
    height      = 100,
    label_visibility = "collapsed",
)

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    image_note = f"📸 Image ready: {uploaded_file.name}" if uploaded_file else ""
    if image_note:
        st.caption(image_note)

with col2:
    send_clicked = st.button("Send 💬", use_container_width=True)

with col3:
    clear_clicked = st.button("Clear ✕", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ── Handle Send ──────────────────────────────────────────────
if send_clicked and (user_input.strip() or uploaded_file):
    msg_text = user_input.strip()

    if not msg_text and uploaded_file:
        msg_text = f"Please diagnose this plant. I am in {city}."

    display_text = msg_text
    if uploaded_file:
        display_text = f"📸 [{uploaded_file.name}]\n{msg_text}"

    st.session_state.messages.append({
        "role": "user", "content": display_text
    })

    with st.spinner("🌿 GreenAdvisor is analyzing your plant..."):
        plant_image = Image.open(uploaded_file) if uploaded_file else None
        reply, _    = diagnose(msg_text, image=plant_image, city=city)

    st.session_state.messages.append({
        "role": "assistant", "content": reply
    })
    st.rerun()


if clear_clicked:
    st.session_state.messages     = []
    st.session_state.groq_history = []
    st.rerun()


# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#aac4b2; font-size:12px; margin-top:32px; padding:16px;
            border-top: 1px solid #ddeee6'>
    🌿 GreenAdvisor — Powered by Groq &amp; Gemini &nbsp;|&nbsp;
    Built for UAE plant lovers &nbsp;|&nbsp;
    <a href='https://wa.me' style='color:#52b788'>
    Contact Toronto Agriculture for professional help
    </a>
</div>
""", unsafe_allow_html=True)
