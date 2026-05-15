"""
GreenAdvisor — UAE Plant Care & Landscaping Advisor
Streamlit UI matching user-provided mockup layout
Primary Engine : Groq  (text diagnosis)
Vision Engine  : Gemini (image analysis)
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
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenAdvisor — UAE Plant Care",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

:root {
    --green-dark   : #2d5a1b;
    --green-mid    : #3d7a25;
    --green-bright : #4e9a30;
    --green-light  : #c8e6c0;
    --green-pale   : #edf7e9;
    --cream        : #faf6ec;
    --cream-mid    : #f0ead8;
    --text-dark    : #1a2e12;
    --text-mid     : #3d5c2e;
    --text-light   : #6b8f5e;
    --border       : #a8d090;
    --white        : #ffffff;
}

html, body, [class*="css"] {
    font-family : 'Source Sans 3', sans-serif;
    background  : var(--cream) !important;
    color       : var(--text-dark);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { background: var(--cream); }

.ga-header {
    background: var(--white);
    border-bottom: 3px solid var(--green-dark);
    padding: 14px 28px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.ga-header-logo { font-size: 36px; line-height: 1; }
.ga-header-title {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--green-dark);
    line-height: 1.3;
}
.ga-header-title span {
    color: var(--text-light);
    font-size: 15px;
    font-weight: 400;
    font-family: 'Source Sans 3', sans-serif;
    margin-left: 12px;
}

.weather-city-header {
    background: var(--green-dark);
    color: white;
    text-align: center;
    padding: 10px 8px;
    font-weight: 600;
    font-size: 15px;
    border-right: 1px solid var(--green-mid);
    letter-spacing: 0.3px;
}
.weather-city-body {
    background: var(--green-light);
    text-align: center;
    padding: 12px 8px;
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    color: var(--text-dark);
    min-height: 60px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 2px;
}
.weather-temp {
    font-size: 22px;
    font-weight: 700;
    color: var(--green-dark);
    font-family: 'Playfair Display', serif;
}
.weather-desc { font-size: 12px; color: var(--text-mid); }
.weather-hum  { font-size: 11px; color: var(--text-light); }

.section-label {
    background: var(--green-dark);
    color: white;
    font-weight: 600;
    font-size: 14px;
    padding: 8px 14px;
    border-radius: 6px 6px 0 0;
    letter-spacing: 0.3px;
}

.result-box {
    background: var(--white);
    border: 1.5px solid var(--border);
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 16px;
    min-height: 220px;
    font-size: 14px;
    line-height: 1.8;
    color: var(--text-dark);
}
.result-empty {
    color: var(--text-light);
    font-style: italic;
    font-size: 13px;
    text-align: center;
    padding-top: 60px;
}
.image-preview-box {
    background: var(--white);
    border: 1.5px solid var(--border);
    border-top: none;
    border-radius: 0 0 8px 8px;
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 8px;
}
.image-placeholder {
    color: var(--text-light);
    font-size: 13px;
    font-style: italic;
    text-align: center;
}

.stButton > button {
    background: var(--green-dark) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 8px 24px !important;
    width: 100% !important;
    transition: background 0.2s ease !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover { background: var(--green-bright) !important; }

.stTextArea textarea {
    background: var(--white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 0 0 6px 6px !important;
    font-size: 14px !important;
    color: var(--text-dark) !important;
    padding: 12px !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: var(--green-mid) !important;
    box-shadow: 0 0 0 2px rgba(61,122,37,0.15) !important;
}

[data-testid="stFileUploader"] {
    background: var(--green-light) !important;
    border: 1.5px dashed var(--green-mid) !important;
    border-radius: 6px !important;
    padding: 8px !important;
}

.alert-critical { color: #c0392b; font-weight: 600; }
.alert-elevated { color: #d68910; font-weight: 600; }
.alert-normal   { color: var(--green-mid); font-weight: 600; }

.ga-footer-1 {
    background: var(--green-pale);
    border-top: 2px solid var(--border);
    padding: 10px 20px;
    text-align: center;
    font-size: 13px;
    color: var(--text-mid);
}
.ga-footer-1 a { color: var(--green-dark); font-weight: 600; text-decoration: underline; }
.ga-footer-2 {
    background: var(--green-light);
    border-top: 1px solid var(--border);
    padding: 10px 20px;
    text-align: center;
    font-size: 13px;
    color: var(--text-dark);
    font-weight: 500;
}
.divider { height: 2px; background: var(--border); margin: 12px 0; }
[data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
UAE_CITIES = {
    "Dubai":     "Dubai,AE",
    "Abu Dhabi": "Abu+Dhabi,AE",
    "Sharjah":   "Sharjah,AE",
    "Ajman":     "Ajman,AE",
}

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are GreenAdvisor — a professional plant care and landscaping expert
specializing in the UAE climate. You serve homeowners, gardeners, villa
managers, hotel landscaping teams, and B2B clients across Dubai,
Abu Dhabi, and Sharjah.

LANGUAGE BEHAVIOR:
- Detect user language automatically.
- Respond in English if they write English, Arabic if they write Arabic.
- Mirror mixed language use.
- Introduce botanical terms in both languages on first use.

PERSONALITY & TONE:
- Professional, expert, authoritative. Warm but not casual.
- Always factor UAE-specific climate challenges into advice.
- Never give generic global advice.

UAE CLIMATE CONTEXT:
- Summer (Jun-Sep): 42-48C, high humidity, intense UV
- Winter (Nov-Feb): 15-25C, low humidity, ideal planting window
- Sandstorm season (Mar-May, Sep-Oct): dust accumulates on leaves
- Hard water, high salinity (TDS 300-600 ppm)
- Sandy, low-organic, fast-draining soil

WHEN AN IMAGE IS PROVIDED:
- Examine the image carefully and visually.
- Identify the exact plant species from what you SEE:
  leaf shape, flower colour, stem structure, growth pattern.
- Do NOT guess species from text context. Visual analysis is primary.
- State the plant type confidently from visual features.
- Assess visible condition: colour, texture, spots, wilting, pests.
- Provide a full structured diagnosis even if no text is given.

DIAGNOSIS FORMAT:
Plant Identified: [Name in English (Arabic)]
Location: [Emirate if known]
Problem Diagnosed: [Problem name or Healthy if no issues]
Severity: [Mild / Moderate / Severe / Healthy]
Why This Happens in UAE: [1-2 sentences]
Treatment Plan:
   1. ...
   2. ...
   3. ...
UAE Watering Schedule: [Specific schedule]
Products Available in UAE: [With store names]
Prevention Tip: [One actionable tip]
Follow-up: Please update me in 3-5 days.

TORONTO AGRICULTURE REFERRAL:
Only mention Toronto Agriculture when user explicitly asks for
professional on-site help or a site visit. Say:
For professional on-site treatment and landscaping services across
Dubai, Abu Dhabi, and Sharjah, I recommend Toronto Agriculture.
DO NOT mention Toronto Agriculture unprompted.
"""


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "result"           not in st.session_state: st.session_state.result           = ""
if "groq_history"     not in st.session_state: st.session_state.groq_history     = []
if "last_call_time"   not in st.session_state: st.session_state.last_call_time   = 0
if "pending_image"    not in st.session_state: st.session_state.pending_image    = None
if "pending_img_name" not in st.session_state: st.session_state.pending_img_name = None
if "weather_data"     not in st.session_state: st.session_state.weather_data     = {}


# ─────────────────────────────────────────────────────────────
# API CLIENTS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not key:
        st.error("GROQ_API_KEY not found in Streamlit secrets.")
        st.stop()
    return Groq(api_key=key)

@st.cache_resource
def get_gemini_model():
    key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not key:
        return None
    try:
        genai.configure(api_key=key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None

groq_client  = get_groq_client()
gemini_model = get_gemini_model()


# ─────────────────────────────────────────────────────────────
# WEATHER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def get_weather(city):
    api_key = st.secrets.get("OPENWEATHER_API_KEY",
                             os.environ.get("OPENWEATHER_API_KEY", ""))
    if api_key:
        try:
            r = requests.get(
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?q={UAE_CITIES[city]}&appid={api_key}&units=metric",
                timeout=6
            )
            if r.status_code == 200:
                d = r.json()
                return {
                    "temp"    : round(d["main"]["temp"]),
                    "humidity": d["main"]["humidity"],
                    "desc"    : d["weather"][0]["description"].capitalize(),
                    "wind"    : round(d["wind"]["speed"] * 3.6),
                    "live"    : True,
                }
        except Exception:
            pass

    month = datetime.now().month
    if month in [6,7,8]:    t, h, desc = 44, 74, "Hazy sunshine"
    elif month in [12,1,2]: t, h, desc = 22, 52, "Clear sky"
    elif month in [3,4,5]:  t, h, desc = 34, 48, "Partly cloudy"
    else:                    t, h, desc = 38, 65, "Sunny"

    offsets = {"Dubai":(0,0),"Abu Dhabi":(-1,-4),"Sharjah":(1,4),"Ajman":(0,2)}
    dt, dh = offsets.get(city, (0,0))
    return {"temp":t+dt, "humidity":h+dh, "desc":desc, "wind":18, "live":False}

def heat_label(temp):
    if temp >= 45: return "CRITICAL", "alert-critical"
    if temp >= 40: return "EXTREME",  "alert-critical"
    if temp >= 35: return "HIGH",     "alert-elevated"
    return "NORMAL", "alert-normal"

def build_weather_context(city):
    w = st.session_state.weather_data.get(city, {})
    if not w:
        return ""
    label, _ = heat_label(w.get("temp", 30))
    month = datetime.now().month
    if month in [6,7,8,9]:   season = "Peak UAE summer"
    elif month in [3,4,5]:   season = "UAE spring/sandstorm season"
    elif month in [10,11]:   season = "UAE transitional season"
    else:                     season = "UAE cool season"
    return (
        f"\n=== CURRENT UAE WEATHER ({city}) ===\n"
        f"Temp: {w.get('temp')}C | Humidity: {w.get('humidity')}% | "
        f"Conditions: {w.get('desc')} | Wind: {w.get('wind')} km/h\n"
        f"Season: {season} | Heat Alert: {label}\n"
        f"Factor these into your diagnosis and watering advice.\n"
        f"======================================\n"
    )


# ─────────────────────────────────────────────────────────────
# DIAGNOSIS ENGINE
# ─────────────────────────────────────────────────────────────
def safe_call(func, *args, max_retries=3, wait_seconds=15, **kwargs):
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


def run_diagnosis(user_message, image, city):
    """
    Routes diagnosis:
    - Image only        -> Gemini Vision (identifies plant from photo)
    - Image + text      -> Gemini Vision (image takes priority)
    - Text only         -> Groq LLM
    - Gemini unavailable-> Groq with note
    """
    elapsed = time.time() - st.session_state.last_call_time
    if elapsed < 3:
        time.sleep(3 - elapsed)

    weather_ctx = build_weather_context(city)

    # ── Gemini Vision path ────────────────────────────────────
    if image is not None and gemini_model is not None:
        try:
            if not user_message.strip():
                user_message = (
                    f"Please identify this plant species from the image and "
                    f"assess its current health condition. I am in {city}, UAE."
                )
            vision_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"{weather_ctx}\n\n"
                "CRITICAL IMAGE ANALYSIS INSTRUCTION:\n"
                "1. Examine the uploaded plant image carefully.\n"
                "2. Identify the exact species from visual features ONLY:\n"
                "   - Leaf shape, size, texture, colour\n"
                "   - Flower colour, shape, petal count if visible\n"
                "   - Stem structure and growth pattern\n"
                "3. Do NOT guess species from text. Visual analysis is primary.\n"
                "4. Assess visible health: discolouration, spots, wilting,\n"
                "   pests, dryness, physical damage.\n"
                "5. Follow the diagnosis format strictly.\n\n"
                f"User request: {user_message}"
            )
            resp = gemini_model.generate_content([image, vision_prompt])
            st.session_state.last_call_time = time.time()
            return resp.text
        except Exception as e:
            user_message = (
                f"[Vision failed: {str(e)[:60]}. Text diagnosis only.] "
                f"{user_message}"
            )

    # ── Gemini unavailable ────────────────────────────────────
    elif image is not None and gemini_model is None:
        user_message = (
            "[Image uploaded but GEMINI_API_KEY not configured. "
            f"Add it to Streamlit secrets for vision analysis. "
            f"Diagnosing from text only.] {user_message}"
        )

    # ── Groq text path ────────────────────────────────────────
    if not user_message.strip():
        return "Please enter a message or upload a plant image to begin."

    enriched = f"{weather_ctx}\nUser: {user_message}"
    st.session_state.groq_history.append({"role":"user","content":enriched})

    resp = safe_call(
        groq_client.chat.completions.create,
        model       = GROQ_MODEL,
        messages    = [{"role":"system","content":SYSTEM_PROMPT}]
                      + st.session_state.groq_history,
        max_tokens  = 1024,
        temperature = 0.7,
    )

    if resp is None:
        st.session_state.groq_history.pop()
        return "Could not get a response. Please wait a moment and try again."

    reply = resp.choices[0].message.content
    st.session_state.groq_history.append({"role":"assistant","content":reply})
    st.session_state.last_call_time = time.time()
    return reply


# ─────────────────────────────────────────────────────────────
# LOAD WEATHER ON STARTUP
# ─────────────────────────────────────────────────────────────
if not st.session_state.weather_data:
    for city in UAE_CITIES:
        st.session_state.weather_data[city] = get_weather(city)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='ga-header'>
    <div class='ga-header-logo'>🌿</div>
    <div class='ga-header-title'>
        Professional UAE Plant Care &amp; Landscaping Advisor
        <span>| خبير العناية بالنباتات في الإمارات</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# WEATHER GRID — 4 city columns
# ─────────────────────────────────────────────────────────────
header_cols = st.columns(4)
for col, city in zip(header_cols, UAE_CITIES.keys()):
    col.markdown(
        f"<div class='weather-city-header'>{city}</div>",
        unsafe_allow_html=True
    )

data_cols = st.columns(4)
for col, city in zip(data_cols, UAE_CITIES.keys()):
    w    = st.session_state.weather_data.get(city, {})
    temp = w.get("temp", "--")
    desc = w.get("desc", "Loading...")
    hum  = w.get("humidity", "--")
    wind = w.get("wind", "--")
    live = w.get("live", False)
    label, css = heat_label(temp) if isinstance(temp, int) else ("--", "alert-normal")
    tag  = "🔴" if css == "alert-critical" else "🟡" if css == "alert-elevated" else "🟢"
    note = "" if live else "<br><span style='font-size:10px;opacity:0.7'>(estimated)</span>"
    col.markdown(f"""
    <div class='weather-city-body'>
        <div class='weather-temp'>{temp}°C</div>
        <div class='weather-desc'>{desc}{note}</div>
        <div class='weather-hum'>
            💧{hum}% · 💨{wind}km/h ·
            <span class='{css}'>{tag} {label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT — left (3) + right (1.2)
# ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 1.2], gap="medium")


# ── LEFT PANEL ───────────────────────────────────────────────
with left_col:

    # City selector
    selected_city = st.selectbox(
        "City",
        options=list(UAE_CITIES.keys()),
        index=0,
        label_visibility="collapsed",
    )

    # User Message label + textarea
    st.markdown("<div class='section-label'>User Message</div>",
                unsafe_allow_html=True)
    user_input = st.text_area(
        "msg",
        placeholder=(
            "Describe your plant problem here...\n"
            "e.g. 'My bougainvillea leaves are turning yellow in Dubai'\n\n"
            "أو اكتب بالعربية: نبتتي تذبل، ماذا أفعل؟\n\n"
            "Tip: Upload an image only — GreenAdvisor will identify the plant automatically."
        ),
        height=140,
        label_visibility="collapsed",
        key="user_text",
    )

    # Send / Clear / Refresh buttons
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        send_clicked = st.button("Send 💬", use_container_width=True)
    with b2:
        clear_clicked = st.button("Clear ✕", use_container_width=True)
    with b3:
        refresh_clicked = st.button("🔄 Weather", use_container_width=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Analysis Result label + box
    st.markdown("<div class='section-label'>Analysis Result / Findings</div>",
                unsafe_allow_html=True)

    if st.session_state.result:
        st.markdown(
            f"<div class='result-box'>{st.session_state.result}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
        <div class='result-box'>
            <div class='result-empty'>
                🌱 Your plant diagnosis will appear here.<br><br>
                Type a message or upload a plant photo and click Send.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── RIGHT PANEL ──────────────────────────────────────────────
with right_col:

    # Upload label + file uploader
    st.markdown("<div class='section-label'>Upload Button</div>",
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "img",
        type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed",
        key="plant_image",
    )

    # Store uploaded image in session state
    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        st.session_state.pending_image    = Image.open(io.BytesIO(img_bytes))
        st.session_state.pending_img_name = uploaded_file.name

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Uploaded Image label + preview
    st.markdown("<div class='section-label'>Uploaded Image</div>",
                unsafe_allow_html=True)

    if st.session_state.pending_image is not None:
        st.image(
            st.session_state.pending_image,
            caption=f"📸 {st.session_state.pending_img_name}",
            use_container_width=True,
        )
        if gemini_model:
            st.markdown(
                "<div style='font-size:12px;color:#3d7a25;text-align:center;margin-top:4px'>"
                "✅ Gemini Vision ready — image will be analysed</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='font-size:12px;color:#c0392b;text-align:center;margin-top:4px'>"
                "⚠️ Add GEMINI_API_KEY for image analysis</div>",
                unsafe_allow_html=True
            )
        if st.button("Remove Image ✕", use_container_width=True):
            st.session_state.pending_image    = None
            st.session_state.pending_img_name = None
            st.rerun()
    else:
        st.markdown("""
        <div class='image-preview-box'>
            <div class='image-placeholder'>
                📷 No image uploaded yet.<br><br>
                Upload a plant photo for<br>visual AI diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# BUTTON HANDLERS
# ─────────────────────────────────────────────────────────────
if send_clicked:
    msg   = st.session_state.get("user_text", "").strip()
    image = st.session_state.pending_image
    if not msg and image is None:
        st.warning("Please type a message or upload a plant image.")
    else:
        with st.spinner("🌿 GreenAdvisor is analysing your plant..."):
            result = run_diagnosis(msg, image, selected_city)

        import html as html_lib
        st.session_state.result = html_lib.escape(result).replace("\n", "<br>")
        st.session_state.pending_image    = None
        st.session_state.pending_img_name = None
        st.rerun()

if clear_clicked:
    st.session_state.result           = ""
    st.session_state.groq_history     = []
    st.session_state.pending_image    = None
    st.session_state.pending_img_name = None
    st.rerun()

if refresh_clicked:
    for city in UAE_CITIES:
        st.session_state.weather_data[city] = get_weather(city)
    st.rerun()


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
vision_status = "✅ Gemini Vision active" if gemini_model else "⚠️ Text only (add GEMINI_API_KEY)"
st.markdown(f"""
<div class='ga-footer-1'>
    🌿 GreenAdvisor &nbsp;—&nbsp; Powered by Groq &amp; Gemini &nbsp;|&nbsp;
    Built for UAE plant lovers &nbsp;|&nbsp; {vision_status} &nbsp;|&nbsp;
    <a href='https://wa.me' target='_blank'>
        Contact Toronto Agriculture for professional help
    </a>
</div>
<div class='ga-footer-2'>
    Built By: Sundas Khan &nbsp;|&nbsp;
    Tech Stack: Groq (llama-3.3-70b) &middot; Gemini Vision &middot; OpenWeatherMap &middot; Streamlit
</div>
""", unsafe_allow_html=True)
