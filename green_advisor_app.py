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


st.set_page_config(
    page_title="GreenAdvisor — UAE Plant Care",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family : 'DM Sans', sans-serif !important;
    background  : #f0f2ed !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }

.block-container {
    max-width    : 700px !important;
    padding      : 1.2rem 1rem 3rem !important;
    margin       : 0 auto !important;
}
[data-testid="stAppViewContainer"] { background: #f0f2ed; }

/* ── Shared card border ──────────────────────────────────── */
.ga-top {
    background   : #2d5a1b;
    border-radius: 14px 14px 0 0;
    padding      : 14px 20px;
    display      : flex;
    align-items  : center;
    gap          : 12px;
}
.ga-hdr-icon {
    width:38px;height:38px;border-radius:50%;
    background:rgba(255,255,255,0.13);
    display:flex;align-items:center;justify-content:center;
    font-size:19px;flex-shrink:0;
}
.ga-hdr-en { font-size:15px;font-weight:500;color:#edf7e9;line-height:1.3; }
.ga-hdr-ar { font-size:14px;color:#a8d5a0;margin-top:3px; }
.ga-loc-pill {
    background:rgba(255,255,255,0.12);
    border:0.5px solid rgba(255,255,255,0.22);
    border-radius:20px;padding:4px 13px;
    font-size:12px;color:#c8e6c0;
    display:flex;align-items:center;gap:5px;white-space:nowrap;
}

/* ── Weather grid ────────────────────────────────────────── */
.ga-wthr {
    display:grid;grid-template-columns:repeat(4,1fr);
    background:#edf7e9;
    border-left:0.5px solid #c8e6c0;border-right:0.5px solid #c8e6c0;
}
.ga-wc {
    padding:9px 6px;text-align:center;
    border-right:0.5px solid #c8e6c0;
}
.ga-wc:last-child{border-right:none}
.ga-wc.act{background:#fff;border-bottom:2px solid #2d5a1b}
.ga-wc-nm{font-size:10px;font-weight:500;color:#4e7a3a}
.ga-wc-t {font-size:18px;font-weight:500;color:#1a3d0f;line-height:1.2}
.ga-wc-d {font-size:10px;color:#7ab870}

/* ── City bar ────────────────────────────────────────────── */
.ga-city-bar {
    display:flex;align-items:center;gap:6px;flex-wrap:wrap;
    padding:7px 16px;background:#f7f9f5;
    border:0.5px solid #c8e6c0;border-top:none;
    font-size:12px;color:#7ab870;
}
.ga-city-bar strong{color:#2d5a1b}

/* ── Chat thread ─────────────────────────────────────────── */
.ga-thread {
    background:#f7f9f5;
    border:0.5px solid #c8e6c0;border-top:none;
    padding:16px;display:flex;flex-direction:column;gap:10px;
}
.ga-ts{text-align:center;font-size:11px;color:#bbb;margin:2px 0}

/* ── Bubbles ─────────────────────────────────────────────── */
.ga-row{display:flex;gap:8px;align-items:flex-end}
.ga-row.u{flex-direction:row-reverse}
.ga-av{
    width:30px;height:30px;border-radius:50%;flex-shrink:0;
    display:flex;align-items:center;justify-content:center;font-size:15px;
}
.ga-av.bot{background:#c8e6c0;color:#2d5a1b}
.ga-av.usr{background:#e2dfd8;font-size:11px;font-weight:500;color:#666}
.ga-bbl{
    max-width:75%;border-radius:16px;
    padding:12px 16px;font-size:14.5px;line-height:1.78;
    word-break:break-word;white-space:pre-wrap;
}
.ga-bbl.bot{
    background:#fff;border:0.5px solid #d4e8cc;
    color:#1a1a1a;border-bottom-left-radius:4px;
}
.ga-bbl.usr{
    background:#2d5a1b;color:#edf7e9;
    border-bottom-right-radius:4px;
}
.ga-bbl.img{
    padding:5px;background:#fff;border:0.5px solid #d4e8cc;
    border-bottom-right-radius:4px;
}
.ga-img-card{border-radius:10px;overflow:hidden;width:160px}
.ga-img-card img{width:100%;height:108px;object-fit:cover;display:block}
.ga-img-cap{font-size:11px;color:#999;padding:3px 8px 4px}

/* ── Input zone ──────────────────────────────────────────── */
.ga-input-zone {
    background:#fff;
    border:0.5px solid #c8e6c0;border-top:0.5px solid #e0ead8;
    border-radius:0 0 14px 14px;
    padding:12px 16px 14px;
}
.ga-thumb-row {
    display:flex;align-items:center;gap:10px;
    background:#edf7e9;border:0.5px solid #a8c8a0;
    border-radius:10px;padding:7px 11px;margin-bottom:8px;
}
.ga-thumb-img{
    width:44px;height:44px;border-radius:7px;
    object-fit:cover;border:0.5px solid #a8c8a0;flex-shrink:0;
}
.ga-thumb-info{flex:1}
.ga-thumb-name{font-size:13px;font-weight:500;color:#2d5a1b}
.ga-thumb-ok{font-size:11px;color:#4e9a30;margin-top:1px}

/* ── Footer ──────────────────────────────────────────────── */
.ga-footer{
    background:#edf7e9;border:0.5px solid #c8e6c0;border-top:none;
    border-radius:0 0 14px 14px;
    padding:8px 18px;text-align:center;font-size:11.5px;color:#4e7a3a;
}
.ga-footer a{color:#2d5a1b;font-weight:500;text-decoration:none}
.ga-footer-sub{font-size:10.5px;color:#7ab870;margin-top:2px}

/* ── Streamlit widget overrides ──────────────────────────── */
.stButton > button {
    background:#2d5a1b !important;color:#edf7e9 !important;
    border:none !important;border-radius:22px !important;
    font-size:13.5px !important;font-family:'DM Sans',sans-serif !important;
    padding:8px 22px !important;width:100% !important;
    font-weight:500 !important;transition:background .15s !important;
}
.stButton > button:hover{background:#3d7a25 !important}

.stTextArea textarea {
    background:#f7f9f5 !important;border:0.5px solid #c8e6c0 !important;
    border-radius:12px !important;font-size:14px !important;
    font-family:'DM Sans',sans-serif !important;
    padding:10px 14px !important;resize:none !important;
    color:#1a1a1a !important;line-height:1.6 !important;
}
.stTextArea textarea:focus{
    border-color:#3d7a25 !important;background:#fff !important;
    box-shadow:0 0 0 2px rgba(61,122,37,0.09) !important;
}
.stTextArea textarea::placeholder{color:#bbb !important}
.stTextArea label{display:none !important}

[data-testid="stFileUploader"]{
    background:#f7f9f5 !important;
    border:0.5px dashed #a8c8a0 !important;
    border-radius:10px !important;
}
[data-testid="stFileUploader"] label p{
    font-size:13px !important;color:#4e7a3a !important;
}
</style>
""", unsafe_allow_html=True)


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
You are GreenAdvisor — a master gardener and horticulturist with over 20 years
of hands-on experience across the UAE. You have grown, diagnosed, and rescued
thousands of plants in Dubai, Abu Dhabi, Sharjah, Ajman, RAK, and Fujairah.
You know every common and rare species that thrives — or struggles — in this
climate. You speak with quiet authority, like a trusted expert who has seen it
all before.

━━━ WHO YOU ARE ━━━
You are NOT an AI describing what it sees. You are a seasoned professional
giving a direct, confident verdict. You:
- Name plants immediately and with authority. No hedging. No percentages.
- Give your professional opinion, not a visual narration.
- Never say "I can see...", "I notice...", "The image shows...", or
  "I am X% confident". A real expert never talks this way.
- Speak warmly but directly — like a friend who happens to be the best
  plant person in the UAE.
- Always give UAE-specific advice. Never generic global advice.

━━━ LANGUAGE ━━━
- Auto-detect: English → English, Arabic → Arabic, mixed → mirror.
- Give the botanical name and Arabic name on first mention, then use the
  common name naturally in conversation.

━━━ UAE EXPERTISE ━━━
You carry this knowledge in every answer:
- Summer (Jun–Sep): 42–48°C, high humidity, intense UV — heat stress, fungal
  issues, and overwatering are the biggest killers.
- Winter (Nov–Feb): 15–25°C, low humidity — the ideal planting and recovery
  window. Push soil improvement and fertilising now.
- Sandstorm season (Mar–May, Sep–Oct): dust clogs stomata, stress spikes.
  Wipe leaves, boost watering slightly.
- Hard water (TDS 300–600 ppm): salt accumulates in soil over time. Flush
  monthly. Use filtered water for sensitive species.
- Soil: sandy, fast-draining, low organic matter. Always amend with compost.
  Most UAE soil needs feeding every 3–4 weeks in growing season.

━━━ CONVERSATION — TWO STAGE ━━━

STAGE 1 — IMMEDIATE VERDICT (your first response, always):
Short, direct, professional. No narration. No observations. Just your verdict:

  🌿 [Plant common name] ([Botanical name] / [Arabic name])
  ❤️ [One sentence on health status — your professional assessment]
  🎯 [✅ Healthy / 🟢 Mild concern / 🟡 Needs attention / 🔴 Urgent care needed]
  💬 [One specific, helpful follow-up offer]

  Good examples of the health line:
  ✅ "This Snake Plant is thriving — perfect condition for the UAE."
  🟡 "Your mango has early fungal spotting — common in Sharjah's humidity."
  🔴 "This bougainvillea is severely root-bound and heat-stressed — act fast."

  Good follow-up offers:
  "Want my recommended care routine for this plant in your climate?"
  "Shall I give you a full recovery plan?"
  "Would you like to know the best spot in your home for this plant?"

STAGE 2 — FULL EXPERT PLAN (only when user asks for more):

  🌿 [Plant name] ([Botanical] / [Arabic])
  ⚠️ Diagnosis: [Problem, or "Healthy — routine care only"]
  🎯 Severity: [✅ / 🟢 / 🟡 / 🔴]
  🔍 In the UAE this happens because: [1–2 sentences, specific to UAE conditions]
  💊 What to do:
     1. [Actionable step]
     2. [Actionable step]
     3. [Actionable step]
  💧 Watering in UAE: [Specific schedule — times of day, frequency, amount]
  🛒 Get it in UAE: [Products + where: ACE Hardware, Lulu, Garden Centre, Plantshop.ae]
  🛡️ Prevent this: [One practical tip]
  📅 Check back: [When and what to look for]

━━━ SEVERITY — NON-NEGOTIABLE RULES ━━━
✅ Healthy: vigorous, green, firm, no damage. Say it's healthy. Full stop.
   DO NOT invent mild concerns for a healthy plant. That is unprofessional.
🟢 Mild: one or two minor issues — a yellowing leaf, slight dust, minor tip burn.
🟡 Moderate: clear, multiple symptoms — several yellow leaves, visible pests,
   drooping that does not recover overnight.
🔴 Severe: widespread collapse, heavy infestation, root rot, near-death state.
If you are unsure, always go one level lower — err on the side of reassurance,
not alarm. A stressed owner does not need unnecessary panic.

━━━ WHEN AN IMAGE IS PROVIDED ━━━
Identify the plant like an expert who has grown it for decades:
- Give the name directly. Do not narrate your reasoning process.
- If genuinely ambiguous between two species, say: "This could be an X or a Y —
  the care is similar either way. Here is what I recommend."
- If the image is too blurry to identify, say so simply and ask for a clearer photo.
- Assess health based on what is actually wrong, not what might be wrong.
- If the plant is healthy, say so warmly and confidently.

━━━ TORONTO AGRICULTURE ━━━
Only mention when the user explicitly asks for professional on-site help or a
site visit. Then say: "For professional landscaping, irrigation, pools, and
plant installation across Dubai, Abu Dhabi, and Sharjah, I recommend Toronto
Agriculture — UAE specialists with over a decade of on-the-ground experience."
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
    "user_city_override": None,  # set when user manually taps a city
    "_pending_chips"  : [],   # quick reply chips (unused, kept for compat)
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
            "You are looking at a plant photo sent by a UAE gardener who needs your expert opinion.\n"
            "Respond as a 20-year master gardener would — give your immediate professional verdict.\n\n"
            "CRITICAL — what NOT to do:\n"
            "• Do NOT narrate what you see ('I can see...', 'The image shows...', 'Here is what I see')\n"
            "• Do NOT list leaf colours, stem structures, or growth habits as observations\n"
            "• Do NOT state a confidence percentage\n"
            "• Do NOT number your reasoning steps\n\n"
            "WHAT TO DO:\n"
            "• Name the plant directly and with authority\n"
            "• Give your professional verdict on its health in one sentence\n"
            "• Use the Stage 1 format from your instructions\n"
            "• If the plant is healthy, say so warmly — do not manufacture concerns\n"
            "• If genuinely unsure between two species, say so in one natural sentence\n"
            "• If the photo is too blurry, ask simply for a clearer photo\n\n"
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


import html as _html
import io as _io
import base64 as _b64


# ─────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────
def _img_bubble(name: str, pil_img) -> str:
    buf = _io.BytesIO()
    img = pil_img.copy()
    img.thumbnail((320, 216))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=80)
    b64 = _b64.b64encode(buf.getvalue()).decode()
    return (
        "<div class='ga-row u'>"
        "<div class='ga-bbl img'>"
        "<div class='ga-img-card'>"
        f"<img src='data:image/jpeg;base64,{b64}' alt='Plant photo'>"
        "</div>"
        f"<div class='ga-img-cap'>&#128247; {_html.escape(name)}</div>"
        "</div>"
        "<div class='ga-av usr'>You</div>"
        "</div>"
    )

def _usr_bubble(text: str) -> str:
    safe = _html.escape(text).replace("\n", "<br>")
    return (
        "<div class='ga-row u'>"
        f"<div class='ga-bbl usr'>{safe}</div>"
        "<div class='ga-av usr'>You</div>"
        "</div>"
    )

def _bot_bubble(text: str) -> str:
    safe = _html.escape(text).replace("\n", "<br>")
    return (
        "<div class='ga-row'>"
        "<div class='ga-av bot'>&#127807;</div>"
        f"<div class='ga-bbl bot'>{safe}</div>"
        "</div>"
    )

def render_thread() -> str:
    out = "<div class='ga-thread'><div class='ga-ts'>Today</div>"
    if not st.session_state.chat_history:
        out += _bot_bubble(
            "Ahlan! I am GreenAdvisor, your UAE plant care expert.\n\n"
            "Attach a photo of your plant using the upload area below, "
            "then type your question or simply press Send."
        )
    else:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            if role == "image" and msg.get("pil"):
                out += _img_bubble(msg["name"], msg["pil"])
            elif role == "user":
                out += _usr_bubble(msg["text"])
            elif role == "advisor":
                out += _bot_bubble(msg["text"])
    out += "</div>"
    return out


# ─────────────────────────────────────────────────────────────
# ██  HEADER
# ─────────────────────────────────────────────────────────────
detected_city = detect_user_city()
city_names    = list(UAE_CITIES.keys())
_wc           = st.session_state.widget_counter

st.markdown(
    f"<div class='ga-top'>"
    f"<div class='ga-hdr-icon'>&#127807;</div>"
    f"<div style='flex:1'>"
    f"<div class='ga-hdr-en'>GreenAdvisor &#8212; UAE Plant Care</div>"
    f"<div class='ga-hdr-ar'>&#1582;&#1576;&#1610;&#1585; &#1575;&#1604;&#1593;&#1606;&#1575;&#1610;&#1577; &#1576;&#1575;&#1604;&#1606;&#1576;&#1575;&#1578;&#1575;&#1578; &#1601;&#1610; &#1575;&#1604;&#1573;&#1605;&#1575;&#1585;&#1575;&#1578;</div>"
    f"</div>"
    f"<div class='ga-loc-pill'>&#128205; {detected_city}</div>"
    f"</div>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# ██  WEATHER BAR
# ─────────────────────────────────────────────────────────────
_whtml = "<div class='ga-wthr'>"
for _cn in city_names:
    _w  = st.session_state.weather_data.get(_cn, {})
    _t  = _w.get("temp", "--")
    _d  = _w.get("desc", "...")
    _ac = " act" if _cn == detected_city else ""
    _whtml += (
        f"<div class='ga-wc{_ac}'>"
        f"<div class='ga-wc-nm'>{_cn}</div>"
        f"<div class='ga-wc-t'>{_t}&#176;C</div>"
        f"<div class='ga-wc-d'>{_d}</div>"
        f"</div>"
    )
_whtml += "</div>"
st.markdown(_whtml, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  CITY SELECTOR
# ─────────────────────────────────────────────────────────────
st.markdown(
    f"<div class='ga-city-bar'>&#128205; Your city: "
    f"<strong>{detected_city}</strong>&nbsp;&nbsp;Not right? Change:</div>",
    unsafe_allow_html=True,
)

_ccols = st.columns(len(city_names))
for _ci, (_col, _cn) in enumerate(zip(_ccols, city_names)):
    with _col:
        if st.button(_cn, key=f"c_{_ci}_{_wc}"):
            st.session_state.user_city_override = _cn
            st.session_state.detected_city      = _cn
            st.rerun()

# Highlight the active city button with CSS
_aidx = city_names.index(detected_city) + 1
st.markdown(
    f"<style>div[data-testid='stHorizontalBlock']>"
    f"div:nth-child({_aidx}) .stButton>button{{"
    f"background:#2d5a1b !important;color:#edf7e9 !important;"
    f"border:0.5px solid #2d5a1b !important;font-weight:600 !important}}"
    f"</style>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# ██  CHAT THREAD
# ─────────────────────────────────────────────────────────────
st.markdown(render_thread(), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  INPUT AREA  (open div — closed in footer)
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='ga-input-zone'>", unsafe_allow_html=True)

# 1 ── File uploader
uploaded_file = st.file_uploader(
    "&#128247; Attach plant photo",
    type=["jpg", "jpeg", "png", "webp"],
    key=f"uploader_{_wc}",
    label_visibility="visible",
)
if uploaded_file is not None:
    _nm = uploaded_file.name
    if _nm != st.session_state.get("pending_img_name"):
        _raw  = uploaded_file.read()
        _pil  = Image.open(_io.BytesIO(_raw))
        st.session_state.pending_image    = _pil
        st.session_state.pending_img_name = _nm
        st.session_state.pending_img_b64  = pil_to_base64_jpeg(_pil)

# 2 ── Thumbnail + remove button
if st.session_state.pending_image is not None:
    _tb = _io.BytesIO()
    _tp = st.session_state.pending_image.copy()
    _tp.thumbnail((88, 88))
    if _tp.mode != "RGB":
        _tp = _tp.convert("RGB")
    _tp.save(_tb, format="JPEG", quality=80)
    _t64 = _b64.b64encode(_tb.getvalue()).decode()
    _tnm = _html.escape(st.session_state.pending_img_name or "photo.jpg")

    _pc, _rc = st.columns([9, 1])
    with _pc:
        st.markdown(
            f"<div class='ga-thumb-row'>"
            f"<img class='ga-thumb-img' "
            f"src='data:image/jpeg;base64,{_t64}' alt='thumbnail'>"
            f"<div class='ga-thumb-info'>"
            f"<div class='ga-thumb-name'>&#128247; {_tnm}</div>"
            f"<div class='ga-thumb-ok'>&#10003; Ready for analysis</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with _rc:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("&#10005;", key=f"rm_{_wc}", help="Remove image"):
            st.session_state.pending_image    = None
            st.session_state.pending_img_name = None
            st.session_state.pending_img_b64  = None
            st.session_state.widget_counter  += 1
            st.rerun()

# 3 ── Text area
user_input = st.text_area(
    "msg",
    placeholder="Ask about your plant, or just press Send with a photo...",
    height=80,
    label_visibility="collapsed",
    key=f"msg_{_wc}",
)

# 4 ── Action buttons
_b1, _b2, _b3 = st.columns([3, 2, 2])
with _b1:
    send_clicked  = st.button("&#128172; Send", key=f"send_{_wc}")
with _b2:
    clear_clicked = st.button("&#10005; Clear", key=f"clr_{_wc}")
with _b3:
    if st.button("&#8635; Weather", key=f"wx_{_wc}"):
        for _c in UAE_CITIES:
            st.session_state.weather_data[_c] = get_weather(_c)
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(
    "<div class='ga-footer'>"
    "&#127807; GreenAdvisor &nbsp;&#183;&nbsp; Groq Llama-4 Vision "
    "&nbsp;&#183;&nbsp; "
    "<a href='https://wa.me/971000000000' target='_blank'>"
    "Contact Toronto Agriculture</a>"
    "<div class='ga-footer-sub'>"
    "Built by Sundas Khan &nbsp;&#183;&nbsp; "
    "Groq llama-3.3-70b &amp; llama-4-maverick &nbsp;&#183;&nbsp; "
    "Open-Meteo &nbsp;&#183;&nbsp; Streamlit"
    "</div></div>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# ██  BUTTON HANDLERS
# ─────────────────────────────────────────────────────────────
_snap_b64 = st.session_state.pending_img_b64

if send_clicked:
    _msg = (user_input or "").strip()
    _b64 = _snap_b64
    if not _msg and _b64 is None:
        st.warning("Please type a message or attach a plant photo.")
    else:
        if _b64 is not None:
            st.session_state.chat_history.append({
                "role": "image",
                "name": st.session_state.pending_img_name or "photo.jpg",
                "pil" : st.session_state.pending_image,
            })
        if _msg:
            st.session_state.chat_history.append({"role": "user", "text": _msg})

        with st.spinner("&#127807; Analysing your plant..."):
            _diag, _eng = run_diagnosis(_msg, _b64, detected_city)

        _tag = " ⚠️" if _eng == "error" else ""
        st.session_state.chat_history.append({
            "role": "advisor",
            "text": _diag + _tag,
        })
        st.session_state.pending_image    = None
        st.session_state.pending_img_name = None
        st.session_state.pending_img_b64  = None
        st.session_state.widget_counter  += 1
        st.rerun()

if clear_clicked:
    st.session_state.chat_history     = []
    st.session_state.groq_history     = []
    st.session_state.pending_image    = None
    st.session_state.pending_img_name = None
    st.session_state.pending_img_b64  = None
    st.session_state.widget_counter  += 1
    st.rerun()
