import os, re, json, time, logging
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from google import genai
from google.genai import types

LANGS = {"uz": "Uzbek (Latin)", "en": "English", "ru": "Russian", "de": "German"}
MODELS = [x for x in [os.getenv("GEMINI_MODEL"), "gemini-2.5-flash", "gemini-flash-latest",
                      "gemini-2.0-flash", "gemini-2.5-flash-lite"] if x]
LIMIT = int(os.getenv("DAILY_LIMIT", "10"))  # har bir IP uchun kunlik so'rovlar
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
app = FastAPI()
hits = defaultdict(list)

FOOD_PROMPT = (
    "Analyze this food photo. Identify each dish/food item, estimate the portion in grams, "
    "and estimate calories and macros for that portion. Use LANGNAME names. "
    'Return ONLY JSON: {"items":[{"name":"","grams":0,"kcal":0,"protein":0,"fat":0,"carbs":0}],'
    '"note":"one short sentence about uncertainty in LANGNAME"}'
)


def check_limit(req: Request):
    ip = (req.headers.get("x-forwarded-for") or req.client.host).split(",")[0].strip()
    now = time.time()
    hits[ip] = [t for t in hits[ip] if now - t < 86400]
    if len(hits[ip]) >= LIMIT:
        raise HTTPException(429, "Bugungi limit tugadi. Ertaga qayta urinib ko'ring.")
    hits[ip].append(now)


def generate(contents, config=None):
    last = None
    for name in MODELS:  # bittasi ishlamasa, keyingisini sinaydi
        try:
            return client.models.generate_content(model=name, contents=contents, config=config)
        except Exception as e:
            last = e
            logging.warning("model %s xato: %s", name, e)
    raise last


def ask(contents):
    try:
        r = generate(contents, types.GenerateContentConfig(response_mime_type="application/json"))
        m = re.search(r"\{.*\}", r.text, re.S)
        return json.loads(m.group(0))
    except Exception as e:
        logging.exception("AI xatosi")
        raise HTTPException(502, "AI xatosi: " + str(e)[:200])


@app.post("/api/analyze")
def analyze(request: Request, image: UploadFile = File(...), lang: str = Form("uz")):
    check_limit(request)
    data = image.file.read()
    if len(data) > 8_000_000:
        raise HTTPException(413, "Rasm juda katta")
    media = image.content_type if image.content_type in ("image/jpeg", "image/png", "image/webp") else "image/jpeg"
    return ask([types.Part.from_bytes(data=data, mime_type=media), FOOD_PROMPT.replace("LANGNAME", LANGS.get(lang, "Uzbek (Latin)"))])


@app.post("/api/plan")
def plan(request: Request, kcal: int = Form(...), protein: int = Form(...),
         fat: int = Form(...), carbs: int = Form(...), goal: str = Form(...), lang: str = Form("uz")):
    check_limit(request)
    q = (f"Create a one-day meal plan (breakfast, lunch, snack, dinner) totaling about {kcal} kcal, "
         f"{protein} g protein, {fat} g fat, {carbs} g carbs. Goal: {goal}. Prefer affordable foods "
         "common in Uzbekistan, including some Uzbek dishes with portions in grams. Write in LANGNAME. "
         'Return ONLY JSON: {"meals":[{"name":"","foods":"","kcal":0,"protein":0}]}')
    return ask(q.replace("LANGNAME", LANGS.get(lang, "Uzbek (Latin)")))


@app.post("/api/calc")
def calc(request: Request, text: str = Form(...), lang: str = Form("uz")):
    check_limit(request)
    text = text.strip()[:500]
    q = ('The user describes what they ate (Uzbek, Russian or English). Include ONLY the foods explicitly mentioned in the text at the end, never add any other food. '
         'Split it into separate food items. For each item convert the quantity (pieces, spoons, cups, grams) '
         'into grams and estimate calories and macros for the WHOLE quantity eaten. Use LANGNAME names '
         'that include the quantity as written by the user. Return ONLY JSON: {"items":[{"name":"","grams":0,'
         '"kcal":0,"protein":0,"fat":0,"carbs":0}],"note":"one short LANGNAME sentence about uncertainty"}. '
         f'Text: "{text}"')
    return ask(q.replace("LANGNAME", LANGS.get(lang, "Uzbek (Latin)")))


@app.post("/api/chat")
def chat(request: Request, message: str = Form(...), history: str = Form(""), lang: str = Form("uz")):
    check_limit(request)
    sysmsg = (
        "You are NutryAi's sports and fitness coach. Answer ONLY questions about sports, exercise, training, "
        "sports nutrition and healthy habits connected to training. If the question is outside this, politely "
        "say you can only help with sports topics. Be concise (under 150 words) and practical. You are not a "
        "doctor: give no diagnoses and suggest seeing a doctor for pain, injuries or medical conditions. "
        "Never recommend extreme diets or very low calories. Reply in " + LANGS.get(lang, "Uzbek (Latin)") + "."
    )
    try:
        r = generate(f"Conversation so far:\n{history[-3000:]}\nUser question: {message[:1000]}", types.GenerateContentConfig(system_instruction=sysmsg))
        return {"answer": r.text}
    except Exception as e:
        logging.exception("AI xatosi")
        raise HTTPException(502, "AI xatosi: " + str(e)[:200])


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))


