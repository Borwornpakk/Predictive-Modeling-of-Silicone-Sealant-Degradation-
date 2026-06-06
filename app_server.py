"""
DOWSIL 795 — FastAPI Backend
รัน: python app_server.py
เปิด: index.html ในเบราว์เซอร์
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib, math, numpy as np
from pathlib import Path

app = FastAPI(title="DOWSIL 795 Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

BASE = Path(__file__).parent
clf = joblib.load(r"C:\Users\User\Videos\Project002\models\rf_classifier.pkl")
reg = joblib.load(r"C:\Users\User\Videos\Project002\models\rf_regressor.pkl")

R, Ea, T_REF = 8.314, 80_000, 298.15

def arrhenius(tc): return math.exp(-Ea/R*(1/(tc+273.15)-1/T_REF))
def dose(uv, tc, rh):
    af = arrhenius(tc); return uv + uv*(af-1)*.30 + uv*(rh/50-1)*.10

class Input(BaseModel):
    temp_celsius: float
    uv_hours: float
    relative_humidity: float

@app.post("/predict")
def predict(data: Input):
    af   = arrhenius(data.temp_celsius)
    d    = dose(data.uv_hours, data.temp_celsius, data.relative_humidity)
    X    = [[data.temp_celsius, data.uv_hours, data.relative_humidity, af, d]]
    drop = float(np.clip(reg.predict(X)[0], 0, 75))
    pf   = int(clf.predict(X)[0])
    prob = float(clf.predict_proba(X)[0][1])
    risk = ("ปกติ" if drop<10 else "ระวัง" if drop<30
            else "เสื่อมสภาพ" if drop<60 else "วิกฤต")
    rec  = {
        "ปกติ":      "ซีลแลนต์อยู่ในสภาพดี ตรวจสอบตามกำหนด",
        "ระวัง":     "วางแผนตรวจสอบเชิงรุกภายใน 12 เดือน",
        "เสื่อมสภาพ":"แนะนำให้ประเมินซ้ำและเตรียมแผนซ่อมแซม",
        "วิกฤต":    "⚠ ควรเปลี่ยนซีลแลนต์โดยเร็ว",
    }[risk]
    return {
        "elongation_drop":   round(drop, 1),
        "elongation_remain": round(400*(1-drop/100), 1),
        "tensile_drop":      round(drop/3.5, 1),
        "tensile_remain":    round(1.2*(1-drop/3.5/100), 3),
        "pass_fail":         pf,
        "pass_probability":  round(prob*100, 1),
        "arrhenius_factor":  round(af, 3),
        "effective_dose":    int(d),
        "risk_level":        risk,
        "recommendation":    rec,
    }

@app.get("/health")
def health(): return {"status": "ok", "model": "RF DOWSIL795"}

# Serve index.html at root
app.mount("/", StaticFiles(directory=str(BASE), html=True), name="static")

if __name__ == "__main__":
    import uvicorn, webbrowser, threading
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8000")).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
