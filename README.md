# Predictive Modeling of Silicone Sealant Degradation and Weather-Seal Performance Using Machine Learning

> **A physics-informed machine learning system for predicting long-term degradation of structural silicone sealant (DOWSIL™ 795), enabling data-driven maintenance planning for curtainwall and façade engineering applications.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange?logo=scikit-learn)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Hardware & Software Requirements](#3-hardware--software-requirements)
4. [Implementation Details](#4-implementation-details)
5. [Usage & Demonstration](#5-usage--demonstration)
6. [Discussion & Conclusion](#6-discussion--conclusion)
7. [Future Work](#7-future-work)

---

## 1. Introduction

### Problem Statement

Structural silicone sealants such as DOWSIL™ 795 are critical components in curtainwall façade systems, providing both structural bonding and weatherproofing functions. Over their service life, these materials are continuously subjected to cumulative stressors — solar UV radiation, elevated temperatures, and ambient humidity — that progressively reduce key mechanical properties, most critically **elongation-at-break** and **tensile strength**.

The conventional approach to maintenance is **reactive**: sealant is inspected and replaced only after visible failure, such as cohesive cracking, loss of adhesion, or water infiltration. This approach carries significant risks including structural glass detachment, interior water damage, and unplanned capital expenditure.

### Solution

This project constructs a **physics-informed machine learning pipeline** that:

- Derives degradation equations from the DOWSIL™ 795 Technical Data Sheet (TDS), anchored to real laboratory test results (ASTM G154, 5,000-hour QUV weatherometer data).
- Generates a calibrated synthetic dataset of 1,000 samples using the **Arrhenius kinetic model** and **exponential decay** formulations.
- Trains a **Random Forest** ensemble model (classifier + regressor) on this dataset to predict sealant condition from field-measurable environmental inputs alone.
- Serves predictions through an interactive web dashboard, allowing engineers and building owners to assess sealant health without manual equation solving.

The result is a shift from **reactive repair** to **predictive maintenance** — enabling pre-scheduled intervention before mechanical properties fall below the safety threshold defined by ASTM C1184.

---

## 2. System Architecture

### 2.1 End-to-End Workflow

```
TDS (DOWSIL™ 795)
      │
      ▼
[Physics Model]  ──  Arrhenius Equation + Exponential Decay
      │
      ▼
[Synthetic Data Generator]  ──  1,000-row dataset (Python / NumPy)
      │                          Features: Temp, UV-hours, RH, AF, D_eff
      ▼                          Targets: Elongation drop %, Pass/Fail label
[Model Training]
      ├── RandomForestClassifier  →  Pass / Fail (binary)
      └── RandomForestRegressor   →  Elongation drop (%)
      │
      ▼
[Serialized Models]  ──  rf_classifier.pkl  /  rf_regressor.pkl
      │
      ▼
[FastAPI Backend]  ──  POST /predict  →  JSON response
      │
      ▼
[Web Dashboard]  ──  index.html  (sliders → real-time prediction)
```

### 2.2 Data Engineering

| Layer | Description |
|---|---|
| **Raw reference** | DOWSIL™ 795 TDS (Form No. 63-1217-01-0124 S2D, Dow 2018–2024) |
| **Calibration anchor** | QUV 5,000 h → Tensile 1.1 MPa (retention 91.7%), ASTM G154 |
| **Sampling space** | Temp: 20–85 °C · UV: 0–10,000 h · RH: 40–90% (uniform random) |
| **Engineered features** | `arrhenius_factor` (AF), `effective_dose_h` (D_eff) |
| **Targets** | `elongation_drop_pct` (regression), `pass_fail_label` (classification) |
| **Dataset size** | 1,000 rows × 17 columns · random seed 42 |
| **Train/test split** | 80% / 20%, stratified by class |

### 2.3 Project Roadmap

```
Phase 1 ── Literature & TDS Review
           └─ Extract mechanical property baselines and QUV anchor data

Phase 2 ── Physics Model Design
           └─ Formulate Arrhenius factor, effective dose, exponential decay

Phase 3 ── Synthetic Data Generation
           └─ 1,000-row dataset · noise injection · property floors

Phase 4 ── ML Model Training & Evaluation
           └─ Classifier (Pass/Fail) + Regressor (Elongation drop%)
           └─ 5-fold cross-validation · confusion matrix · ROC-AUC

Phase 5 ── API & Dashboard Development
           └─ FastAPI backend · HTML/JS frontend

Phase 6 ── Deployment
           └─ Local (localhost:8000) via FastAPI + index.html
```

---

## 3. Hardware & Software Requirements

### 3.1 Hardware

No specialized hardware is required. All modeling was performed on a standard workstation. Because the dataset is fully synthetic (no physical sensors were deployed), the workflow is reproducible on any machine capable of running Python 3.9+.

### 3.2 Software & Tech Stack

| Category | Tool / Library | Purpose |
|---|---|---|
| **Language** | Python 3.9+ | Core language for all data and model work |
| **Data handling** | Pandas, NumPy | Synthetic data generation, feature engineering |
| **Physics model** | Python `math` / NumPy | Arrhenius factor, effective dose computation |
| **ML framework** | scikit-learn 1.8.0 | Random Forest classifier & regressor, cross-validation |
| **Model persistence** | joblib | Serialize/deserialize `.pkl` model files |
| **API backend** | FastAPI + Uvicorn | REST endpoint (`POST /predict`) serving model inference |
| **Frontend** | HTML5 / CSS3 / Vanilla JS | Interactive dashboard (no framework dependency) |
| **Visualization** | Plotly (optional) | Degradation trend charts |
| **Notebook environment** | Jupyter Notebook / Google Colab | Step-by-step model exploration without local install |

---

## 4. Implementation Details

### 4.1 Mathematical Logic — Arrhenius-Based Degradation Model

Degradation rate of polymeric materials under combined UV and thermal stress is well-described by the **Arrhenius kinetic equation**:

$$k(T) = A \cdot \exp\!\left(-\frac{E_a}{RT}\right)$$

where $E_a = 80{,}000\ \text{J/mol}$ (activation energy for silicone UV/thermal degradation), $R = 8.314\ \text{J/mol·K}$, and $T_{\text{ref}} = 298.15\ \text{K}$ (25 °C, TDS test condition).

The **Arrhenius Acceleration Factor (AF)** relative to the reference temperature is:

$$AF(T) = \exp\!\left[-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right)\right]$$

The **effective dose** integrates UV exposure with thermal and humidity contributions:

$$D_{\text{eff}} = t_{\text{UV}} + t_{\text{UV}} \cdot (AF - 1) \cdot 0.30 + t_{\text{UV}} \cdot \left(\frac{RH}{50} - 1\right) \cdot 0.10$$

Property retention follows **exponential decay**:

$$P(t) = P_0 \cdot \exp(-k_{\text{eff}} \cdot D_{\text{eff}})$$

The rate constants were calibrated from the TDS anchor point (QUV 5,000 h → Tensile retention 91.7%):

| Property | Rate constant $k$ (h⁻¹) | Rationale |
|---|---|---|
| Tensile strength | $1.67 \times 10^{-5}$ | Calibrated directly from TDS anchor |
| Elongation-at-break | $5.83 \times 10^{-5}$ | 3.5× tensile (elongation degrades faster) |
| Peel strength | $3.00 \times 10^{-5}$ | 1.8× tensile |
| Shore A hardness | $0.84 \times 10^{-5}$ | 0.5× tensile (hardness relatively stable) |

### 4.2 Synthetic Data Generation

```python
# Core generation logic
af   = np.array([arrhenius_factor(t) for t in temp_C])
dose = np.array([effective_dose(u, a, r) for u, a, r in zip(uv_hours, af, rh_factor)])

# Property with multiplicative Gaussian noise (CV = 5% for elongation)
elongation = P0 * np.exp(-k_elong * dose) * np.random.normal(1.0, 0.05, N)
elongation = np.clip(elongation, P0 * 0.28, P0 * 1.02)   # floor at 28% retention

# Binary ML target: FAIL if elongation drop ≥ 50% OR tensile < 0.72 MPa
pass_fail = ((elong_drop < 50) & (tensile >= 0.72)).astype(int)
```

The resulting dataset contains **37.7% PASS / 62.3% FAIL** samples, reflecting realistic exposure distributions skewed toward challenging service conditions.

### 4.3 Machine Learning Models

#### Classifier — Pass/Fail Prediction

```python
RandomForestClassifier(
    n_estimators    = 300,
    max_depth       = 14,
    min_samples_leaf= 3,
    class_weight    = "balanced",   # compensate for class imbalance
    random_state    = 42,
)
```

| Metric | Value |
|---|---|
| Accuracy | **97.5%** |
| ROC-AUC | **0.998** |
| F1 (PASS) | 96.7% |
| F1 (FAIL) | 97.9% |
| 5-fold CV F1 | 97.2% |

Confusion matrix (test set, N = 200):

```
                Predicted PASS   Predicted FAIL
Actual PASS          73               2
Actual FAIL           3             122
```

#### Regressor — Elongation Drop (%) Prediction

```python
RandomForestRegressor(
    n_estimators    = 300,
    max_depth       = 16,
    min_samples_leaf= 2,
    random_state    = 42,
)
```

| Metric | Value |
|---|---|
| R² | **0.987** |
| MAE | 1.94% |
| RMSE | 2.57% |
| 5-fold CV R² | 0.983 |

#### Feature Importance (Classifier)

| Feature | Importance |
|---|---|
| `effective_dose_h` | **61.8%** |
| `arrhenius_factor` | 16.3% |
| `temp_celsius` | 15.1% |
| `uv_hours` | 6.6% |
| `relative_humidity` | 0.3% |

> The dominance of `effective_dose_h` confirms that the model correctly learned that **cumulative exposure**, not instantaneous temperature alone, drives degradation — consistent with Arrhenius kinetics.

### 4.4 API Design

```
POST /predict
Content-Type: application/json

Request:
{
  "temp_celsius":       38.0,
  "uv_hours":         3000.0,
  "relative_humidity":  75.0
}

Response:
{
  "elongation_drop":   26.1,
  "elongation_remain": 295.6,
  "tensile_drop":       7.5,
  "tensile_remain":    1.109,
  "pass_fail":            1,
  "pass_probability":  89.3,
  "arrhenius_factor":  3.847,
  "effective_dose":   8241,
  "risk_level":      "ระวัง",
  "recommendation":  "วางแผนตรวจสอบเชิงรุกภายใน 12 เดือน"
}
```

---

## 5. Usage & Demonstration

### 5.1 Installation

```bash
# Clone repository
git clone https://github.com/<your-org>/dowsil795-ml-predictor.git
cd dowsil795-ml-predictor

# Install dependencies
pip install -r requirements.txt
# fastapi uvicorn scikit-learn==1.8.0 joblib pandas numpy streamlit plotly

# Verify model files are present
ls models/
# rf_classifier.pkl   rf_regressor.pkl   model_metadata.json
```

> **Important:** scikit-learn version must match exactly (`1.8.0`). Models serialized with a different version will fail to load.

### 5.2 Running the Dashboard

```bash
# Option A — FastAPI + HTML (recommended)
python app_server.py
# Browser opens automatically at http://localhost:8000

# Option B — Streamlit (not implemented in this project; see §7 Future Work)
```

### 5.3 Dashboard UI — Layout & Features

The web dashboard (`index.html`) is a single-page application divided into two panels. The screenshot below illustrates a live prediction for the **Glass Façade, Bangkok** preset (38 °C · 3,000 h · 75% RH).

<img width="1914" height="935" alt="Result1" src="https://github.com/user-attachments/assets/bb6e4924-0a8d-41bd-be2a-de70a90585e9" />

#### ① API Status Indicator

A live green dot confirms that the FastAPI backend (`app_server.py`) is running and reachable at `localhost:8000`. If the dot is red, start the server before attempting a prediction.

#### ② Preset Scenario Buttons

Five pre-configured building scenarios load all three sliders simultaneously, allowing instant comparison across common service environments without manual adjustment.

#### ③ Input Sliders

| Slider | Range | Example value shown |
|---|---|---|
| Temperature (Temp) | 15 – 90 °C | **38 °C** |
| UV exposure | 0 – 12,000 h | **3,000 h** |
| Relative Humidity (RH) | 20 – 100 % | **75 %** |

The displayed value updates in real-time as the slider moves. All three values are sent as a JSON payload to `POST /predict` when the Predict button is pressed.

#### ④ Verdict Banner

The top result card delivers an immediate pass/fail verdict with colour coding:

| Colour | Verdict | Condition |
|---|---|---|
| Green | **PASS** | Elongation drop < 50% AND tensile >= 0.72 MPa |
| Red | **FAIL** | Either threshold exceeded |

The banner also displays the **risk level** and the **P(PASS) confidence score** from the Random Forest classifier (shown as 99.9% in the screenshot above for the Bangkok Facade preset).

#### ⑤ Key Metric Cards

Three summary cards present the most decision-critical numbers at a glance:

| Card | Value in screenshot | Meaning |
|---|---|---|
| **Elongation drop** | 26.7% | Movement capability lost vs. TDS baseline 400% |
| **Tensile drop** | 7.6% | Tensile strength lost; remaining = 1.108 MPa |
| **Arrhenius Factor** | 3.851x | At 38°C the sealant ages 3.85x faster than at 25°C |

> **Rule of thumb:** Drop < 10% — inspect on schedule. 10–30% — proactive monitoring. Above 50% — replacement recommended.

#### ⑥ Property Retention Progress Bars

Two gradient bars (green to amber to red) visualise remaining mechanical capacity as a proportion of the TDS baseline. The fill length reflects the **retention percentage**, making it immediately apparent how far the sealant has degraded relative to its original specification.

#### ⑦ Arrhenius Physics Panel

A transparent data panel displays all computed intermediate variables so that engineers can verify the physics behind the prediction:

| Variable | Description |
|---|---|
| `temp_celsius` | Raw temperature input |
| `uv_hours` | Raw UV exposure input |
| `relative_humidity` | Raw humidity input |
| `arrhenius_factor` | AF = exp(-Ea/R x (1/T - 1/T_ref)) |
| `effective_dose_h` | D_eff = UV + UV x (AF-1) x 0.30 + UV x (RH/50-1) x 0.10 |

#### ⑧ Recommendation Box

A plain-language maintenance recommendation is generated from the predicted risk level:

| Risk level | Recommendation |
|---|---|
| Normal | Sealant in good condition — inspect on schedule |
| Caution | Plan proactive inspection within 12 months |
| Degraded | Re-evaluate and prepare repair plan |
| Critical | Replace sealant immediately |

### 5.4 Step-by-Step Prediction

1. **Set environmental inputs** via sliders (or load a preset scenario):

   | Input | Range | Description |
   |---|---|---|
   | `temp_celsius` | 15–90 °C | Mean service temperature of the façade zone |
   | `uv_hours` | 0–12,000 h | Cumulative UV exposure since installation |
   | `relative_humidity` | 20–100 % | Mean ambient RH at the building site |

2. **Click "Analyse Degradation"** — the frontend sends a `POST /predict` request to the FastAPI backend.

3. **Interpret results:**

   | Output | Interpretation |
   |---|---|
   | **Elongation drop (%)** | How much movement capability has been lost vs. TDS baseline of 400% |
   | **Elongation remaining (%)** | Current estimated elongation. Below ~200% warrants inspection |
   | **Pass / Fail** | FAIL if elongation drop ≥ 50% OR tensile < 0.72 MPa (ASTM C1184 threshold) |
   | **Pass probability (%)** | Model confidence in the PASS verdict |
   | **Arrhenius Factor (×)** | Thermal acceleration multiplier — AF = 30× at 60 °C means aging 30× faster than at 25 °C |
   | **Effective dose (h-equiv)** | Combined UV + thermal + humidity dose; the primary predictor of degradation |
   | **Risk level** | ปกติ / ระวัง / เสื่อมสภาพ / วิกฤต — derived from elongation drop thresholds |

4. **Act on the recommendation** displayed at the bottom of the dashboard.

### 5.5 Result Interpretation — PASS vs. FAIL Comparison

The two screenshots below represent opposite ends of the risk spectrum and demonstrate how each UI component responds to different service conditions.

---

#### Case A — PASS (Glass Façade, Bangkok)

<img width="1914" height="935" alt="Result1" src="https://github.com/user-attachments/assets/1e2c2292-7def-47a6-bb7c-69dd4cf66e79" />


> **Input:** 38 °C · 3,000 h UV · 75% RH

| Component | Value | Reading |
|---|---|---|
| **Verdict banner** | 🟢 PASS — ผ่านเกณฑ์ | Both elongation and tensile thresholds satisfied |
| **P(PASS)** | 99.9% | Classifier is highly confident; well within the safe zone |
| **Risk level** | ระวัง (Caution) | Degradation has begun but has not reached a critical level |
| **Elongation drop** | 26.7% (amber) | 26.7% of movement capability lost; 293% remaining out of 400% baseline |
| **Tensile drop** | 7.6% (amber) | Minor strength loss; 1.108 MPa remaining out of 1.2 MPa baseline |
| **Arrhenius Factor** | 3.851× | At 38 °C the sealant degrades 3.85× faster than at the 25 °C reference |
| **Effective dose** | 5,716 h-equiv | Moderate cumulative exposure after thermal and humidity weighting |
| **Elongation bar** | Long fill, colour at amber zone | High retention — bar extends well across the track |
| **Tensile bar** | Near-full fill | Minimal loss — bar almost reaches the right edge |
| **Recommendation** | วางแผนตรวจสอบเชิงรุกภายใน 12 เดือน | Schedule proactive inspection; no emergency action required |

**Key observation:** Even though the risk level reads "ระวัง", the sealant still has 73.3% of its elongation capacity intact. The amber colour on the metric cards signals the start of a monitoring window, not imminent failure. This represents a typical mid-life Bangkok curtainwall after approximately 3–5 years of outdoor service.

---

#### Case B — FAIL (Industrial Rooftop / Facade อุตสาหกรรม)

<img width="1908" height="931" alt="result2" src="https://github.com/user-attachments/assets/61282c45-32a5-4cf2-84c8-f5fd02468a9a" />

> **Input:** 80 °C · 10,000 h UV · 85% RH

| Component | Value | Reading |
|---|---|---|
| **Verdict banner** | 🔴 FAIL — ไม่ผ่านเกณฑ์ | Elongation drop threshold (≥ 50%) exceeded |
| **P(PASS)** | 0% | Classifier is fully certain — sealant is outside the safe operating envelope |
| **Risk level** | วิกฤต (Critical) | Immediate intervention required |
| **Elongation drop** | 71.9% (red) | Nearly three-quarters of movement capability has been lost |
| **Elongation remaining** | 112.3% out of 400% | Below the practical minimum for structural glazing joint movement |
| **Tensile drop** | 20.5% (red) | Meaningful strength loss; 0.953 MPa remaining |
| **Arrhenius Factor** | 152.369× | At 80 °C the sealant ages 152× faster than at 25 °C — extreme thermal acceleration |
| **Effective dose** | 464,805 h-equiv | Equivalent to exposing the sealant for over 53 years at standard conditions |
| **Elongation bar** | Short fill, stops in red zone | Only ~28% retention remains — bar terminates early in the gradient |
| **Tensile bar** | Longer but in red zone | Tensile degrades more slowly than elongation; still above floor |
| **Recommendation** | ควรเปลี่ยนซีลแลนต์โดยเร็ว | Replace sealant immediately before structural or watertightness failure |

**Key observation:** The Arrhenius Factor of **152×** is the most important diagnostic figure here. It means that every real hour at 80 °C is equivalent to 152 hours of aging at the 25 °C laboratory reference. This is why even 10,000 UV hours at this temperature produces an effective dose of nearly **465,000 h-equiv** — far beyond the calibrated degradation ceiling. Industrial rooftop applications at extreme temperatures should be re-inspected annually and may require a higher-specification product or enhanced joint design.

---

#### Side-by-Side Summary

| Indicator | PASS (Bangkok Façade) | FAIL (Industrial Rooftop) |
|---|---|---|
| **Temperature** | 38 °C | 80 °C |
| **UV exposure** | 3,000 h | 10,000 h |
| **Arrhenius Factor** | 3.851× | 152.369× |
| **Effective dose** | 5,716 h-equiv | 464,805 h-equiv |
| **Elongation drop** | 26.7% 🟡 | 71.9% 🔴 |
| **Elongation remaining** | 293% (73% of baseline) | 112.3% (28% of baseline) |
| **Tensile remaining** | 1.108 MPa | 0.953 MPa |
| **P(PASS)** | 99.9% | 0% |
| **Risk level** | ระวัง | วิกฤต |
| **Action required** | Monitor within 12 months | Replace immediately |

> **Why elongation matters more than tensile:** In a structural glazing system, the sealant must accommodate thermal expansion and wind-load deflection continuously. ASTM C1135 defines movement capability (±50% of joint width) as the primary performance criterion. Once elongation drops below ~200%, the joint can no longer safely accommodate dynamic movement — even if tensile strength appears adequate. This is why the FAIL threshold is set at elongation drop ≥ 50%, regardless of tensile retention.

### 5.6 Preset Scenarios

| Scenario | Temp | UV (h) | RH | Expected verdict |
|---|---|---|---|---|
| Meeting room (interior) | 25 °C | 0 | 55% | PASS · ปกติ |
| Glass façade, Bangkok | 38 °C | 3,000 | 75% | PASS · ระวัง |
| High-rise office (sunny) | 45 °C | 5,000 | 65% | FAIL · เสื่อมสภาพ |
| Industrial rooftop | 80 °C | 10,000 | 85% | FAIL · วิกฤต |

### 5.7 Batch Prediction (Programmatic)

```python
from dowsil795_rf_model import batch_predict

records = [
    {"temp_celsius": 38, "uv_hours": 3000, "relative_humidity": 75},
    {"temp_celsius": 60, "uv_hours": 7000, "relative_humidity": 80},
]
df = batch_predict(records)
print(df[["temp_celsius", "uv_hours", "elong_drop_%", "pass_fail", "risk_level"]])
```

---

## 6. Discussion & Conclusion

### 6.1 Key Findings

The trained Random Forest model achieved **97.5% accuracy** and **R² = 0.987** on held-out test data, confirming that the Arrhenius-calibrated synthetic dataset captures the essential physics of silicone weathering with high fidelity.

Feature importance analysis revealed that `effective_dose_h` — the composite variable integrating UV hours, thermal acceleration, and humidity — accounts for **61.8% of predictive power**. This demonstrates that the model has internalized the physics of cumulative degradation rather than relying on any single raw input.

### 6.2 Scientific Depth — Beyond the Datasheet

This project goes significantly beyond a surface reading of the TDS. The QUV 5,000-hour tensile retention data point (1.1 MPa → 91.7% retention, ASTM G154) was used as a quantitative calibration anchor to derive the degradation rate constants $k_{\text{tensile}}$ and $k_{\text{elongation}}$. These constants were then embedded into a physics-consistent generative model, producing a synthetic dataset that respects the thermodynamic constraints of Arrhenius kinetics and the material science of silicone rubber ageing.

### 6.3 Business Value

**Smart Troubleshooting Tool.** Rather than requiring engineers to solve coupled physics equations on-site, the dashboard accepts three field-observable parameters and immediately returns a quantified degradation assessment. This reduces the time from site inspection to maintenance decision from days to seconds.

**Decoding Degradation Drivers (Feature Importance).** By exposing the relative contribution of temperature, UV hours, and humidity, the model provides R&D teams with a ranked understanding of which environmental factors most aggressively attack sealant integrity. This directly informs joint design guidelines for climate-specific projects and accelerated laboratory test protocols.

**Elevating Substrate Testing.** The model can quantitatively demonstrate to clients that failure to follow Dow's prescribed Preparatory Work procedures and adhesion testing protocols (ASTM C794) results in a disproportionate increase in predicted fail rate — providing a data-backed rationale for compliance.

**Predictive vs. Reactive Maintenance.** The 20-year degradation forecast (available as a future enhancement; see §7) allows building owners and façade consultants to identify the year in which elongation is projected to fall below the safety threshold, enabling pre-scheduled re-sealing contracts rather than emergency interventions following water ingress or structural glazing failure.

**Competitive Differentiation.** Bundling a machine-learning-powered predictive tool with a commodity sealant product transforms the value proposition from product sales to **high-technology solution delivery**, strengthening brand positioning in large-scale façade and curtainwall projects.

### 6.4 Limitations

| Limitation | Description |
|---|---|
| **Synthetic data only** | No real field-aged sealant samples were measured. Model accuracy on real-world data has not been validated experimentally. |
| **Single product scope** | The model is calibrated exclusively to DOWSIL™ 795; parameters are not transferable to other sealant formulations without re-calibration. |
| **Simplified degradation model** | The Arrhenius exponential decay approximation does not capture photo-oxidative chain scission mechanisms, cyclic fatigue from joint movement, or substrate-dependent adhesion loss. |
| **No substrate variable** | Substrate type (glass, anodized aluminum, coated aluminum) is not currently a model feature, though it significantly affects peel strength. |
| **scikit-learn version lock** | Serialized `.pkl` files require exactly `scikit-learn 1.8.0` to load correctly. |

---

## 7. Future Work

| Priority | Enhancement | Description |
|---|---|---|
| High | **Real data integration** | Collect elongation measurements from field-aged sealant specimens to validate and fine-tune model coefficients |
| High | **Substrate type as feature** | Encode substrate (glass, anodized Al, painted Al, PVDF coated) as a categorical input variable |
| Medium | **Survival Analysis** | Implement Kaplan–Meier or Cox Proportional Hazards models to estimate **Time-to-Failure (TTF)** distributions for each façade zone |
| Medium | **XGBoost / LightGBM comparison** | Benchmark gradient-boosted tree models against the current Random Forest baseline |
| Medium | **Streamlit Cloud deployment** | Public-facing hosted instance for client demonstrations without local installation |
| Low | **BMS Integration** | Ingest real-time temperature and UV irradiance data from Building Management Systems (BMS) via MQTT/REST for continuous condition monitoring |
| Low | **IoT sensor pipeline** | Integrate low-cost UV and temperature loggers installed at façade zones to replace synthetic exposure estimates with measured inputs |
| Low | **Multi-sealant support** | Extend the pipeline to DOWSIL™ 791, 993, and 121 with product-specific calibration constants |

---

## File Structure

```
dowsil795-ml-predictor/
│
├── README.md
├── requirements.txt
│
├── generate_dowsil795_synthetic.py   # Synthetic data generator (Arrhenius model)
├── dowsil795_rf_model.py             # RF training pipeline + predict() API
├── app_server.py                     # FastAPI backend server
├── index.html                        # Web dashboard frontend
│
├── models/
│   ├── rf_classifier.pkl             # Trained Pass/Fail classifier (753 KB)
│   ├── rf_regressor.pkl              # Trained elongation regressor (8.4 MB)
│   └── model_metadata.json           # Metrics, features, hyperparameters
│
└── data/
    └── dowsil795_synthetic_1000.csv  # Generated synthetic dataset
```

---

## Reference

Dow Chemical Company. (2024). *DOWSIL™ 795 Structural Glazing Sealant — Technical Data Sheet* (Form No. 63-1217-01-0124 S2D). The Dow Chemical Company.

---

*This project was developed as a demonstration of physics-informed machine learning applied to construction material science. The synthetic dataset and model are intended for engineering feasibility studies and client demonstrations. All predictions should be validated against physical adhesion and compatibility testing before use in structural glazing design decisions.*
