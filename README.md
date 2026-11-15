# KisanMandi AI: Agricultural Price Intelligence & Farmer Action Advisory

**Course:** CSE 111 – Fundamentals of AI  
**Institution:** SASTRA Deemed University  
**Project Title:** Multi-Horizon Crop Price Forecasting and Spatial Arbitrage Optimization for Distress-Sale Prevention in Agricultural Mandis  

---

## 🌟 Project Overview
Smallholder farmers in India frequently suffer severe financial losses due to extreme market price volatility and localized supply gluts (the Cobweb Phenomenon). **KisanMandi AI** is an advanced decision-support platform that provides:

1. **Multi-Horizon Probabilistic Price Forecasting:** Uses LightGBM with Quantile Loss ($P_{10}, P_{50}, P_{90}$) to predict 15-day and 30-day wholesale commodity prices.
2. **Action-Oriented Farmer Advisory:** Translates complex ML outputs into a 1-2-3 plain-language action checklist (Timing, Destination Mandi, and Expected Money in Hand).
3. **Personalized Expectation Optimization:** Adapts recommendations based on farmer priorities (*Maximize Profit*, *Urgent Cash*, *Risk-Averse*) and target price expectations.
4. **Inter-District Spatial Transport Arbitrage:** Compares synchronized prices across 24 Tamil Nadu mandis and computes net freight margins deducting fuel, labor, cess, and transit shrinkage.
5. **Storage & Spoilage Optimization:** Balances price appreciation against ICAR produce moisture shrinkage rates and warehousing costs.

---

## 🚀 How to Run the Project

### 1. Prerequisites & Environment Setup
Open your terminal / command prompt in the extracted project folder and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
Run the Streamlit dashboard:

```bash
streamlit run app.py
```

Open your browser and navigate to:
```text
http://localhost:8501
```

---

## 📂 Project Directory Structure

```text
aji proj/
│
├── app.py                            # Streamlit web application dashboard
├── requirements.txt                  # Python dependencies
├── README.md                         # Quick start guide & architecture overview
├── viva_defense_guide.md             # Comprehensive Viva-Voce defense & rubric reference
├── Agriculture_price_dataset.csv     # Primary Agmarknet dataset (737,392 rows, 55.2 MB)
│
├── .streamlit/
│   └── config.toml                   # Unified agricultural theme configuration
│
├── src/
│   ├── data_pipeline.py              # Data cleaning, unit normalization & 31 engineered features
│   ├── models.py                     # LightGBM (Quantile), Random Forest & Ridge baseline models
│   └── decision_engine.py            # Sell-vs-Hold optimizer & Inter-Mandi Spatial Arbitrage
│
└── models/
    └── thanjavur_onion_15d.joblib    # Serialized pre-trained model weights
```

---

## 📊 Dataset Ground Truth
* **Source:** Official Agmarknet (Ministry of Agriculture & Farmers Welfare, GoI)
* **Total Volume:** 737,392 rows across 30 Indian states.
* **Tamil Nadu Scope:** 66,602 rows across 24 districts (Thanjavur, Coimbatore, Tiruchirappalli, Dindigul, Salem, Madurai, etc.) and 184 local mandis.
* **Commodities:** High-volatility perishables including Onion (Bellary) and Potato (Red Nanital/Local).

---

## 🎓 SASTRA CSE 111 Rubric Compliance
Refer to `viva_defense_guide.md` for complete answers to potential Viva-Voce questions, mathematical formulas, and phase-by-phase evaluation rubrics.
