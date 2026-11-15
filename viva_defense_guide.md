# KisanMandi AI: Project Report & Viva-Voce Defense Guide
**Course:** CSE 111 – Fundamentals of AI | **Institution:** SASTRA Deemed University  
**Project Title:** *Multi-Horizon Crop Price Forecasting and Spatial Arbitrage Optimization for Distress-Sale Prevention in Agricultural Mandis*

---

## 1. Quick Launch Instructions

To launch the interactive classic-meets-modern dashboard:

```bash
# From the project root folder:
streamlit run app.py
```

The web application will open in your browser at `http://localhost:8501`.

---

## 2. Google Form Submission Reference

| Form Field | Exact Entry to Submit |
| :--- | :--- |
| **Tentative Project Title** | *Multi-Horizon Crop Price Forecasting and Spatial Arbitrage Optimization for Distress-Sale Prevention in Agricultural Mandis* |
| **Agricultural data collected from state** | **Tamil Nadu** |
| **Agricultural data collected from district** | **Thanjavur** *(and Cauvery Delta: Tiruchirappalli, Thiruvarur, Dindigul)* |
| **Crop type and Crop variety** | *High-volatility perishables: Onion (Bellary) and Potato (Red Nanital / Local)* |
| **Type of problem you want to solve** | ☑ **Crop price forecasting**<br>☑ **Trend analysis**<br>☑ **Supply chain and demand forecasting of agricultural crops** |
| **Data source URL(s)** | • `https://agmarknet.gov.in/home`<br>• `https://data.gov.in/`<br>• `https://upag.gov.in/` |
| **Type of learning methods** | ☑ **Supervised learning** (Gradient Boosted Trees, Quantile Regression, Time-Series Lags) |
| **Type of LLM and Coding software** | ☑ `https://www.python.org/`<br>☑ `https://colab.research.google.com/` |

### Official Project Abstract (276 Words — Strictly $\le 300$ Words):
> Smallholder farmers in India frequently suffer severe financial losses due to extreme market price volatility and localized supply gluts—a structural failure known as the agricultural Cobweb Phenomenon. Lacking forward price visibility, farmers often resort to distress selling at unremunerative prices immediately post-harvest. This project develops an advanced, multi-perspective Machine Learning and Spatial Econometric decision-support system to forecast mandi commodity prices and optimize post-harvest marketing strategies without requiring on-farm hardware.
>
> Focusing on high-volatility perishable commodities (Onion, Potato) across agricultural markets in Tamil Nadu (Thanjavur, Tiruchirappalli, Dindigul, Salem), the system ingests 737,392 multi-year daily market records from Agmarknet, incorporating modal prices, minimum/maximum trading spreads, and seasonal calendar harmonics. An automated data cleaning pipeline standardizes retail reporting disparities (converting Uzhavar Sandhai ₹/kg entries to wholesale ₹/Quintal standards) and engineers 31 temporal features capturing price momentum, rolling volatility, and annual cyclicality.
>
> Multi-horizon probabilistic forecasts (15 and 30 days) are generated using an ensemble of Gradient Boosted Decision Trees (LightGBM) with quantile loss functions alongside Random Forest and Ridge regression baselines, outputting $P_{10}$, $P_{50}$, and $P_{90}$ predictive uncertainty bands. Beyond forecasting, the platform integrates two operational engines: (1) a mathematical "Sell-Now vs. Hold" optimizer that balances projected price increases against biological moisture shrinkage rates and warehousing fees, and (2) an Inter-District Spatial Transport Arbitrage module calculating net freight margins.
>
> Model performance is evaluated using RMSE, MAE, MAPE, $R^2$, and Mean Directional Accuracy (MDA). Deployed via a classic-modern web interface, this tool empowers farmers to transition from reactive price-takers to proactive market strategists.

---

## 3. Rubric-Aligned Phase Breakdown

### Phase 1: Problem Identification & Dataset Understanding (10 Marks)
* **The Problem:** The Cobweb Cycle in Indian agriculture causes dramatic price crashes (from ₹50/kg down to ₹5/kg in a few weeks). Farmers lack forward visibility and sell immediately out of fear of spoilage.
* **The Dataset:** Official Agmarknet daily telemetry (`Agriculture_price_dataset.csv`):
  * **737,392 rows** spanning 2023 to 2025 across 30 states.
  * **66,602 rows** specifically for Tamil Nadu across 24 districts and 184 markets.
  * **2,515 rows** for Thanjavur district (Kumbakonam, Pattukottai, Papanasam, and Thanjavur Uzhavar Sandhais).
  * **Unit Standardization:** Identified and resolved a reporting anomaly where ~2% of Uzhavar Sandhai records were entered in ₹/kg instead of the standard ₹/Quintal.

---

### Phase 2: Methodology & Conceptual Understanding (20 Marks)
* **Supervised Machine Learning:**
  * **LightGBM Regressor (Champion):** Trained with both L2 (mean) loss and Pinball/Quantile loss ($\alpha = 0.10, 0.90$) to provide probabilistic lower and upper prediction envelopes.
  * **Random Forest Regressor:** Non-linear ensemble baseline.
  * **Ridge Regression:** Linear $L_2$-regularized baseline.
* **Feature Engineering (31 Features):**
  * *Lags:* $t-1, t-2, t-3, t-7, t-14, t-21, t-30$
  * *Rolling Statistics:* 7d, 14d, 30d rolling mean, standard deviation (volatility), min, max
  * *Momentum:* $\text{Price}_t / \text{MA}_{7}$, $\text{Price}_t / \text{MA}_{30}$, $(\text{Price}_t - \text{Price}_{t-7}) / \text{Price}_{t-7}$
  * *Cyclical Seasonality:* $\sin(2\pi \cdot \text{day} / 365.25)$, $\cos(2\pi \cdot \text{day} / 365.25)$
* **Mathematical Optimization Formulations:**
  1. **Optimal Sell Day ($t^*$):**
     $$\max_{t \in [0, 30]} \text{Net Revenue}(t) = P(t) \cdot Q_0 \cdot (1 - \delta \cdot t) - (C_{\text{storage}} \cdot Q_0 \cdot t)$$
     where $\delta$ is the daily biological weight shrinkage rate (0.8%/day for onion in ambient shed, 0.15% in cold storage), and $C_{\text{storage}}$ is storage cost.
  2. **Spatial Transport Arbitrage:**
     $$\text{Net Profit} = (P_{\text{dest}} - P_{\text{origin}}) \cdot Q - (\text{Distance} \cdot \text{Rate/km} + \text{Mandi Cess} + \text{Transit Shrinkage})$$

---

### Phase 3: Results & Viva-Voce Defense (50 Marks $\to$ Scaled to 20 Marks)

#### Top 10 Viva-Voce Questions with Model Answers:

**Q1: Why did you choose LightGBM over traditional ARIMA or LSTM?**
> *"ARIMA is univariate and cannot easily incorporate our 31 engineered features such as rolling intra-day spreads, momentum indicators, and cyclical Fourier terms. LSTMs require extensive sequence padding and are computationally heavy. LightGBM handles non-linear interactions, trains in seconds, handles missing historical trading days seamlessly, and directly supports Quantile Loss for probabilistic intervals ($P_{10}, P_{50}, P_{90}$). "*

**Q2: How did you ensure your model does not suffer from data leakage in time-series forecasting?**
> *"We implemented a strict chronological (out-of-time) split: the first 80% of historical dates were used strictly for training, and the final 20% for testing. No future observations or target values leaked into the lag or rolling features."*

**Q3: What is the significance of the $P_{10}$ and $P_{90}$ prediction bands?**
> *"In agricultural trading, point estimates are risky because weather shocks or transport strikes create uncertainty. $P_{10}$ represents the conservative downside price (bearish scenario with 90% confidence of being exceeded), while $P_{90}$ represents the optimistic ceiling. This allows risk-averse farmers to plan conservative budgets."*

**Q4: How does your system account for produce spoilage if a farmer decides to hold?**
> *"In our 'Sell Now vs. Hold' engine, we integrated ICAR post-harvest biological decay coefficients. For instance, Onion loses approximately 0.8% weight per day in ambient sheds due to respiration and moisture loss. Our objective function penalizes the revenue on day $t$ by both physical weight loss and warehouse rent, ensuring holding is only recommended when projected price gains exceed cumulative degradation."*

**Q5: How does the Spatial Arbitrage module work between Tamil Nadu mandis?**
> *"It synchronizes prices on the same trading day between Thanjavur and neighboring markets like Tiruchirappalli (55 km), Dindigul (140 km), and Salem (195 km). It factors in vehicle freight costs (₹24/km round trip for a light commercial vehicle), 1.5% APMC market cess, and 0.5%/100km in-transit shrinkage to report the exact net profit delta."*

**Q6: What is Mean Directional Accuracy (MDA) and why is it important here?**
> *"While RMSE measures numerical error in rupees, MDA measures whether the model correctly predicted if prices will go UP or DOWN relative to today. For farmers deciding whether to harvest or wait, getting the direction right is often more critical than the exact rupee figure."*

**Q7: How did you handle data quality issues in Agmarknet?**
> *"We discovered a unit reporting discrepancy where some Uzhavar Sandhai records were entered in ₹/kg (e.g., ₹44/kg) while standard mandi entries were in ₹/Quintal (₹4,400/Qtl). Our pipeline detected and normalized all prices below ₹150 by a factor of 100."*

**Q8: What are the main limitations of this system?**
> *"The system currently assumes that transport freight rates and storage rental fees are piecewise linear. In future work, real-time diesel price feeds and satellite-based crop acreage data can be integrated to capture unexpected macro-supply surges."*

**Q9: Why does your project not require physical IoT sensors?**
> *"Because price determination in mandis is driven by macro-market trading telemetry, arrival volumes, and inter-district price propagation—not micro-field sensor readings. Using genuine Agmarknet transaction records provides real-world economic ground truth."*

**Q10: What is the Cobweb Phenomenon and how does your app detect it?**
> *"The Cobweb Phenomenon occurs when high prices in season $t$ induce over-planting, leading to an over-supply glut and catastrophic price collapse in season $t+1$. Our app monitors 14-day price velocity, spread volatility, and deceleration to issue early warning alerts before market saturation occurs."*
