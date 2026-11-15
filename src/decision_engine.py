"""
KisanMandi AI - Operational Decision & Arbitrage Engines
Includes:
1. Sell-Now vs. Hold Financial Optimizer (incorporating biological weight decay & storage costs)
2. Spatial Arbitrage Engine (Inter-district transport economics across all Tamil Nadu mandis)
3. Cobweb Volatility & Market Crash Warning Index
"""

import numpy as np
import pandas as pd

# Biological weight decay rates (% per day) based on ICAR post-harvest guidelines
STORAGE_PROFILES = {
    'Ambient Shed (On-Farm)': {
        'Onion': {'decay_rate': 0.008, 'cost_per_qtl_day': 0.0},
        'Potato': {'decay_rate': 0.005, 'cost_per_qtl_day': 0.0},
        'Tomato': {'decay_rate': 0.025, 'cost_per_qtl_day': 0.0},
        'Wheat': {'decay_rate': 0.0005, 'cost_per_qtl_day': 0.0},
        'Rice': {'decay_rate': 0.0005, 'cost_per_qtl_day': 0.0}
    },
    'Zero Energy Cool Chamber (ZECC)': {
        'Onion': {'decay_rate': 0.004, 'cost_per_qtl_day': 1.5},
        'Potato': {'decay_rate': 0.0025, 'cost_per_qtl_day': 1.5},
        'Tomato': {'decay_rate': 0.008, 'cost_per_qtl_day': 2.0},
        'Wheat': {'decay_rate': 0.0003, 'cost_per_qtl_day': 1.0},
        'Rice': {'decay_rate': 0.0003, 'cost_per_qtl_day': 1.0}
    },
    'Commercial Cold Storage': {
        'Onion': {'decay_rate': 0.0015, 'cost_per_qtl_day': 4.0},
        'Potato': {'decay_rate': 0.0008, 'cost_per_qtl_day': 3.5},
        'Tomato': {'decay_rate': 0.003, 'cost_per_qtl_day': 5.0},
        'Wheat': {'decay_rate': 0.0001, 'cost_per_qtl_day': 2.5},
        'Rice': {'decay_rate': 0.0001, 'cost_per_qtl_day': 2.5}
    }
}

# District geographic coordinates across Tamil Nadu
TN_DISTRICT_COORDS = {
    'Thanjavur': (10.7870, 79.1378),
    'Thiruchirappalli': (10.7905, 78.7047),
    'Dindigul': (10.3673, 77.9803),
    'Madurai': (9.9252, 78.1198),
    'Coimbatore': (11.0168, 76.9558),
    'Salem': (11.6643, 78.1460),
    'Erode': (11.3410, 77.7172),
    'Thiruvarur': (10.7725, 79.6365),
    'Nagapattinam': (10.7656, 79.8424),
    'Thirunelveli': (8.7139, 77.7567),
    'Kancheepuram': (12.8342, 79.7036),
    'Vellore': (12.9165, 79.1325),
    'The Nilgiris': (11.4102, 76.6950),
    'Namakkal': (11.2189, 78.1674),
    'Dharmapuri': (12.1211, 78.1582),
    'Krishnagiri': (12.5186, 78.2137),
    'Cuddalore': (11.7480, 79.7714),
    'Villupuram': (11.9401, 79.4861),
    'Virudhunagar': (9.5680, 77.9624),
    'Theni': (10.0104, 77.4768),
    'Ramanathapuram': (9.3639, 78.8395),
    'Karur': (10.9601, 78.0766),
    'Perambalur': (11.2342, 78.8821),
    'Ranipet': (12.9224, 79.3326)
}

def calculate_road_distance(origin, destination):
    """
    Computes road distance (km) using Haversine with a 1.25 road winding factor.
    """
    if origin.title() == destination.title():
        return 0
    c1 = TN_DISTRICT_COORDS.get(origin.title())
    c2 = TN_DISTRICT_COORDS.get(destination.title())
    if not c1 or not c2:
        return 120  # Fallback reasonable inter-district distance
    lat1, lon1 = np.radians(c1)
    lat2, lon2 = np.radians(c2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return int(round(6371 * c * 1.25))

def optimize_sell_vs_hold(current_price, forecasted_price_30d, quantity_qtl=50, commodity='Onion', facility='Ambient Shed (On-Farm)'):
    """
    Computes the optimal sell day t* by balancing price appreciation vs weight loss and storage cost.
    """
    profile = STORAGE_PROFILES.get(facility, STORAGE_PROFILES['Ambient Shed (On-Farm)']).get(commodity, {'decay_rate': 0.005, 'cost_per_qtl_day': 1.0})
    delta = profile['decay_rate']
    cost_per_day = profile['cost_per_qtl_day']
    
    current_revenue = current_price * quantity_qtl
    
    days = np.arange(0, 31)
    price_path = current_price + (forecasted_price_30d - current_price) * (1 - np.exp(-days / 12)) / (1 - np.exp(-30 / 12))
    qty_path = quantity_qtl * np.maximum(0, 1.0 - (delta * days))
    cumulative_storage_cost = cost_per_day * quantity_qtl * days
    
    net_revenue_path = (price_path * qty_path) - cumulative_storage_cost
    profit_delta_path = net_revenue_path - current_revenue
    
    best_day = int(np.argmax(net_revenue_path))
    max_net_gain = float(profit_delta_path[best_day])
    
    trajectory_df = pd.DataFrame({
        'Day': days,
        'Projected_Price': np.round(price_path, 2),
        'Effective_Qty_Qtl': np.round(qty_path, 2),
        'Cumulative_Cost': np.round(cumulative_storage_cost, 2),
        'Net_Revenue': np.round(net_revenue_path, 2),
        'Net_Profit_Delta': np.round(profit_delta_path, 2)
    })
    
    if best_day > 2 and max_net_gain > 500:
        recommendation = f"HOLD FOR {best_day} DAYS"
        action_detail = (
            f"Holding is projected to earn an additional Rs. {max_net_gain:,.0f} net surplus "
            f"(+{max_net_gain/current_revenue*100:.1f}%) after accounting for {delta*best_day*100:.1f}% "
            f"produce shrinkage and storage fees."
        )
        status_color = "success"
    else:
        recommendation = "SELL IMMEDIATELY"
        action_detail = (
            f"Produce decay and storage costs will erode future price gains. "
            f"Selling today at Rs. {current_price:,.0f}/Qtl maximizes total realization."
        )
        status_color = "warning"
        
    return {
        'recommendation': recommendation,
        'best_day': best_day,
        'max_net_gain': round(max_net_gain, 2),
        'current_revenue': round(current_revenue, 2),
        'optimal_revenue': round(float(net_revenue_path[best_day]), 2),
        'action_detail': action_detail,
        'status_color': status_color,
        'trajectory': trajectory_df
    }

def calculate_spatial_arbitrage(df, origin_district='Thanjavur', commodity='Onion', quantity_qtl=50, date=None):
    """
    Compares prices between the origin mandi and all other reporting Tamil Nadu mandis.
    Dynamically computes road distances, fuel, labor, cess, and net profit.
    """
    tn_df = df[df['STATE_CLEAN'] == 'Tamil Nadu'].copy()
    if commodity:
        tn_df = tn_df[tn_df['Commodity_Clean'].str.lower() == commodity.lower()]
        
    if date is not None:
        target_date = pd.to_datetime(date)
        window = tn_df[(tn_df['Price Date'] >= target_date - pd.Timedelta(days=5)) &
                       (tn_df['Price Date'] <= target_date)]
    else:
        latest_date = tn_df['Price Date'].max()
        window = tn_df[tn_df['Price Date'] >= latest_date - pd.Timedelta(days=7)]
        
    district_prices = window.groupby('District_Clean')['Modal_Price'].median().to_dict()
    
    origin_title = origin_district.title()
    origin_price = district_prices.get(origin_title, None)
    if origin_price is None:
        origin_price = window['Modal_Price'].median()
        
    results = []
    for dest, dest_price in district_prices.items():
        if dest.lower() == origin_district.lower():
            continue  # Do not compare origin against itself
            
        dist_km = calculate_road_distance(origin_title, dest)
        if dist_km <= 5:
            continue
            
        price_diff = dest_price - origin_price
        gross_gain = price_diff * quantity_qtl
        
        transport_cost = dist_km * 24.0
        mandi_cess = (dest_price * quantity_qtl) * 0.015
        transit_spoilage = (origin_price * quantity_qtl) * (0.005 * (dist_km / 100.0))
        total_costs = transport_cost + mandi_cess + transit_spoilage
        
        net_profit = gross_gain - total_costs
        roi_pct = (net_profit / ((origin_price * quantity_qtl) + total_costs)) * 100
        
        results.append({
            'Destination_Mandi': dest,
            'Distance_KM': dist_km,
            'Origin_Price_Rs_Qtl': round(origin_price, 2),
            'Dest_Price_Rs_Qtl': round(dest_price, 2),
            'Price_Spread': round(price_diff, 2),
            'Transport_Cost': round(transport_cost, 2),
            'Net_Profit_Gain': round(net_profit, 2),
            'ROI_Pct': round(roi_pct, 1),
            'Is_Profitable': net_profit > 0
        })
        
    if len(results) == 0:
        return pd.DataFrame(), origin_price
        
    results_df = pd.DataFrame(results).sort_values('Net_Profit_Gain', ascending=False).reset_index(drop=True)
    return results_df, origin_price

def assess_cobweb_volatility(recent_series):
    """
    Computes market volatility, price deceleration, and Cobweb supply glut risk.
    """
    prices = recent_series['Modal_Price'].tail(30).values
    if len(prices) < 14:
        return {'risk_level': 'Normal', 'score': 20, 'advice': 'Market is behaving normally.'}
        
    cv = (np.std(prices[-14:]) / (np.mean(prices[-14:]) + 1e-5)) * 100
    velocity = prices[-1] - prices[-7]
    prev_velocity = prices[-7] - prices[-14]
    
    spreads = (recent_series['Max_Price'] - recent_series['Min_Price']).tail(14).values
    avg_spread_ratio = np.mean(spreads / (prices[-14:] + 1e-5)) * 100
    
    risk_score = min(100, int((cv * 2.5) + (max(0, -velocity) / (prices[-1] + 1e-5) * 50) + (avg_spread_ratio * 0.5)))
    
    if risk_score >= 65:
        risk_level = "High Crash Risk (Supply Glut Alert)"
        advice = "Rapid downward price velocity indicates market saturation. Immediate liquidation recommended."
    elif risk_score >= 40:
        risk_level = "Moderate Volatility (Caution)"
        advice = "Market experiencing price divergence. Monitor arrivals closely."
    else:
        risk_level = "Stable Market Trend"
        advice = "Predictable demand-supply equilibrium. Safe for holding strategies."
        
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'cv_pct': round(cv, 2),
        'velocity_7d': round(velocity, 2),
        'advice': advice
    }
