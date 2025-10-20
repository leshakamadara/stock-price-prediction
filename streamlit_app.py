import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Try to import yfinance, else set flag
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Apple Stock Price Predictor",
    page_icon=None,
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #000000;
        text-align: center;
        margin-bottom: 10px;
    }
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #FFFFFF;
        }
    }
    .sub-header {
        font-size: 20px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    try:
        with open('stock_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please run the Jupyter notebook first to train the model.")
        return None, None

model, scaler = load_models()

# Header
st.markdown('<div class="main-header">Apple Stock Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict tomorrow\'s closing price using Machine Learning</div>', unsafe_allow_html=True)


# Sidebar
st.sidebar.title("Input Parameters")
st.sidebar.markdown("Enter today's stock data to predict tomorrow's closing price")
st.sidebar.info("Technical features (MA, Volatility, Volume MA) are suggested from historical data, but you can edit them.")

# Initialize session state for all fields if not present
default_fields = {
    'open': 150.0,
    'high': 155.0,
    'low': 149.0,
    'close': 152.0,
    'volume': 50000000,
    'ma_5': 248.15,
    'ma_10': 247.63,
    'volatility': 2.5,
    'volume_ma': 54400000,
    'implied_change': 0.0
}
for k, v in default_fields.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Add fetch-all-fields button
fetch_all = st.sidebar.button("Fetch all fields from Yahoo Finance")

# Helper: load historical data and compute rolling features
@st.cache_resource
def load_historical_data(path='apple_stock.csv'):
    try:
        df_hist = pd.read_csv(path)
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        df_hist = df_hist.sort_values('Date').reset_index(drop=True)
        df_hist['MA_5'] = df_hist['Close'].rolling(window=5).mean()
        df_hist['MA_10'] = df_hist['Close'].rolling(window=10).mean()
        df_hist['Volatility_5'] = df_hist['Close'].rolling(window=5).std()
        df_hist['Volume_MA_63'] = df_hist['Volume'].rolling(window=63).mean()
        return df_hist
    except Exception:
        return None

df_hist = load_historical_data()



# If fetch_all is pressed, fetch all fields from Yahoo Finance and update session state
if fetch_all:
    if not YF_AVAILABLE:
        st.error("yfinance is not installed. Please run: pip install yfinance")
    else:
        try:
            data = yf.download('AAPL', period='70d')  # 70 days to ensure enough for 63-day MA
            closes = data['Close'].dropna()
            highs = data['High'].dropna()
            lows = data['Low'].dropna()
            opens = data['Open'].dropna()
            vols = data['Volume'].dropna()
            if closes.empty or highs.empty or lows.empty or opens.empty or vols.empty:
                st.error("No data returned from Yahoo Finance. Try again later.")
            elif len(closes) < 10 or len(vols) < 63:
                st.error("Not enough recent data to calculate all fields. Need at least 10 closes and 63 volumes.")
            else:
                def to_scalar(val, fallback=0.0, is_int=False):
                    try:
                        if hasattr(val, 'item'):
                            v = val.item()
                        else:
                            v = float(val)
                        if pd.isna(v):
                            return fallback
                        return int(v) if is_int else float(v)
                    except Exception as e:
                        st.warning(f"Debug: Could not convert value {val} to scalar: {e}")
                        return fallback

                # Use the last available row for price/volume fields
                st.session_state['open'] = to_scalar(opens.iloc[-1], 0.0)
                st.session_state['high'] = to_scalar(highs.iloc[-1], 0.0)
                st.session_state['low'] = to_scalar(lows.iloc[-1], 0.0)
                st.session_state['close'] = to_scalar(closes.iloc[-1], 0.0)
                st.session_state['volume'] = to_scalar(vols.iloc[-1], 0, is_int=True)
                st.session_state['ma_5'] = to_scalar(closes[-5:].mean(), 0.0)
                st.session_state['ma_10'] = to_scalar(closes[-10:].mean(), 0.0)
                st.session_state['volatility'] = to_scalar(closes[-5:].std(), 0.0)
                st.session_state['volume_ma'] = to_scalar(vols[-63:].mean(), 0, is_int=True)
                # implied_change remains user input
                st.success("Fetched all fields from Yahoo Finance!")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Stock Data")
    
    # Create input fields in a form
    with st.form("prediction_form"):
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            open_price = st.number_input(
                "Opening Price ($)", 
                min_value=0.0, 
                value=st.session_state['open'], 
                step=0.01,
                key='open',
                help="Today's opening stock price"
            )
            high_price = st.number_input(
                "Highest Price ($)", 
                min_value=0.0, 
                value=st.session_state['high'], 
                step=0.01,
                key='high',
                help="Today's highest stock price"
            )
            low_price = st.number_input(
                "Lowest Price ($)", 
                min_value=0.0, 
                value=st.session_state['low'], 
                step=0.01,
                key='low',
                help="Today's lowest stock price"
            )
            close_price = st.number_input(
                "Closing Price ($)", 
                min_value=0.0, 
                value=st.session_state['close'], 
                step=0.01,
                key='close',
                help="Today's closing stock price"
            )
        
        with input_col2:
            volume = st.number_input(
                "Volume", 
                min_value=0, 
                value=st.session_state['volume'], 
                step=1000000,
                key='volume',
                help="Number of shares traded today"
            )
            ma_5 = st.number_input(
                "5-Day Moving Average ($)", 
                min_value=0.0, 
                value=round(st.session_state['ma_5'], 2), 
                step=0.01,
                key='ma_5',
                help="Average closing price of last 5 days (suggested, editable)"
            )
            ma_10 = st.number_input(
                "10-Day Moving Average ($)", 
                min_value=0.0, 
                value=round(st.session_state['ma_10'], 2), 
                step=0.01,
                key='ma_10',
                help="Average closing price of last 10 days (suggested, editable)"
            )
            volatility = st.number_input(
                "Volatility ($)", 
                min_value=0.0, 
                value=round(st.session_state['volatility'], 4), 
                step=0.0001,
                key='volatility',
                help="Standard deviation of last 5 days' closing prices (suggested, editable)"
            )
        
        input_col3, input_col4 = st.columns(2)
        with input_col3:
            volume_ma = st.number_input(
                "3-Month Volume Moving Average", 
                min_value=0, 
                value=st.session_state['volume_ma'],
                step=100000,
                key='volume_ma',
                help="Average trading volume of last 3 months (suggested, editable)"
            )
        with input_col4:
            implied_change = st.number_input(
                "Implied Change (%)", 
                min_value=-100.0,
                max_value=100.0,
                value=st.session_state['implied_change'],
                step=0.1,
                key='implied_change',
                help="Expected percentage change based on market sentiment"
            )
        
        # Submit button MUST be inside the form
        submit_button = st.form_submit_button("Predict Tomorrow's Price", use_container_width=True)

with col2:
    st.subheader("Model Info")
    st.success("""
**Model Selection:** Multi-Model Comparison

**Models Trained:**
- Linear Regression
- Random Forest
- XGBoost

**Best Model:** Automatically Selected

**Features Used:**
- Open, High, Low, Close
- Volume & Volume MA
- 5-Day & 10-Day MA
- Volatility

**Performance:** R Square Score > 0.96
    """)

# Prediction section - OUTSIDE the form
if submit_button:
    if model is None or scaler is None:
        st.error("Model not loaded. Please check if model files exist.")
    else:
        # Validate inputs
        if high_price < max(open_price, close_price, low_price):
            st.error("High price should be the maximum value!")
        elif low_price > min(open_price, close_price, high_price):
            st.error("Low price should be the minimum value!")
        else:
            # Prepare input data
            input_data = np.array([[
                open_price, high_price, low_price, close_price, 
                volume, ma_5, ma_10, volatility, volume_ma
            ]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            base_prediction = model.predict(input_scaled)[0]
            
            # Adjust prediction with implied change
            prediction = base_prediction * (1 + implied_change / 100)
            
            # Calculate change
            price_change = prediction - close_price
            percent_change = (price_change / close_price) * 100
            
            # Display prediction with styling
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Create three columns for metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="Today's Closing Price",
                    value=f"${close_price:.2f}",
                    delta=None
                )
            
            with metric_col2:
                st.metric(
                    label="Predicted Tomorrow's Price",
                    value=f"${prediction:.2f}",
                    delta=f"${price_change:.2f} ({percent_change:+.2f}%)",
                    delta_color="normal"
                )
            
            with metric_col3:
                trend = "Upward" if price_change > 0 else "Downward" if price_change < 0 else "Stable"
                st.metric(
                    label="Price Trend",
                    value=trend,
                    delta=None
                )
            
            # Visualization
            st.subheader("Price Comparison")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Today\'s Close', 'Tomorrow\'s Prediction'],
                y=[close_price, prediction],
                text=[f'${close_price:.2f}', f'${prediction:.2f}'],
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e']
            ))
            
            fig.update_layout(
                title='Price Comparison',
                yaxis_title='Price ($)',
                showlegend=False,
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            st.markdown("---")
            st.subheader("Insights")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.info(f"""
                **Price Range Today:**
                - High: ${high_price:.2f}
                - Low: ${low_price:.2f}
                - Range: ${high_price - low_price:.2f}
                """)
            
            with insight_col2:
                confidence_level = "High" if abs(percent_change) < 2 else "Moderate" if abs(percent_change) < 5 else "Low"
                st.warning(f"""
                **Prediction Analysis:**
                - Expected Change: {percent_change:+.2f}%
                - Implied Change: {implied_change:+.2f}%
                - Confidence: {confidence_level}
                - Volume: {volume:,} shares
                """)