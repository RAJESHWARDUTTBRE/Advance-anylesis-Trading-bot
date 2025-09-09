import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import io
from PIL import Image
import telegram.ext  # This is the key change
import asyncio
import httpx
import threading
import time

# Your Telegram Bot credentials
# NOTE: For production, it's highly recommended to use Streamlit's st.secrets for security.
TELEGRAM_BOT_TOKEN = "8382011014:AAHnGQifO3-XbdC40zJcPjEizqFLIDxp6bY"
TELEGRAM_CHAT_ID = "8398305736"

# Initialize Telegram bot using the correct modern method
try:
    application = telegram.ext.ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    bot = application.bot
except Exception as e:
    st.error(f"Telegram Bot initialization failed: {e}")
    bot = None

def _send_telegram_signal_thread(message):
    """Function to run in a separate thread to send the async Telegram message."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def send_message_async():
            # Check if bot is initialized before attempting to send a message
            if bot:
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        
        loop.run_until_complete(send_message_async())
    except Exception as e:
        st.error(f"Failed to send Telegram message from thread: {e}")
    finally:
        loop.close()

def send_telegram_signal(message):
    """Sends a message to Telegram in a non-blocking way."""
    if bot:
        thread = threading.Thread(target=_send_telegram_signal_thread, args=(message,))
        thread.start()

# Function to load the file (CSV or Excel) and find the correct columns
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, dtype=str)
        elif file.name.endswith('.xlsx') or file.name.endswith('.xlsm'):
            df = pd.read_excel(file, dtype=str)
        else:
            st.error("Invalid file format. Please upload a .csv or .xlsx file.")
            return None, None, None

        df.columns = df.columns.astype(str).str.strip().str.lower()
        df = df.loc[:, ~df.columns.str.contains('^unnamed')]
        ce_col = next((col for col in df.columns if 'option symbol ce' in col), None)
        pe_col = next((col for col in df.columns if 'option symbol pe' in col), None)

        if ce_col and pe_col:
            return df, ce_col, pe_col
        else:
            st.error("File mein 'option symbol ce' ya 'option symbol pe' columns nahi mile. Kripya file check karein.")
            return None, None, None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None, None

# Function to extract the base stock ticker from the option symbol
def get_base_ticker(option_symbol):
    match = re.search(r'([A-Z0-9-]+)25SEP', option_symbol)
    if match:
        base_name = match.group(1)
        return f"{base_name}.NS"
    return None

# Function to perform analysis and provide a trade signal
def analyze_trade(df):
    if df.empty or 'Close' not in df.columns or len(df) < 14: # RSI needs 14 periods
        return "Not enough data for analysis.", None

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['SMA_3'] = df['Close'].rolling(window=3).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    latest_rsi = df['RSI_14'].iloc[-1]

    if df['SMA_3'].iloc[-2] < df['EMA_9'].iloc[-2] and df['SMA_3'].iloc[-1] > df['EMA_9'].iloc[-1]:
        return "CE", latest_rsi
    elif df['SMA_3'].iloc[-2] > df['EMA_9'].iloc[-2] and df['SMA_3'].iloc[-1] < df['EMA_9'].iloc[-1]:
        return "PE", latest_rsi
    else:
        return "Neutral", latest_rsi

# Function to create a combined chart for price, volume, and indicators
def create_full_chart(ticker_symbol, latest_price, trade_rec):
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="3mo", interval="1d")

        if data.empty:
            return None

        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data['SMA_3'] = data['Close'].rolling(window=3).mean()
        data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        data['RSI_14'] = 100 - (100 / (1 + rs))

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=[f"Price Chart for {ticker_symbol.split('.')[0]}", 'Volume', 'RSI'],
                            row_heights=[0.6, 0.2, 0.2])
        
        # Candlestick chart for Price
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                                     name="Price", increasing_line_color='green', decreasing_line_color='red'), row=1, col=1)
        
        # SMA3 line
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_3'], mode='lines', name='SMA3', line=dict(color='blue', width=2)), row=1, col=1)
        
        # EMA9 line
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_9'], mode='lines', name='EMA9', line=dict(color='orange', width=2)), row=1, col=1)

        # Volume bar chart
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color='purple'), row=2, col=1)

        # RSI bar chart
        colors_rsi = ['red' if val > 70 or val < 30 else 'grey' for val in data['RSI_14']]
        fig.add_trace(go.Bar(x=data.index, y=data['RSI_14'], name="RSI", marker_color=colors_rsi), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Update layout with larger fonts
        fig.update_layout(
            height=900,
            title_text=f"Technical Analysis for {ticker_symbol.split('.')[0]}",
            xaxis_rangeslider_visible=False,
            font=dict(size=14)
        )

        fig.update_yaxes(title_text="Price", row=1, col=1, title_font=dict(size=16))
        fig.update_yaxes(title_text="Volume", row=2, col=1, title_font=dict(size=16))
        fig.update_yaxes(title_text="RSI", row=3, col=1, title_font=dict(size=16))
        
        # Add analysis text to the chart with larger font
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=0.95,
            text=f"**Current Price:** ₹{latest_price:.2f}<br>**Recommendation:** {trade_rec}",
            showarrow=False,
            font=dict(size=16, color="black"),
            align="left"
        )
        
        return fig
    except Exception as e:
        return None

# --- Main Logic and UI ---
st.set_page_config(layout="wide")

# Send "Start" message when the app begins, only once.
if 'app_started' not in st.session_state:
    st.session_state.app_started = True
    send_telegram_signal("🚀 *Streamlit app has started.*")

st.title("Comprehensive Stock Analysis Tool")
st.markdown("Yeh app aapki file se stocks ka analysis karta hai aur trade signal deta hai, jise aap Excel aur PDF mein download bhi kar sakte hain.")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx', 'xlsm'])

if uploaded_file is not None:
    df, ce_col, pe_col = load_data(uploaded_file)
    if df is not None:
        all_tickers = sorted(list(set([get_base_ticker(s) for s in df[ce_col].dropna().unique()])))
        all_stocks = [t.split('.')[0] for t in all_tickers if t]
        
        st.sidebar.header("Select Stocks")
        select_all = st.sidebar.checkbox("Select all stocks", value=True)
        
        if select_all:
            selected_stocks = all_stocks
        else:
            selected_stocks = st.sidebar.multiselect('Or select specific stocks', all_stocks)
        
        if not selected_stocks:
            st.warning("Please select at least one stock to analyze.")
        else:
            st.subheader("Analysis in Progress...")
            
            analysis_df = pd.DataFrame(columns=['Stock', 'Trade Recommendation'])
            chart_images = []
            
            for stock in selected_stocks:
                base_ticker = f"{stock}.NS"
                
                try:
                    data = yf.Ticker(base_ticker).history(period="1mo", interval="1d")
                    
                    if not data.empty:
                        trade_rec, latest_rsi = analyze_trade(data)
                        latest_price = data['Close'].iloc[-1]
                        
                        new_row = pd.DataFrame([{'Stock': stock, 'Trade Recommendation': trade_rec}])
                        analysis_df = pd.concat([analysis_df, new_row], ignore_index=True)

                        # Send signal to Telegram if it's a CE or PE recommendation
                        if trade_rec in ["CE", "PE"]:
                            # Calculate SL and Target prices
                            sl_price = 0
                            target_price = 0
                            if trade_rec == "CE":
                                sl_price = latest_price * (1 - 0.0015)  # -0.15% for CE
                                target_price = latest_price * (1 + 0.0045) # +0.45% for CE
                            elif trade_rec == "PE":
                                sl_price = latest_price * (1 + 0.0015)  # +0.15% for PE
                                target_price = latest_price * (1 - 0.0045) # -0.45% for PE
                                
                            message = (
                                f"🚨 **New Trade Signal** 🚨\n\n"
                                f"**Stock:** {stock}\n"
                                f"**Recommendation:** {trade_rec}\n"
                                f"**Entry Price:** ₹{latest_price:.2f}\n"
                                f"**Stop Loss:** ₹{sl_price:.2f}\n"
                                f"**Target:** ₹{target_price:.2f}\n"
                                f"**RSI:** {latest_rsi:.2f}"
                            )
                            send_telegram_signal(message)
                        
                        chart_fig = create_full_chart(base_ticker, latest_price, trade_rec)
                        if chart_fig:
                            # Display the chart in the app
                            st.plotly_chart(chart_fig, use_container_width=True)
                            
                            # Convert to image for PDF
                            img_buffer = io.BytesIO()
                            chart_fig.write_image(img_buffer, format='png')
                            img_buffer.seek(0)
                            chart_images.append(Image.open(img_buffer))
                    else:
                        st.warning(f"Warning: No data found for {stock}. Skipping.")
                except Exception as e:
                    st.error(f"Error fetching data for {stock}: {e}")
                    continue

            # Create PDF from images
            pdf_buffer = io.BytesIO()
            if chart_images:
                first_image = chart_images[0]
                other_images = chart_images[1:]
                first_image.save(
                    pdf_buffer,
                    "PDF",
                    resolution=100.0,
                    save_all=True,
                    append_images=other_images
                )
            pdf_buffer.seek(0)
            
            st.subheader("Analysis Results")
            st.table(analysis_df)

            col1, col2 = st.columns(2)
            with col1:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    analysis_df.to_excel(writer, index=False, sheet_name='Trade_Recommendations')
                excel_buffer.seek(0)
                
                st.download_button(
                    label="Download Analysis Excel",
                    data=excel_buffer,
                    file_name='stock_analysis.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            
            with col2:
                if chart_images:
                    st.download_button(
                        label="Download All Charts (PDF)",
                        data=pdf_buffer,
                        file_name="all_stock_charts.pdf",
                        mime="application/pdf"
                    )