import streamlit as st
from datetime import datetime
import pandas as pd
from fetch_metal_prices import fetch_metal_prices
from history_scrapper import scrap_history

st.title("Precious Metal Price Dashboard")

st.sidebar.header("Select Metal and Time Range")

metal = st.sidebar.selectbox("Choose Metal", ("gold", "silver"))

option = st.sidebar.radio("View", ["Live Prices", "Historical Data"])


def download_data(df: pd.DataFrame, filename: str) -> None:
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Data",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def display_live_prices(metal: str) -> None:
    df = fetch_metal_prices(metal=metal)
    if df is not None:
        st.subheader(f"Live {metal.capitalize()} Prices")
        st.write(df)

        download_data(df, f"{metal}_live_prices.csv")
    else:
        st.error("Failed to fetch live prices.")


def display_historical_prices(ticker: str) -> None:
    data = scrap_history(ticker, start=start_date, end=end_date)
    if not data.empty:
        st.subheader(f"Historical {metal.capitalize()} Prices")
        st.line_chart(data["Close"])
        st.dataframe(data, use_container_width=True)

        download_data(data, f"{metal}_historical_prices.csv")
    else:
        st.error("No historical data available for this time range.")


if option == "Live Prices":
    display_live_prices(metal)
else:
    start_date = st.sidebar.date_input("Start Date", datetime(2000, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    ticker = "GC=F" if metal == "gold" else "SI=F"
    display_historical_prices(ticker)
