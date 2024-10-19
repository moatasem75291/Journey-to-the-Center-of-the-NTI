import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Gold and silver historical URLs
gold_historical = "https://services.bullionstar.com/spot-chart/getChart?product=false&productId=0&productTo=false&productIdTo=0&fromIndex=XAU&toIndex=USD&period=MAX&width=600&height=300&timeZoneId=Africa%2FCairo&weightUnit=g"
silver_historical = "https://services.bullionstar.com/spot-chart/getChart?product=false&productId=0&productTo=false&productIdTo=0&fromIndex=XAG&toIndex=USD&period=MAX&width=600&height=300&timeZoneId=Africa%2FCairo&weightUnit=g"

# USD to EGP GraphQL API
usd_to_egp_url = "https://webql-redesign.cnbcfm.com/graphql?operationName=getQuoteChartData&variables=%7B%22symbol%22%3A%22EGP%3D%22%2C%22timeRange%22%3A%22ALL%22%7D&extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%2261b6376df0a948ce77f977c69531a4a8ed6788c5ebcdd5edd29dd878ce879c8d%22%7D%7D"


def fetch_data(url: str) -> dict:
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch data. Status code: {response.status_code}")
        return None


def process_usd_egp_data(data: dict) -> pd.DataFrame:
    bars = data["data"]["chartData"]["priceBars"]
    dates = []
    prices = []

    for bar in bars:
        trade_time = bar.get("tradeTimeinMills")
        close_price = bar.get("close")

        if trade_time is not None and close_price != "-9999401.0000":
            try:
                dates.append(int(trade_time))
                prices.append(float(close_price))
            except (ValueError, TypeError):
                st.warning(
                    f"Invalid data encountered for trade time: {trade_time} or close price: {close_price}"
                )

    if not dates or not prices:
        st.error("No valid data available for the selected time range.")
        return pd.DataFrame()

    df = pd.DataFrame({"date": dates, "price": prices})
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    return df


def process_metal_data(data: dict) -> pd.DataFrame:
    dates = []
    prices = []

    for entry in data["dataSeries"]:
        startDate = 946940400000
        startDate_seconds = startDate / 1000
        date = int(str(entry["d"]) + "00")
        timestamp = startDate_seconds + date
        date = datetime.utcfromtimestamp(timestamp)
        dates.append(date)

        price = entry["v"]
        prices.append(price)

    df = pd.DataFrame({"date": dates, "price": prices})
    return df


def plot_descriptive(df: pd.DataFrame, currency: str) -> None:
    st.subheader(f"Descriptive Charts ({currency})")
    st.write(df.describe())

    plt.figure(figsize=(10, 6))
    plt.plot(df["date"], df["price"], label=f"Price ({currency})")
    plt.title(f"Price Over Time ({currency})")
    plt.xlabel("Date")
    plt.ylabel(f"Price ({currency})")
    plt.legend()
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.histplot(df["price"], bins=30, kde=True)
    plt.title(f"Price Distribution ({currency})")
    plt.xlabel(f"Price ({currency})")
    st.pyplot(plt)


def plot_predictive(df: pd.DataFrame, currency: str) -> None:
    st.subheader(f"Predictive Chart (Simple Moving Average) in {currency}")

    df["SMA_30"] = df["price"].rolling(window=30).mean()
    df["SMA_90"] = df["price"].rolling(window=90).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df["date"], df["price"], label=f"Price ({currency})", alpha=0.5)
    plt.plot(df["date"], df["SMA_30"], label="SMA 30", color="green")
    plt.plot(df["date"], df["SMA_90"], label="SMA 90", color="red")
    plt.title(f"Price with Simple Moving Averages ({currency})")
    plt.xlabel("Date")
    plt.ylabel(f"Price ({currency})")
    plt.legend()
    st.pyplot(plt)


def plot_continuous_discrete(df: pd.DataFrame, currency: str) -> None:
    st.subheader(f"Continuous Charts in {currency}")

    df["price_change"] = df["price"].pct_change()

    plt.figure(figsize=(10, 6))
    plt.plot(
        df["date"],
        df["price_change"],
        label=f"Price Change ({currency})",
        color="purple",
    )
    plt.title(f"Price Change Over Time ({currency})")
    plt.xlabel("Date")
    plt.ylabel("Price Change (%)")
    st.pyplot(plt)


def download_data(df: pd.DataFrame, filename: str) -> None:
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Data",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def main() -> None:
    st.title("Gold, Silver, and USD to EGP Price Analysis")

    data_choice = st.selectbox("Select Data:", ("Gold", "Silver", "USD to EGP"))

    if data_choice == "Gold":
        url = gold_historical
        currency = "USD"
        data = fetch_data(url)
        df = process_metal_data(data) if data else None
    elif data_choice == "Silver":
        url = silver_historical
        currency = "USD"
        data = fetch_data(url)
        df = process_metal_data(data) if data else None
    else:
        url = usd_to_egp_url
        currency = "EGP"
        data = fetch_data(url)
        df = process_usd_egp_data(data) if data else None

    if df is not None:
        st.subheader(f"{data_choice} Price Data in {currency}")
        start_date = st.date_input("Start date", value=df["date"].min().date())
        end_date = st.date_input("End date", value=df["date"].max().date())

        mask = (df["date"] >= pd.to_datetime(start_date)) & (
            df["date"] <= pd.to_datetime(end_date)
        )
        df_filtered = df[mask]

        st.write(df_filtered.head())

        plot_descriptive(df_filtered, currency)
        plot_predictive(df_filtered, currency)
        plot_continuous_discrete(df_filtered, currency)
        if currency == "EGP":
            download_data(df_filtered, f"{currency}_VS_USD.csv")
        else:
            download_data(df_filtered, f"{data_choice}_{currency}.csv")


if __name__ == "__main__":
    main()
