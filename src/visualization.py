import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def display_forecast_table(dates, predictions):
    """
    Displays the forecast table and returns the DataFrame.
    """
    df_forecast = pd.DataFrame({"Date": dates, "Predicted Close Price": predictions})
    st.dataframe(df_forecast, use_container_width=True)
    return df_forecast

def display_forecast_plot(dates, predictions, ticker, forecast_days):
    """
    Displays the forecast plot.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(dates, predictions, marker='o', linestyle='-', color='b', label="Predicted Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price ($)")
    plt.title(f"{ticker} - {forecast_days}-Day Predicted Close Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    st.pyplot(plt)
