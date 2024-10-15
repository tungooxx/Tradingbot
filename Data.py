import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from datetime import datetime
import yfinance as yf
import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as pl
import requests

st.set_page_config(page_title="Data", page_icon="🌌️", layout="wide", initial_sidebar_state="expanded")
st.title("Data 👁️‍🗨️️")

api_key = ""


class TradingBotApp:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, openai_api_key=api_key)
        self.tools = load_tools(["ddg-search"])
        self.agent = initialize_agent(self.tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

    # async def get_stock_symbols(self, company_name):
    #     st_callback = StreamlitCallbackHandler(st.container())
    #     search_results = await asyncio.to_thread(
    #         self.agent.run,
    #         f"{company_name} stock symbol(s). Return only the symbols separated by spaces. Don't add any type of punctuation",
    #         callbacks=[st_callback]
    #     )
    #     symbols = search_results.split()
    #     return symbols


    async def get_gnews_api(self):
        url = f"https://gnews.io/api/v4/top-headlines?lang=en&token={self.api_key}"
        response = requests.get(url)
        news = response.json()
        return news

    async def get_gnews_api_spec(self, search_term):
        url = f"https://gnews.io/api/v4/search?q={search_term}&token=8488063c1346f99630178978b0360f2e"
        response = requests.get(url)
        news = response.json()
        return news

    async def get_stock_history(self,symbol,date):
        ticker = yf.Ticker(symbol)
        data = ticker.history(start="2020-1-1", end=date)
        return data


    

    async def run(self):
        date_now = datetime.now()
        date_year = date_now.year
        date_month = date_now.month
        date_day = date_now.day
        day_of_week = date_now.strftime("%A")

        date_d = "{}-{}-{}".format(date_year,date_month,date_day)

        st.title(":blue[Welcome!]")
        st.header("_Trading Bot App_")
        st.subheader(f" :green[_{day_of_week}_]")
        st.subheader(f"{day_of_week}")

        company_name = st.text_input("Enter a company name:")

        if company_name:
            # symbols = await self.get_stock_symbols(company_name)
            # gnews_api_spec = await self.get_gnews_api_spec(symbol)
            gnews_api_spec = await self.get_gnews_api_spec("TSLA")
            symbols = ["TSLA"]
            # st.write(f"Stock symbol(s) for {company_name}: {', '.join(symbols)}")
            for symbol in symbols:
                
                # left_column,right_column = st.columns(2)

                with st.spinner("Searching...."):
                    with st.sidebar:
                        with st.expander(symbol):
                            st.subheader("## News from GNews API", divider = 'rainbow')                  
                            for article in gnews_api_spec['articles']:
                                st.write(f"**Title:** {article['title']}")
                                st.write(f"**Description:** {article['description']}")
                                st.write(f"**URL:** {article['url']}")
                                st.markdown("""---""")
                
                plot_placeholder = st.empty()

                st.markdown("""---""")

                df = await self.get_stock_history(symbol,date_d)
                
                data_date = df.index.to_numpy()

                data_open_price = df['Open'].to_numpy()
                data_high_price = df['High'].to_numpy()
                data_low_price = df['Low'].to_numpy()
                data_close_price = df['Close'].to_numpy()
                feg = pl.Figure(data = [pl.Candlestick(x = data_date,open=data_open_price,high=data_high_price,low=data_low_price,close = data_close_price)])
                feg.update_layout(title_text = "Candle Graph of {symbol}")
                feg.update_layout(xaxis_rangeslider_visible = False)
                plot_placeholder.plotly_chart(feg,use_container_width = True)
if __name__ == "__main__":
    app = TradingBotApp()
    asyncio.run(app.run())