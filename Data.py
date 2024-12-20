import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import ChatMessage
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from yahooquery import Ticker
import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as pl
import requests
from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain

st.set_page_config(page_title="Data", page_icon="🌌️", layout="wide", initial_sidebar_state="expanded")
st.title("Data 👁️‍🗨️️")

api_key = ""


class StreamHandler(BaseCallbackHandler):
	def __init__(self, container, initial_text = ""):
		self.container = container
		self.text = initial_text

	def on_llm_new_token(self, token: str, **kwargs) -> None:
		self.text += token
		self.container.markdown(self.text)

class TradingBotApp:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, openai_api_key=api_key)
        self.tools = load_tools(["ddg-search"])
        self.query = "Please, give me an exahustive fundamental analysis about the companies that you find in the documented knowledge. I want to know the pros and cons of a large-term investment. Please, base your answer on what you know about the company, but also on wht you find useful about the documented knowledge. I want you to also give me your opinion in, if it is worthy to invest on that company given the fundamental analysis you make. If you conclude that is actually wise to invest on a given company, or in multiple companies (focus only on the ones in the documented knowledge) then come up also with some strategies that I could follow to make the best out of my investments."
        self.agent = initialize_agent(self.tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
        self.DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The
		AI is an AI-powered fundamental analist. It uses Documented Information to give fundamental insights on an assest detemrined by the user. It is specific, and gives full relevant
		perspective to let the user know, if it is worthy to make investment on a certain asset, and can also build intelligent strategies with the given information, as well as from intel that it
		already knows, or will generate. Take into account, that the documented knowledge comes in the next scrutcture. **Title:** (Message of the Title), **Description:** (Message of the
		Descricption)\n\n, and so on. All articles from the documented knowledge have a title and a description (both of which are separated by comas), and all articles are separated with the \n\n
		command between one another.

		Documented Information:
		{docu_knowledge},

		(You do not need to use these pieces of information if not relevant)

		Current conversation:
		Human: {input}
		AI-bot:"""
        if 'symbol' not in st.session_state:
            st.session_state.symbol = {}
        if 'fetch_data' not in st.session_state:
            st.session_state.fetch_data = {}
    # ChatGPT
    async def get_stock_symbols(self, company_name):
        st_callback = StreamlitCallbackHandler(st.container())
        search_results = await asyncio.to_thread(
            self.agent.run,
            f"{company_name} stock symbol(s). Return only the symbols separated by spaces. Don't add any type of punctuation",
            callbacks=[st_callback]
        )
        symbols = search_results.split()
        return symbols

    # ollama
    # async def get_stock_symbols(self, company_name):
    #     model = OllamaLLM(model="llama3.1")
    #
    #     ollama_prompt = f"{company_name} stock symbol(s). Return only the symbols separated by spaces. Don't add any type of punctuation."
    #
    #     prompt_template = ChatPromptTemplate.from_template(ollama_prompt)
    #
    #     llm_chain = LLMChain(llm=model, prompt=prompt_template)
    #
    #     try:
    #         # Call the LLM chain to get the response
    #         response = await llm_chain.arun(company_name=company_name)
    #
    #         # Process the response to extract stock symbols
    #         symbols = response.strip().split()
    #         return symbols
    #
    #     except Exception as e:
    #         print(f"Error fetching stock symbols using Ollama: {e}")
    #         return []

    async def get_fin_statements(self, symbol):
        df = Ticker(symbol)
        df1 = df.income_statement().reset_index(drop=True).transpose()
        df2 = df.balance_sheet().reset_index(drop=True).transpose()
        df3 = df.cash_flow().reset_index(drop=True).transpose()
        return df1, df2, df3

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

    def get_response(self, user_message, docu_knowledge, financial_statement, temperature = 0):
        prompt = PromptTemplate(input_variables = ['input','docu_knowledge',"financial_statement"],template = self.DEFAULT_TEMPLATE)

        stream_handler = StreamHandler(st.empty())

        chat = ChatOpenAI(streaming = True,callbacks = [stream_handler], temperature = 0, openai_api_key=api_key)

        conversation = LLMChain(llm = chat, prompt = prompt, verbose = True)

        output = conversation.predict(input = user_message, docu_knowledge = docu_knowledge, financial_statement = financial_statement)

        return output
    async def get_stock_history(self,symbol,date):
        ticker = yf.Ticker(symbol)
        data = ticker.history(start="2020-1-1", end=date)
        return data

    async def fetch_data(self, df, symbol):
        data_date = df.index.to_numpy().reshape(1, -1)
        data_open_price = df['Open'].to_numpy().reshape(1, -1)
        data_high_price = df['High'].to_numpy().reshape(1, -1)
        data_low_price = df['Low'].to_numpy().reshape(1, -1)
        data_close_price = df['Close'].to_numpy().reshape(1, -1)
        df_data = np.concatenate((data_date, data_open_price, data_high_price, data_low_price, data_close_price), axis = 0)
        return df_data

    async def normalized_data(self,df):
        data_open_price = df[1].reshape(-1, 1)
        data_high_price = df[2].reshape(-1, 1)
        data_low_price = df[3].reshape(-1, 1)
        data_close_price = df[4].reshape(-1, 1)
        df_data = np.concatenate((data_open_price, data_high_price, data_low_price, data_close_price), axis = 0)

        scaler = MinMaxScaler(feature_range=(0, 1))
        df_data = scaler.fit_transform(df_data).ravel()

        shape = data_open_price.shape[0]

        data_open_price_norm = df_data[:shape]
        data_high_price_norm = df_data[shape:shape*2]
        data_low_price_norm = df_data[shape*2:shape*3]
        data_close_price_norm = df_data[shape*3:shape*4]

        norm_data = np.concatenate((data_open_price_norm, data_high_price_norm,data_low_price_norm,data_close_price_norm), axis = 0)
        return norm_data

    async def prepare_data_x(self, x, window_size):
        n_row = x.shape[0] - window_size + 1
        output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
        return output[:-1], output[-1]


    async def prepare_data_y(self, x, window_size):
        output = x[window_size:]
        return output

    async def data_on_percent(self, datas, percent=0.9):
        data = datas[0]
        data_x, data_unseen_x = await self.prepare_data_x(data, window_size = 20)
        data_y = await self.prepare_data_y(data, window_size=20)

    async def get_latest_price(self, data):
        latest_price = data['Close'].iloc[-1]
        return latest_price


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

        if "messages" not in st.session_state:
            st.session_state["messages"] = [ChatMessage(role = "assistant", content = "")]

        if company_name:
            # if company_name in st.session_state.symbol:
            #     symbols = st.session_state.symbol[company_name]
            # else:
            #     symbols = await self.get_stock_symbols(company_name)
            #     st.session_state.symbol[company_name] = symbols
            articles_string = ""
            financial_statement = ""
            gnews_api_spec = await self.get_gnews_api_spec("TSLA")
            symbols = ["TSLA"]
            st.write(f"Stock symbol(s) for {company_name}: {', '.join(symbols)}")
            for symbol in symbols:

                with st.spinner("Searching...."):
                    # gnews_api_spec = await self.get_gnews_api_spec(symbol)
                    # left_column,right_column = st.columns(2)
                    try:
                        income_statement, balance_sheet, cash_flow = await self.get_fin_statements(symbol)

                        financial_statement += (f"{symbol}") + "\n\n"
                        df_price = await self.get_stock_history(symbol, date_d)
                        stock_price = await self.get_latest_price(df_price)
                        financial_statement += "Current Market Price:" + "\n\n"
                        financial_statement += (f"{stock_price}") + "\n\n"
                        financial_statement += "Income Statement:" + "\n\n"
                        financial_statement += income_statement.to_string() + "\n\n"
                        financial_statement += "Balance Sheet:" + "\n\n"
                        financial_statement += balance_sheet.to_string() + "\n\n"
                        financial_statement += "Cash Flow:" + "\n\n"
                        financial_statement += cash_flow.to_string() + "\n\n"

                        st.write(income_statement, balance_sheet, cash_flow)
                    except Exception as e:
                        st.subheader(f":red[Financial Statement for _{symbol}_ couldn't be found 😥️]")
                    with st.sidebar:
                        with st.expander(symbol):
                            st.subheader("## News from GNews API", divider = 'rainbow')                  
                            for article in gnews_api_spec['articles']:
                                st.write(f"**Title:** {article['title']}")
                                st.write(f"**Description:** {article['description']}")
                                st.write(f"**URL:** {article['url']}")
                                st.markdown("""---""")
                                article_string = f"**Title:** {article['title']}, **Description:** {article['description']} \n"
                                articles_string += article_string + "\n"

                
                plot_placeholder = st.empty()

                st.markdown("""---""")

                if symbol in st.session_state.fetch_data:
                    stock_data = await self.fetch_data(st.session_state.fetch_data[symbol], symbol)
                else:
                    df = await self.get_stock_history(symbol, date_d)
                    stock_data = await self.fetch_data(df, symbol)
                    st.session_state.fetch_data[symbol] = df

                num_data_points = len(stock_data[0])

                feg = pl.Figure(data = [pl.Candlestick(x = stock_data[0], open = stock_data[1], high = stock_data[2], low = stock_data[3], close = stock_data[4])])
                feg.update_layout(title_text = "Full Data")
                feg.update_layout(xaxis_rangeslider_visible = False)
                plot_placeholder.plotly_chart(feg, use_container_width = True)

                norm_data = self.normalized_data(stock_data)

                with st.chat_message("assistant"):
                    user_input = self.query
                    output = self.get_response(user_input, articles_string, financial_statement, temperature = 0)
                    st.session_state.messages.append(ChatMessage(role = "assistant", content = output))
if __name__ == "__main__":
    app = TradingBotApp()
    asyncio.run(app.run())