import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from datetime import datetime
import asyncio

st.set_page_config(page_title="Data", page_icon="🌌️", layout="wide", initial_sidebar_state="expanded")
st.title("Data 👁️‍🗨️️")

api_key = ""


class TradingBotApp:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, openai_api_key=api_key)
        self.tools = load_tools(["ddg-search"])
        self.agent = initialize_agent(self.tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

    async def get_stock_symbols(self, company_name):
        st_callback = StreamlitCallbackHandler(st.container())
        search_results = await asyncio.to_thread(
            self.agent.run,
            f"{company_name} stock symbol(s). Return only the symbols separated by spaces. Don't add any type of punctuation",
            callbacks=[st_callback]
        )
        symbols = search_results.split()
        return symbols

    async def run(self):
        date_now = datetime.now()
        date_formatted = date_now.strftime("%Y-%m-%d")
        day_of_week = date_now.strftime("%A")

        st.title(":blue[Welcome!]")
        st.header("_Trading Bot App_")
        st.subheader(f" :green[_{date_formatted}_]")
        st.subheader(f"{day_of_week}")

        company_name = st.text_input("Enter a company name:")

        if company_name:
            symbols = await self.get_stock_symbols(company_name)
            if symbols:
                st.write(f"Stock symbol(s) for {company_name}: {', '.join(symbols)}")
            else:
                st.write("No stock symbols found.")

if __name__ == "__main__":
    app = TradingBotApp()
    asyncio.run(app.run())