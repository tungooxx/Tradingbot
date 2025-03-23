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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from Architecture import *
st.set_page_config(page_title="Trading App", page_icon="🌌️", layout="wide", initial_sidebar_state="expanded")

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
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True, openai_api_key=api_key)
        self.tools = load_tools(["ddg-search"])
        self.window_size = 20
        self.train_perc = 0.9
        self.batch_size = 64
        self.device = "cuda"
        self.lr = 0.01
        self.scheduler = 40
        self.epochs = 150
        self.prompt_template = """ stock symbol(s). Return only the symbols separated by spaces. Don't add any type of punctuantion. If you already know the answer, there is no need to search for
        it on duckduck go"""
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
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = {}
    # ChatGPT
    async def get_stock_symbols(self, company_name):
        st_callback = StreamlitCallbackHandler(st.container())
        search_results = self.agent.run(company_name + self.prompt_template, callbacks=[st_callback])
        symbols = search_results.split(" ")
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

    async def normalized_data(self, asset_data):

        data_open_price = asset_data[1].reshape(-1, 1)
        data_high_price = asset_data[2].reshape(-1, 1)
        data_low_price = asset_data[3].reshape(-1, 1)
        data_close_price = asset_data[4].reshape(-1, 1)

        datas = np.concatenate((data_open_price, data_high_price, data_low_price, data_close_price), axis=0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(np.array(datas))
        normalized_data = normalized_data.T[0]

        int_size = data_open_price.shape[0]

        data_open_price_norm = normalized_data[:int_size]
        data_high_price_norm = normalized_data[int_size:(int_size * 2)]
        data_low_price_norm = normalized_data[(int_size * 2):(int_size * 3)]
        data_close_price_norm = normalized_data[(int_size * 3):(int_size * 4)]

        data_open_price_norm = data_open_price_norm.reshape(1, -1)
        data_high_price_norm = data_high_price_norm.reshape(1, -1)
        data_low_price_norm = data_low_price_norm.reshape(1, -1)
        data_close_price_norm = data_close_price_norm.reshape(1, -1)

        norm_data = np.concatenate(
            (data_open_price_norm, data_high_price_norm, data_low_price_norm, data_close_price_norm), axis=0)

        return norm_data, scaler

    async def prepare_data_x(self, x, window_size):
        n_row = x.shape[0] - window_size + 1
        output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
        return output[:-1], output[-1]


    async def prepare_data_y(self, x, window_size):
        output = x[window_size:]
        return output


    async def data_on_percent(self, datas, percent):
        data = datas[0]
        data_x, data_x_unseen = await self.prepare_data_x(data, window_size=self.window_size)
        data_y = await self.prepare_data_y(data, window_size=self.window_size)

        split_index = int(data_y.shape[0] * percent)
        data_x_train = data_x[:split_index]
        data_x_val = data_x[split_index:]
        data_y_train = data_y[:split_index]
        data_y_val = data_y[split_index:]

        dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
        dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

        dataset_train.y = dataset_train.y.reshape(-1, 1)
        dataset_val.y = dataset_val.y.reshape(-1, 1)

        data_x_unseen = data_x_unseen.reshape(1, -1)
        datas = np.delete(datas, 0, axis=0)

        for data in datas:
            data_x_, data_x_unseen_ = await self.prepare_data_x(data, window_size=self.window_size)
            data_y_ = await self.prepare_data_y(data, window_size=self.window_size)

            data_x_train_ = data_x_[:split_index]
            data_x_val_ = data_x_[split_index:]
            data_y_train_ = data_y_[:split_index]
            data_y_val_ = data_y_[split_index:]

            dataset_train_ = TimeSeriesDataset(data_x_train_, data_y_train_)
            dataset_val_ = TimeSeriesDataset(data_x_val_, data_y_val_)

            dataset_train_.y = dataset_train_.y.reshape(-1, 1)
            dataset_val_.y = dataset_val_.y.reshape(-1, 1)

            data_x_unseen_ = data_x_unseen_.reshape(1, -1)

            dataset_train.x = np.concatenate((dataset_train.x, dataset_train_.x), axis=2)
            dataset_val.x = np.concatenate((dataset_val.x, dataset_val_.x), axis=2)
            dataset_train.y = np.concatenate((dataset_train.y, dataset_train_.y), axis=1)
            dataset_val.y = np.concatenate((dataset_val.y, dataset_val_.y), axis=1)
            data_x_unseen = np.concatenate((data_x_unseen, data_x_unseen_), axis=0)

        return data_x_unseen, dataset_train, dataset_val, split_index

    async def run_epoch(self, dataloader, model, optimizer, criterion, scheduler, is_training=False):
        epoch_loss = 0

        if is_training:
            model.train()
        else:
            model.eval()

        for idx, (x, y) in enumerate(dataloader):
            if is_training:
                optimizer.zero_grad()

            batchsize = x.shape[0]

            x = x.to(self.device)
            y = y.to(self.device)

            out = model(x)
            loss = criterion(out.contiguous(), y.contiguous())

            if is_training:
                loss.backward()
                optimizer.step()

            epoch_loss += (loss.detach().item() / batchsize)

        lr = scheduler.get_last_lr()[0]

        return epoch_loss, lr

    async def train_model(self, model, dataset_train, dataset_val):

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=True)

        model = model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler, gamma=0.1)

        my_bar = st.progress(0, text="Starting")

        for epoch in range(self.epochs):
            loss_train, lr_train = await self.run_epoch(train_dataloader, model, optimizer, criterion, scheduler,
                                                        is_training=True)
            loss_val, lr_val = await self.run_epoch(val_dataloader, model, optimizer, criterion, scheduler)
            scheduler.step()

            my_bar.progress((epoch + 1) / self.epochs,
                            text='Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'.format(epoch + 1,
                                                                                                    self.epochs,
                                                                                                    loss_train,

                                                                                             loss_val, lr_train))

        return model

    async def launch_model(self, model, dataset_train, dataset_val, chunck_of_unseen_data, predi_days=1):

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=False)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False)

        model.eval()

        predicted_train = np.empty((1, dataset_val.y.shape[1]), dtype=float)

        for idx, (x, y) in enumerate(train_dataloader):
            x = x.to(self.device)
            out = model(x)
            out = out.cpu().detach().numpy()
            predicted_train = np.concatenate((predicted_train, out))

        predicted_train = np.delete(predicted_train, 0, axis=0)

        predicted_val = np.empty((1, dataset_val.y.shape[1]), dtype=float)

        for idx, (x, y) in enumerate(val_dataloader):
            x = x.to(self.device)
            out = model(x)
            out = out.cpu().detach().numpy()
            predicted_val = np.concatenate((predicted_val, out))

        predicted_val = np.delete(predicted_val, 0, axis=0)

        Day_to_predict = predi_days
        data_x_unseen = np.array([chunck_of_unseen_data.T])

        predicted_day = np.empty((1, dataset_val.y.shape[1], 1), dtype=float)

        for _ in range(Day_to_predict):
            model.eval()

            x = torch.tensor(data_x_unseen).float().to(self.device)
            prediction = model(x)
            prediction = prediction.cpu().detach().numpy()
            predictions = prediction[0].reshape(1, 1, -1)

            data_x_unseen = np.concatenate((data_x_unseen, predictions), axis=1)
            data_x_unseen = np.delete(data_x_unseen, 0, axis=1)

            prediction = np.expand_dims(prediction, axis=2)
            predicted_day = np.concatenate((predicted_day, prediction), axis=2)

        predicted_day = np.delete(predicted_day, 0, axis=2)

        return predicted_train, predicted_val, predicted_day

    async def graph_preds(self, data_date, num_data_points, predicted_train, predicted_val, predicted_day,
                          data_close_price, scaler, split_index, predi_days=5):

        predicted_train_open = predicted_train.T[0]
        predicted_val_open = predicted_val.T[0]
        predicted_train_high = predicted_train.T[1]
        predicted_val_high = predicted_val.T[1]
        predicted_train_low = predicted_train.T[2]
        predicted_val_low = predicted_val.T[2]
        predicted_train_close = predicted_train.T[3]
        predicted_val_close = predicted_val.T[3]

        to_plot_data_y_train_pred_open = np.zeros(num_data_points)
        to_plot_data_y_train_pred_high = np.zeros(num_data_points)
        to_plot_data_y_train_pred_low = np.zeros(num_data_points)
        to_plot_data_y_train_pred_close = np.zeros(num_data_points)

        to_plot_data_y_val_pred_open = np.zeros(num_data_points)
        to_plot_data_y_val_pred_high = np.zeros(num_data_points)
        to_plot_data_y_val_pred_low = np.zeros(num_data_points)
        to_plot_data_y_val_pred_close = np.zeros(num_data_points)

        predicted_train_open = np.array([predicted_train_open]).T
        predicted_train_high = np.array([predicted_train_high]).T
        predicted_train_low = np.array([predicted_train_low]).T
        predicted_train_close = np.array([predicted_train_close]).T

        predicted_val_open = np.array([predicted_val_open]).T
        predicted_val_high = np.array([predicted_val_high]).T
        predicted_val_low = np.array([predicted_val_low]).T
        predicted_val_close = np.array([predicted_val_close]).T

        predicted_train_open = scaler.inverse_transform(predicted_train_open).T[0]
        predicted_train_high = scaler.inverse_transform(predicted_train_high).T[0]
        predicted_train_low = scaler.inverse_transform(predicted_train_low).T[0]
        predicted_train_close = scaler.inverse_transform(predicted_train_close).T[0]

        predicted_val_open = scaler.inverse_transform(predicted_val_open).T[0]
        predicted_val_high = scaler.inverse_transform(predicted_val_high).T[0]
        predicted_val_low = scaler.inverse_transform(predicted_val_low).T[0]
        predicted_val_close = scaler.inverse_transform(predicted_val_close).T[0]

        to_plot_data_y_train_pred_open[self.window_size:split_index + self.window_size] = predicted_train_open
        to_plot_data_y_train_pred_high[self.window_size:split_index + self.window_size] = predicted_train_high
        to_plot_data_y_train_pred_low[self.window_size:split_index + self.window_size] = predicted_train_low
        to_plot_data_y_train_pred_close[self.window_size:split_index + self.window_size] = predicted_train_close

        to_plot_data_y_val_pred_open[split_index + self.window_size:] = predicted_val_open
        to_plot_data_y_val_pred_high[split_index + self.window_size:] = predicted_val_high
        to_plot_data_y_val_pred_low[split_index + self.window_size:] = predicted_val_low
        to_plot_data_y_val_pred_close[split_index + self.window_size:] = predicted_val_close

        to_plot_data_y_train_pred_open = np.where(to_plot_data_y_train_pred_open == 0, None,
                                                  to_plot_data_y_train_pred_open)
        to_plot_data_y_train_pred_high = np.where(to_plot_data_y_train_pred_high == 0, None,
                                                  to_plot_data_y_train_pred_high)
        to_plot_data_y_train_pred_low = np.where(to_plot_data_y_train_pred_low == 0, None,
                                                 to_plot_data_y_train_pred_low)
        to_plot_data_y_train_pred_close = np.where(to_plot_data_y_train_pred_close == 0, None,
                                                   to_plot_data_y_train_pred_close)

        to_plot_data_y_val_pred_open = np.where(to_plot_data_y_val_pred_open == 0, None, to_plot_data_y_val_pred_open)
        to_plot_data_y_val_pred_high = np.where(to_plot_data_y_val_pred_high == 0, None, to_plot_data_y_val_pred_high)
        to_plot_data_y_val_pred_low = np.where(to_plot_data_y_val_pred_low == 0, None, to_plot_data_y_val_pred_low)
        to_plot_data_y_val_pred_close = np.where(to_plot_data_y_val_pred_close == 0, None,
                                                 to_plot_data_y_val_pred_close)

        data_close_price = data_close_price.reshape(-1, 1)

        fag = pl.Figure()
        fag.add_trace(
            pl.Scatter(name="Actual prices", x=data_date, y=data_close_price.T[0], line=dict(color="#fd7f20")))
        fag.add_trace(pl.Scatter(name="Predicted prices (train)", x=data_date, y=to_plot_data_y_train_pred_close,
                                 line=dict(color="#fdb750")))
        fag.add_trace(pl.Scatter(name="Predicted prices (test)", x=data_date, y=to_plot_data_y_val_pred_close,
                                 line=dict(color="#d3d3cb")))
        fag.update_layout(title_text="Data prediction")
        fag.update_xaxes(showgrid=True, ticklabelmode="period")
        fag.update_layout(xaxis_rangeslider_visible=False)

        plot_range = 10
        to_plot_data_y_val = np.zeros(plot_range - 1)

        to_plot_data_y_val_pred_open = np.zeros(plot_range - 1)
        to_plot_data_y_val_pred_high = np.zeros(plot_range - 1)
        to_plot_data_y_val_pred_low = np.zeros(plot_range - 1)
        to_plot_data_y_val_pred_close = np.zeros(plot_range - 1)

        to_plot_data_y_test_pred_open = np.zeros(plot_range - 1)
        to_plot_data_y_test_pred_high = np.zeros(plot_range - 1)
        to_plot_data_y_test_pred_low = np.zeros(plot_range - 1)
        to_plot_data_y_test_pred_close = np.zeros(plot_range - 1)

        data_y_val = data_close_price.T[0]
        to_plot_data_y_val[:plot_range - 1] = data_y_val[-plot_range + 1:]

        to_plot_data_y_val_pred_open[:plot_range - 1] = predicted_val_open[-plot_range + 1:]
        to_plot_data_y_val_pred_high[:plot_range - 1] = predicted_val_high[-plot_range + 1:]
        to_plot_data_y_val_pred_low[:plot_range - 1] = predicted_val_low[-plot_range + 1:]
        to_plot_data_y_val_pred_close[:plot_range - 1] = predicted_val_close[-plot_range + 1:]

        plot_date_test = data_date[-plot_range + 1:]

        PREDI_VAL_open = []
        PREDI_VAL_high = []
        PREDI_VAL_low = []
        PREDI_VAL_close = []

        Day_to_predict = predi_days
        prediction = predicted_day.T

        for day_predi in range(Day_to_predict):
            prediction_open = prediction[day_predi][0]
            prediction_high = prediction[day_predi][1]
            prediction_low = prediction[day_predi][2]
            prediction_close = prediction[day_predi][3]

            predi_conv_open = np.array([prediction_open]).T
            predi_conv_high = np.array([prediction_high]).T
            predi_conv_low = np.array([prediction_low]).T
            predi_conv_close = np.array([prediction_close]).T

            predi_conv_open = scaler.inverse_transform(predi_conv_open).T[0]
            predi_conv_high = scaler.inverse_transform(predi_conv_high).T[0]
            predi_conv_low = scaler.inverse_transform(predi_conv_low).T[0]
            predi_conv_close = scaler.inverse_transform(predi_conv_close).T[0]

            to_plot_data_y_val = np.append(to_plot_data_y_val, 0)

            to_plot_data_y_val_pred_open = np.append(to_plot_data_y_val_pred_open, 0)
            to_plot_data_y_val_pred_high = np.append(to_plot_data_y_val_pred_high, 0)
            to_plot_data_y_val_pred_low = np.append(to_plot_data_y_val_pred_low, 0)
            to_plot_data_y_val_pred_close = np.append(to_plot_data_y_val_pred_close, 0)

            to_plot_data_y_test_pred_open = np.append(to_plot_data_y_test_pred_open, predi_conv_open)
            to_plot_data_y_test_pred_high = np.append(to_plot_data_y_test_pred_high, predi_conv_high)
            to_plot_data_y_test_pred_low = np.append(to_plot_data_y_test_pred_low, predi_conv_low)
            to_plot_data_y_test_pred_close = np.append(to_plot_data_y_test_pred_close, predi_conv_close)

            PREDI_VAL_open.append(predi_conv_open)
            PREDI_VAL_high.append(predi_conv_high)
            PREDI_VAL_low.append(predi_conv_low)
            PREDI_VAL_close.append(predi_conv_close)

            new_day = plot_date_test[-1] + np.timedelta64(1, 'D')
            plot_date_test = np.append(plot_date_test, new_day)

            to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

            to_plot_data_y_val_pred_open = np.where(to_plot_data_y_val_pred_open == 0, None,
                                                    to_plot_data_y_val_pred_open)
            to_plot_data_y_val_pred_high = np.where(to_plot_data_y_val_pred_high == 0, None,
                                                    to_plot_data_y_val_pred_high)
            to_plot_data_y_val_pred_low = np.where(to_plot_data_y_val_pred_low == 0, None, to_plot_data_y_val_pred_low)
            to_plot_data_y_val_pred_close = np.where(to_plot_data_y_val_pred_close == 0, None,
                                                     to_plot_data_y_val_pred_close)

            to_plot_data_y_test_pred_open = np.where(to_plot_data_y_test_pred_open == 0, None,
                                                     to_plot_data_y_test_pred_open)
            to_plot_data_y_test_pred_high = np.where(to_plot_data_y_test_pred_high == 0, None,
                                                     to_plot_data_y_test_pred_high)
            to_plot_data_y_test_pred_low = np.where(to_plot_data_y_test_pred_low == 0, None,
                                                    to_plot_data_y_test_pred_low)
            to_plot_data_y_test_pred_close = np.where(to_plot_data_y_test_pred_close == 0, None,
                                                      to_plot_data_y_test_pred_close)

        PREDI_VAL_open = np.array(PREDI_VAL_open)
        PREDI_VAL_high = np.array(PREDI_VAL_high)
        PREDI_VAL_low = np.array(PREDI_VAL_low)
        PREDI_VAL_close = np.array(PREDI_VAL_close)

        fog = pl.Figure()
        fog.add_trace(
            pl.Scatter(name="Actual prices", x=plot_date_test, y=to_plot_data_y_val, line=dict(color="#fd7f20")))
        fog.add_trace(pl.Scatter(name="Past predicted prices", x=plot_date_test, y=to_plot_data_y_val_pred_close,
                                 line=dict(color="#fdb750")))
        fog.add_trace(
            pl.Scatter(name="Predicted price for next day", x=plot_date_test, y=to_plot_data_y_test_pred_close,
                       marker_symbol="star", line=dict(color="#d3d3cb")))
        fog.add_trace(pl.Candlestick(name="Train Data Close", x=plot_date_test,
                                     open=to_plot_data_y_val_pred_open,
                                     high=to_plot_data_y_val_pred_high,
                                     low=to_plot_data_y_val_pred_low,
                                     close=to_plot_data_y_val_pred_close,
                                     increasing_line_color='cyan',
                                     decreasing_line_color='orange'))
        fog.add_trace(pl.Candlestick(name="Predict Data Close", x=plot_date_test,
                                     open=to_plot_data_y_test_pred_open,
                                     high=to_plot_data_y_test_pred_high,
                                     low=to_plot_data_y_test_pred_low,
                                     close=to_plot_data_y_test_pred_close))
        fog.update_layout(title_text="Stock Forecastings")
        fog.update_xaxes(showgrid=True, ticklabelmode="period")
        fog.update_layout(xaxis_rangeslider_visible=False)

        PREDI_VAL_open = np.append(to_plot_data_y_val, PREDI_VAL_open.T[0])
        PREDI_VAL_high = np.append(to_plot_data_y_val, PREDI_VAL_high.T[0])
        PREDI_VAL_low = np.append(to_plot_data_y_val, PREDI_VAL_low.T[0])
        PREDI_VAL_close = np.append(to_plot_data_y_val, PREDI_VAL_close.T[0])

        PREDI_VALS_open = [i for i in PREDI_VAL_open if i is not None]
        PREDI_VALS_high = [i for i in PREDI_VAL_high if i is not None]
        PREDI_VALS_low = [i for i in PREDI_VAL_low if i is not None]
        PREDI_VALS_close = [i for i in PREDI_VAL_close if i is not None]

        predi_time = plot_date_test[-(self.window_size + Day_to_predict):]

        FinalPred = pd.DataFrame(
            {'Date': predi_time, 'Open': PREDI_VALS_open, 'High': PREDI_VALS_high, 'Low': PREDI_VALS_low,
             'Close': PREDI_VALS_close})
        FinalPred = FinalPred.set_index('Date')
        FinalPred = FinalPred.tail(Day_to_predict)

        return fag, fog, FinalPred

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

        company_name = st.text_input("Enter a company name:")

        if "messages" not in st.session_state:
            st.session_state["messages"] = [ChatMessage(role = "assistant", content = "")]

        if company_name:
            if company_name in st.session_state.symbol:
                symbols = st.session_state.symbol[company_name]
            else:
                symbols = await self.get_stock_symbols(company_name)
                st.session_state.symbol[company_name] = symbols
            articles_string = ""
            financial_statement = ""
            # gnews_api_spec = await self.get_gnews_api_spec()
            # symbols = []
            st.write(f"Stock symbol(s) for {company_name}: {', '.join(symbols)}")
            for symbol in symbols:

                with st.spinner("Searching...."):
                    gnews_api_spec = await self.get_gnews_api_spec(symbol)
                    left_column, right_column = st.columns(2)
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
                        pass
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
                with left_column:
                    st.header(symbol)
                with right_column:
                    plot_placeholder = st.empty()
                with left_column:
                    plot_placeholder_daily = st.empty()
                    price_pred_placeholder = st.empty()
                with right_column:
                    table_placeholder_daily = st.empty()
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

                data_normalized, scaler = await self.normalized_data(stock_data)
                unseen_data, dataset_train, dataset_val, split_index = await self.data_on_percent(data_normalized,
                                                                                                  self.train_perc)
                print("total",dataset_val)
                net_in_out = dataset_val.y.shape[1]
                # print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
                # print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)
                model = DNN_Net(input_size=net_in_out, hidden_layer_size=32, num_layers=2, output_size=net_in_out,
                                dropout=0.2)
                if symbol in st.session_state.trained_model:
                    trained_model = st.session_state.trained_model[symbol]
                else:
                    trained_model = await self.train_model(model, dataset_train, dataset_val)
                    st.session_state.trained_model[symbol] = trained_model

                predicted_train, predicted_val, predicted_day = await self.launch_model(trained_model, dataset_train,
                                                                                        dataset_val, unseen_data,
                                                                                        predi_days=10)
                fag, fog, FinalPred = await self.graph_preds(stock_data[0],
                                                             num_data_points,
                                                             predicted_train,
                                                             predicted_val,
                                                             predicted_day,
                                                             stock_data[4],
                                                             scaler,
                                                             split_index,
                                                             predi_days=10)

                table_placeholder_daily.plotly_chart(fag, use_container_width=True)
                plot_placeholder_daily.plotly_chart(fog, use_container_width=True)
                price_pred_placeholder.dataframe(FinalPred)
                with st.chat_message("assistant"):
                    user_input = self.query
                    output = self.get_response(user_input, articles_string, financial_statement, temperature = 0)
                    st.session_state.messages.append(ChatMessage(role = "assistant", content = output))
if __name__ == "__main__":
    app = TradingBotApp()
    asyncio.run(app.run())