# Load required libraries:

import random
import datetime as dt
import yfinance as yf
import talib as ta
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import quantstats as qs
from scipy.optimize import minimize

start_train = dt.datetime(2014, 4, 1)
end_train = dt.datetime(2022, 9, 30)
start_test = dt.datetime(2022, 10, 1)
end_test = dt.datetime(2025, 3, 31)

start = start_train
end = end_test  

# stocks from NIFTY50 divided into their respective industry
oil_gas_stocks = ['RELIANCE.NS', 'COALINDIA.NS','ONGC.NS']
power_stocks = ['NTPC.NS', 'POWERGRID.NS']
technology_stocks = ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS']
fmcg_stocks = ['HINDUNILVR.NS', 'ITC.NS', 'TATACONSUM.NS', 'NESTLEIND.NS']
healthcare_stocks = ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'APOLLOHOSP.NS']
financial_stocks = ['HDFCLIFE.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'SBILIFE.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'HDFCBANK.NS', 'INDUSINDBK.NS', 'JIOFIN.NS', 'KOTAKBANK.NS', 'SHRIRAMFIN.NS', 'SBIN.NS']
materials_stocks = ['ULTRACEMCO.NS', 'GRASIM.NS']
auto_stocks = ['MARUTI.NS', 'EICHERMOT.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'M&M.NS']
capital_goods_stocks = ['BEL.NS']
construction_stocks = ['LT.NS']
consumer_durables_stocks = ['ASIANPAINT.NS', 'TITAN.NS']
consumer_services_stocks = ['Eternal.NS', 'TRENT.NS']
metal_mining_stocks = ['ADANIENT.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS']
services_stocks = ['ADANIPORTS.NS']
telecomm_stocks = ['BHARTIARTL.NS']

# Discard stock duplicate:
combined_stocks = list(set(
    oil_gas_stocks +
    power_stocks +
    technology_stocks +
    fmcg_stocks +
    healthcare_stocks +
    financial_stocks +
    materials_stocks +
    auto_stocks +
    capital_goods_stocks +
    construction_stocks +
    consumer_durables_stocks +
    consumer_services_stocks +
    metal_mining_stocks +
    services_stocks +
    telecomm_stocks
))

print("Combined list without duplicates:", combined_stocks)

# Create a list to store tickers with complete data:
valid_stocks = []

# Define the expected number of business days in the date range
trading_days = pd.date_range(start=start, end=end, freq='B')

for ticker in combined_stocks:
    data = yf.download(ticker, start=start, end=end)
    # Reindex the data to include all business days within the date range
    data = data.reindex(trading_days, method='ffill').dropna()
    # Check if the length of the data matches the number of trading days
    if len(data) == len(trading_days):
        valid_stocks.append(ticker)

print("Tickers with complete data:", valid_stocks)
# Load stocks with complete data:

print(f" you have {len(valid_stocks)} mix stocks in your portfolio.")
Tickers=valid_stocks

# Load data for valid_stocks:
stock_df = yf.download(Tickers, start=start, end=end, progress=False)
returns=stock_df['Close'].pct_change().dropna()

# Calculate the standard deviation of daily returns for each stock
volatility = returns.std()*np.sqrt(252)

# Sort the stocks by their volatility in descending order:
sorted_volatility = volatility.sort_values(ascending=False)
print( sorted_volatility)

# Add market features:
nsei_df=yf.download("^NSEI", start=start, end=end, progress=False)
vix_df=pd.read_csv(r"Data\India_Vix.csv", index_col=0, parse_dates=True)

# Add technical indicators as features:
def calculate_indicators(tickers, start, end):
    macd = []
    rsi = []
    cci = []
    adx = []
    stoch = []
    willr = []
    bb_upper = []
    bb_middle = []
    bb_lower = []
    mfi = []
    ema = []
    atr = []
    sar = []
    obv = []

    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start, end=end, progress=False)

            close_prices = stock_data['Close'].astype(np.float64).values.flatten()
            high_prices = stock_data['High'].astype(np.float64).values.flatten()
            low_prices = stock_data['Low'].astype(np.float64).values.flatten()
            volume = stock_data['Volume'].astype(np.float64).values.flatten()

            if close_prices.ndim > 1:
                close_prices = close_prices.flatten()
            if high_prices.ndim > 1:
                high_prices = high_prices.flatten()
            if low_prices.ndim > 1:
                low_prices = low_prices.flatten()
            if volume.ndim > 1:
                volume = volume.flatten()

            macd.append(ta.MACD(close_prices)[0])
            rsi.append(ta.RSI(close_prices))
            cci.append(ta.CCI(high_prices, low_prices, close_prices))
            adx.append(ta.ADX(high_prices, low_prices, close_prices))
            stoch_k, stoch_d = ta.STOCH(high_prices, low_prices, close_prices)
            stoch.append(stoch_k)  # Assuming you want to append %K line of Stochastic
            willr.append(ta.WILLR(high_prices, low_prices, close_prices))
            bb_upper_ticker, bb_middle_ticker, bb_lower_ticker = ta.BBANDS(close_prices)
            bb_upper.append(bb_upper_ticker)
            bb_middle.append(bb_middle_ticker)
            bb_lower.append(bb_lower_ticker)
            ema.append(ta.EMA(close_prices))
            atr.append(ta.ATR(high_prices, low_prices, close_prices))
            sar.append(ta.SAR(high_prices, low_prices ))
            obv.append(ta.OBV(close_prices, volume))

            # Calculate the Money Flow Index (MFI)
            mfi.append(ta.MFI(high=high_prices, low=low_prices, close=close_prices, volume=volume))

        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

    return np.array(macd), np.array(rsi), np.array(cci), np.array(adx), np.array(stoch), np.array(willr), np.array(bb_upper), np.array(bb_middle), np.array(bb_lower), np.array(mfi), np.array(ema), np.array(atr), np.array(sar), np.array(obv)


# Function to handle NaNs
def handle_nans(indicators, fill_value=0):
    inds_nan = np.isnan(indicators)
    if inds_nan.any():
        indicators = np.where(inds_nan, fill_value, indicators)
    return indicators

# Function to normalize indicators:
def normalize_indicators(indicators):
    indicators = handle_nans(indicators)  # Handle NaNs before normalization
    if indicators.ndim == 1:
        indicators = indicators.reshape(-1, 1)  # Reshape 1D array to 2D array

    min_val = np.min(indicators, axis=1, keepdims=True)
    max_val = np.max(indicators, axis=1, keepdims=True)

    # Avoid divide-by-zero error by adding a small epsilon
    epsilon = 1e-8
    normalized = (indicators - min_val) / (max_val - min_val + epsilon)

    print(f"Normalized indicators shape: {normalized.shape}")
    return normalized

# Normalize nsei and VIX :
normalized_nsei = normalize_indicators(nsei_df['Close'].values)
normalized_vix = normalize_indicators(vix_df['Close'].values)
print(f"Normalized nsei shape: {normalized_nsei.shape}")
print(f"Normalized VIX shape: {normalized_vix.shape}")

indicators = calculate_indicators(Tickers, start, end)

# Normalize indicators and store them in a dictionary:

normalized_indicators = {}
indicator_names = ['macd', 'rsi', 'cci', 'adx', 'stoch', 'willr', 'bb_upper', 'bb_middle', 'bb_lower', 'mfi', 'ema', 'atr', 'sar', 'obv']
for i, name in enumerate(indicator_names):
    normalized_indicators[name] = normalize_indicators(indicators[i])

# Checking shapes
for name in normalized_indicators:
    print(f"{name} shape: {normalized_indicators[name].shape}")

normalized_nsei=normalize_indicators(nsei_df['Close'].values)
normalized_vix=normalize_indicators(vix_df['Close'].values)

# Initialize parameters for model:

D = len(Tickers)

# state_dim = (14 technical indicators+ holdings)*D+ NSEI + VIX + balance + portfolio value:

state_dim = 15 * D + 4  # (14 technical indicators+ holdings)*D+ 
action_dim = D * 3  # Actions: Buy, Sell, Hold for each ticker
print(f"Calculated state dimension: {state_dim}")

# Data into training and testing sets:
train_df = stock_df.loc[start_train:end_train]
test_df = stock_df.loc[start_test:end_test]
train_nsei = nsei_df.loc[start_train:end_train]
test_nsei = nsei_df.loc[start_test:end_test]
train_vix = vix_df.loc[start_train:end_train]
test_vix = vix_df.loc[start_test:end_test]


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.2)
        self.actor = nn.Linear(128, action_dim)  
        self.critic = nn.Linear(128, 1)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# Ensure proper initialization
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def initialize_ppo_model(device):
    """
    Initialize the PPO model, optimizer, and loss function.
    """
    model = PPO(state_dim, action_dim).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    return model, optimizer, mse_loss

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def ppo_update(model, optimizer, mse_loss, memory, gamma, clip_epsilon, device, update_steps, batch_size):
    for _ in range(update_steps):
        if len(memory) < batch_size:
            continue  # Skip if not enough samples

        batch_indices = np.random.choice(len(memory), batch_size)
        batch = [memory[i] for i in batch_indices]

        states, actions, rewards, old_probs, next_states = zip(*batch)

        states = torch.stack(states).to(device).float()
        next_states = torch.stack(next_states).to(device).float()
        actions = torch.tensor(actions).to(device).long()
        old_probs = torch.tensor(old_probs).to(device).float()
        rewards = torch.tensor(rewards).to(device).float().unsqueeze(-1)  # shape [B, 1]

        new_probs, state_values = model(states)
        _, next_state_values = model(next_states)

        # Compute advantages
        with torch.no_grad():
            target_values = rewards + gamma * next_state_values
            advantages = target_values - state_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get new log probs of taken actions
        new_probs = new_probs.gather(1, actions.unsqueeze(1))

        # Compute ratio
        ratio = new_probs / (old_probs.unsqueeze(-1) + 1e-8)
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        critic_loss = mse_loss(state_values, target_values)

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

'''    
The advantage of PPO is its flexibility and robustness to different settings, so one can experiment with its hyperparameters to see what works best for each scenario.
'''

def train_ppo(train_df, episodes=250, gamma=0.997, epsilon=0.08, clip_epsilon=0.15, update_steps=20, batch_size=128, early_stop_threshold=0.002, patience=60, lr=0.0005):
    """
    Trains a Proximal Policy Optimization (PPO) model for stock trading.

    Args:
        train_df (pd.DataFrame): DataFrame containing historical stock data.
        episodes (int, optional): Number of training episodes. Defaults to 100.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
        epsilon (float, optional): Probability of taking a random action. Defaults to 0.1.
        clip_epsilon (float, optional): Clipping parameter for PPO. Defaults to 0.2.
        update_steps (int, optional): Number of PPO update steps per batch. Defaults to 10.
        batch_size (int, optional): Batch size for PPO updates. Defaults to 32.
        early_stop_threshold (float, optional): Early stopping threshold for average reward improvement. Defaults to 0.001.
        patience (int, optional): Number of episodes to wait for improvement before early stopping. Defaults to 10.

    Returns:
        tuple: A tuple containing the final holdings, final portfolio value, list of tickers with buy or hold actions,
               normalized final weights of tickers, and a DataFrame of final selected tickers and their weights.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO(state_dim, action_dim).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    tickers_with_last_buy_or_hold = set()
    total_iterations = 0

    final_holdings = None
    final_portfolio_value = None
    rewards_history = []
    portfolio_values_history = []
    best_avg_reward = -float('inf')
    patience_counter = 0

    for episode in range(episodes):
        initial_holdings = np.zeros(D)
        initial_balance = 100000
        initial_portfolio_value = initial_balance + np.sum(train_df['Close'].iloc[0] * initial_holdings)
        current_step = 0

        holdings = initial_holdings.copy()
        balance = initial_balance
        portfolio_value = initial_portfolio_value

        episode_rewards = []
        memory = []

        while current_step < len(train_df) - 1:
            state_components = []
            for i, ticker in enumerate(Tickers):
                for name in indicator_names:
                    state_components.append(normalized_indicators[name][i, current_step])
            state_components.extend([normalized_nsei[current_step, 0], normalized_vix[current_step, 0]])
            state_components.extend(holdings)
            state_components.extend([balance, portfolio_value])
            state = torch.FloatTensor(state_components).float().to(device)

            action_probs, state_value = model(state)

            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                raise ValueError("NaNs or Infs found in action probabilities")

            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                action = torch.argmax(action_probs).item()

            action_type = action % 3
            ticker_idx = action // 3
            action_desc = ['hold', 'buy', 'sell'][action_type]

            if action_desc in ['buy', 'hold']:
                tickers_with_last_buy_or_hold.add(Tickers[ticker_idx])
            else:
                tickers_with_last_buy_or_hold.discard(Tickers[ticker_idx])

            next_prices = train_df['Close'].iloc[current_step + 1]
            next_portfolio_value = balance + np.sum(next_prices * holdings)

            max_shares = 3
            price = next_prices.iloc[ticker_idx]
            if action_desc == 'buy' and balance >= price:
                shares = 1
                holdings[ticker_idx] += shares
                balance -= shares * price
            elif action_desc == 'sell' and holdings[ticker_idx] > 0:
                shares = min(holdings[ticker_idx], max_shares)
                holdings[ticker_idx] -= shares
                balance += shares * price


            transaction_cost = 0.0002  # ~0.02% penalty
            drawdown = max(0, (portfolio_value - next_portfolio_value) / portfolio_value)
            reward = (next_portfolio_value - portfolio_value) / portfolio_value
            reward -= transaction_cost * (action_type != 0)
            reward -= 0.005 * drawdown  # penalize sharp drops
            # portfolio_volatility = np.std(holdings * train_df['Close'].pct_change().iloc[current_step])
            # reward -= 0.01 * portfolio_volatility
            episode_rewards.append(reward)
            portfolio_value = next_portfolio_value

            next_state_components = []
            for i, ticker in enumerate(Tickers):
                for name in indicator_names:
                    next_state_components.append(normalized_indicators[name][i, current_step + 1])
            next_state_components.extend([normalized_nsei[current_step + 1, 0], normalized_vix[current_step + 1, 0]])
            next_state_components.extend(holdings)
            next_state_components.extend([balance, portfolio_value])
            next_state = torch.FloatTensor(next_state_components).float().to(device)

            _, next_state_value = model(next_state)
            advantage = reward + gamma * next_state_value.item() - state_value.item()
            target_value = torch.FloatTensor([reward + gamma * next_state_value.item() + 1e-8]).to(device).view_as(state_value)
            critic_loss = mse_loss(state_value, target_value)
            actor_loss = -torch.log(action_probs[action] + 1e-8) * advantage
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            memory.append((state, action, reward, action_probs[action].item(), next_state))
            current_step += 1

            if len(memory) >= 128:
                ppo_update(model, optimizer, mse_loss, memory, gamma, clip_epsilon, device, update_steps, batch_size)
                memory = []

        final_holdings = holdings
        final_portfolio_value = portfolio_value
        total_reward = np.sum(episode_rewards)
        rewards_history.append(total_reward)
        portfolio_values_history.append(portfolio_value)

        avg_reward = np.mean(rewards_history[-patience:])
        print(f"Episode {episode+1}/{episodes}, Reward: {total_reward}, Avg Reward: {avg_reward}")

        if avg_reward > best_avg_reward + early_stop_threshold:
            best_avg_reward = avg_reward
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at episode {episode+1}")
            break

    tickers_list = list(tickers_with_last_buy_or_hold)
    print(f"Tickers with 'buy' or 'hold' actions at the end of the last iteration: {tickers_with_last_buy_or_hold}")
    print("Final portfolio value:", final_portfolio_value)
    print("Final holdings:")
    for i, ticker in enumerate(Tickers):
        print(f"{ticker} holdings: {final_holdings[i]}")

    weights = final_holdings * train_df['Close'].iloc[-1]
    total_weights = np.sum(weights)
    normalized_weights = weights / total_weights

    final_weights_df = pd.DataFrame({
        'Ticker': Tickers,
        'Weight': normalized_weights
    })

    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, label='Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward History')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values_history, label='Portfolio Value')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Values History')
    plt.legend()
    plt.show()

    return final_holdings, final_portfolio_value, tickers_list, normalized_weights, final_weights_df


# Train the PPO model
final_holdings, final_portfolio_value, tickers_list, normalized_weights, final_weights_df = train_ppo(train_df )

# Display results
print("Best Holdings:", final_holdings)
print("Best Portfolio Value:", final_portfolio_value)
print("Tickers with Last Buy or Hold:", tickers_list)
print("Best Weights:", normalized_weights)


# Define comparison portfolios:

def calculate_cvar(returns, confidence_level=0.95):
    """
    Calculates the Conditional Value at Risk (CVaR) of a portfolio.

    Args:
        returns (np.ndarray): Array of portfolio returns.
        confidence_level (float, optional): Confidence level for VaR calculation. Defaults to 0.95.

    Returns:
        float: The Conditional Value at Risk (CVaR) of the portfolio.
    """
    # Calculate VaR
    var = np.percentile(returns, 100 * (1 - confidence_level))

    # Calculate CVaR
    cvar = returns[returns <= var].mean()

    return cvar

# Define mean conditional value-at-risk algorithm:

def mCVAR_optimization(returns, confidence_level=0.95):
    """
    Optimizes the portfolio weights to minimize Mean Conditional Value at Risk (mCVaR).

    Args:
        returns (pd.DataFrame): DataFrame containing historical stock returns.
        confidence_level (float, optional): Confidence level for VaR calculation. Defaults to 0.95.

    Returns:
        np.ndarray: Optimal portfolio weights that minimize mCVaR.
    """
    tickers = returns.columns
    n_tickers = len(tickers)

    # Objective function to minimize mCVaR
    def objective(weights):
        portfolio_returns = returns.dot(weights)
        return calculate_cvar(portfolio_returns, confidence_level)

    # Constraints: Weights sum to 1
    constraints = ({
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    })

    # Bounds: Weights between 0 and 1
    bounds = [(0, 1)] * n_tickers

    # Initial guess for weights (equal distribution)
    initial_weights = np.ones(n_tickers) / n_tickers

    # Minimize the objective function
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

    return result.x


# Definie Hierarchical Risk Parity algorithm:

def get_IVP(cov, **kargs):
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def get_cluster_var(cov, cItems):
    cov_ = cov.loc[cItems, cItems]
    w_ = get_IVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar

def get_quasi_diag(link):
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()

def get_rec_bipart(cov, sortIx):
    w = pd.Series(1.0, index=sortIx)  # Ensure w is of float type
    cItems = [sortIx]
    while len(cItems) > 0:
        cItems = [i[int(j):int(k)] for i in cItems for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]
        for i in range(0, len(cItems), 2):
            cItems0 = cItems[i]
            cItems1 = cItems[i + 1]
            cVar0 = get_cluster_var(cov, cItems0)
            cVar1 = get_cluster_var(cov, cItems1)
            alpha = float(1 - cVar0 / (cVar0 + cVar1))  # Explicitly cast alpha to float
            w[cItems0] *= alpha
            w[cItems1] *= 1 - alpha
    return w

def HRP_Allocation(returns):
    cov = returns.cov()
    corr = returns.corr()
    dist = squareform(((1 - corr) / 2.)**.5)
    link = linkage(dist, 'single')
    sortIx = get_quasi_diag(link)
    sortIx = returns.columns[sortIx].tolist()
    hrp = get_rec_bipart(cov, sortIx)
    return hrp.sort_index()

# Mean variance optimization (MVO) algorithm:

def optimize_portfolio(returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def portfolio_performance(weights):
        portfolio_returns = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_returns, portfolio_volatility

    def negative_sharpe_ratio(weights, risk_free_rate=0):
        p_returns, p_volatility = portfolio_performance(weights)
        return - (p_returns - risk_free_rate) / p_volatility

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(returns.shape[1]))
    initial_guess = returns.shape[1] * [1. / returns.shape[1]]

    optimized_result = minimize(negative_sharpe_ratio, initial_guess,
                                method='SLSQP', bounds=bounds,
                                constraints=constraints)

    return optimized_result.x


# Display the head of train and test returns dataframes
# print("Train Returns:")
# # print(train_returns.head())

# print("\nTest Returns:")
# # print(test_returns.head())


# Function to calculate portfolio returns:
def calculate_portfolio_returns(weights, test_returns):
    portfolio_returns = (weights * test_returns).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()-1
    return cumulative_returns

# Align indices of returns with stock_df data:
returns_df = returns.reindex(stock_df.index)

# Split the data into train and test sets
# train_df, test_df = train_test_split(stock_df, test_size=0.2, shuffle=False)

# Get the indices for train and test data
train_indices = train_df.index
test_indices = test_df.index

# Filter the returns dataframe using the train and test indices
train_returns = returns_df.loc[start_train:end_train]
test_returns = returns_df.loc[start_test:end_test]

train_returns = train_returns.dropna(axis=0, how='any')
test_returns = test_returns.dropna(axis=0, how='any')

# Display the head of the correctly split train and test returns dataframes
print("\nTrain Returns (using manual dates):")
print(train_returns.head())
print(f"Shape: {train_returns.shape}")


print("\nTest Returns (using manual dates):")
print(test_returns.head())
print(f"Shape: {test_returns.shape}")

# Filter the test returns to include only the selected tickers returned by PPO model:

selected_tickers = final_weights_df['Ticker']
test_returns_filtered = test_returns[selected_tickers]

# Convert the weights to a numpy array
drl_weights = final_weights_df['Weight'].values

# Calculate the portfolio returns on the test data
drl_portfolio_returns = test_returns_filtered.dot(drl_weights)

# Calculate cumulative returns for the portfolio
drl_cumulative_returns = (1 + drl_portfolio_returns).cumprod() - 1

# Display the portfolio returns and cumulative returns
print("DRL Portfolio Returns on Test Data:")
print(drl_portfolio_returns.head())

print("\nDRL Cumulative Returns on Test Data:")
print(drl_cumulative_returns.head())

# MVO portfolio on test data:

# Optimize portfolio using train returns:
mvo_weights = optimize_portfolio(train_returns)
print(mvo_weights)
mvo_portfolio_returns = test_returns.dot(mvo_weights)
mvo_cumulative_returns = calculate_portfolio_returns(mvo_weights, test_returns)

# HRP portfolio on test data:

hrp_weights = HRP_Allocation(train_returns)

print("HRP train weights:\n")
print(hrp_weights)
print()

hrp_portfolio_returns = test_returns.dot(hrp_weights)
hrp_cumulative_returns = calculate_portfolio_returns(hrp_weights, test_returns)

# MCVAR portfolio on test data:
mcvar_weights = mCVAR_optimization(train_returns)
print(" mCVAR Optimal Portfolio Weights:", mcvar_weights)
mcvar_portfolio_returns = test_returns.dot(mcvar_weights)
mcvar_cumulative_returns = calculate_portfolio_returns(mcvar_weights, test_returns)

# Create DataFrames to compare HRP , MVO and DRL  performance:

performance_df = pd.DataFrame({
    'DRL Cumulative Returns': drl_cumulative_returns,
    'MVO Cumulative Returns': mvo_cumulative_returns,
    'HRP Cumulative Returns': hrp_cumulative_returns,
    'mCVAR Cumulative Returns': mcvar_cumulative_returns
})

print("Performance Comparison:\n", performance_df)

# Plot the performance:
performance_df.plot(title="DRL vs HRP vs MVO vs mCVAR Performance")


mvo_last_cumulative_return = mvo_cumulative_returns.iloc[-1]
hrp_last_cumulative_return = hrp_cumulative_returns.iloc[-1]
drl_last_cumulative_return = drl_cumulative_returns.iloc[-1]
mcvar_last_cumulative_return = mcvar_cumulative_returns.iloc[-1]

last_cumul = {
    'cumulative_returns[-1]': [
        mvo_last_cumulative_return,
        hrp_last_cumulative_return,
        drl_last_cumulative_return,
        mcvar_last_cumulative_return
    ]
}

port_names = ['MVO', 'HRP', 'DRL', 'mCVAR']

cumul_df = pd.DataFrame(last_cumul, index=port_names)

print(cumul_df)


# Use quantstats to define the function to calculate performance metrics :

# Existing portfolio data
portfolios = {
    "MVO Portfolio": mvo_portfolio_returns,
    "HRP Portfolio": hrp_portfolio_returns,
    "DRL Portfolio": drl_portfolio_returns,
    "mCVAR Portfolio": mcvar_portfolio_returns
}

# List of metrics to store
metrics = ['Sharpe Ratio', 'Omega Ratio', 'Volatility', 'Max Drawdown', 'Sortino Ratio', 'Calmar Ratio', 'Tail Ratio', 'Risk Return', 'Skew', 'Kurtosis']

# Create an empty dictionary to store the metrics
metrics_data = {'Metrics': metrics}

# Initialize risk-free rate
risk_free_rate = 0.0419

# Function to calculate performance metrics:

def calculate_performance_metrics(returns, risk_free_rate=0.0):
    returns_series = pd.Series(returns).dropna()
    
    # Fix: Make sure returns are properly formatted for quantstats functions
    sharpe_ratio = qs.stats.sharpe(returns_series, rf=risk_free_rate, periods=252, annualize=True)
    
    # Fix: For omega ratio, convert Series to numeric values if needed
    try:
        # Pass the Series directly instead of converting to DataFrame
        omega_ratio = qs.stats.omega(returns_series, required_return=0.0, rf=risk_free_rate, periods=252)
    except Exception as e:
        print(f"Error calculating Omega ratio: {e}")
        omega_ratio = None
        
    volatility = qs.stats.volatility(returns_series, periods=252, annualize=True)
    max_drawdown = qs.stats.max_drawdown(returns_series)
    sortino_ratio = qs.stats.sortino(returns_series, rf=risk_free_rate, periods=252)
    
    # Fix: For functions that use prepare_returns, set it to False since we're already passing returns
    calmar_ratio = qs.stats.calmar(returns_series)
    tail_ratio = qs.stats.tail_ratio(returns_series, cutoff=0.95)
    risk_return = qs.stats.risk_return_ratio(returns_series)
    skew = qs.stats.skew(returns_series)
    kurtosis = qs.stats.kurtosis(returns_series)

    return {
        'Sharpe Ratio': sharpe_ratio,
        'Omega Ratio': omega_ratio,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Tail Ratio': tail_ratio,
        'Risk Return': risk_return,
        'Skew': skew,
        'Kurtosis': kurtosis
    }

# Calculate metrics for each portfolio and store in the dictionary:

for name, returns in portfolios.items():
    print(f"\nPerformance Metrics for {name}:")
    performance_metrics = calculate_performance_metrics(returns, risk_free_rate)
    metrics_data[name] = [performance_metrics[metric] for metric in metrics]

# Convert the dictionary to a DataFrame
metrics_df = pd.DataFrame(metrics_data)
metrics_df.set_index('Metrics', inplace=True)

# Display the DataFrame
print(metrics_df)