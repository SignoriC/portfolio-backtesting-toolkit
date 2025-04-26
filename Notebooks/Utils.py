"""Functions for portfolio analyses"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def prices(ticker, source='yf', price='Adj Close'):
    """Retrieves daily price data for a given ticker symbol from the Milan Stock Exchange (Borsa Italiana).

    Args:
        ticker (str): The ticker symbol of the stock.
        source (str, optional): The source of price data to retrive. Can be either 'csv' or 'yf'.
            - 'csv': Reads data from a CSV file named '{ticker}.csv' in the 'historical_series/' subdirectory.
            - 'yf' (default): Fetches data from Yahoo Finance. '.MI' is automatically appended.
        price (str, optional): The type of price data to retrieve. Can be either 'Close' or 'Adj Close'.
            - 'Close': Takes the Closing price of the ticker.
            - 'Adj Close' (default): Takes the Adjusted Closing price of the ticker.

    Returns:
        pandas.Series: daily price data for the specified ticker, indexed by date.
    """
    if source == 'csv':
        data = pd.read_csv('historical_series/' + ticker + '.csv')
        # Convert 'Date' column to datetime objects if it's not already
        data['Date'] = pd.to_datetime(data['Date'])
        # sorting data by 'Date' and resetting index
        data = data.sort_values(by='Date').set_index('Date')
        data = data.loc[:,'Price']
            
    else:
        if price != 'Adj Close':
            # append '.MI' to the ticker to fetch data from the Milan Stock Exchange.
            data = yf.download(tickers=ticker+'.MI', 
                                progress=False).loc[:,('Close',ticker+'.MI')]
        else:
            # append '.MI' to the ticker to fetch data from the Milan Stock Exchange.
            data = yf.download(tickers=ticker+'.MI', 
                                progress=False, auto_adjust=False).loc[:,('Adj Close',ticker+'.MI')]
    
    # giving ticker's name to data
    data.name = ticker
    
    return data

def portfolio_prices(data):
    """Structures in a DataFrame daily price data of tickers in portfolio starting from a common date.
    
    Args:
        data (list): List of pd.Series per each ticker historical prices.

    Returns:
        pandas.DataFrame: Daily price data of tickers in portfolio, indexed by date.
    
    """
    return pd.concat(data, axis=1, join='inner')

def backtest_portfolio(prices: pd.DataFrame,
                       weights: list,
                       rebalance: bool = False,
                       rebalance_freq: int = 252,
                       initial_value: float = 1.0) -> pd.DataFrame:
    """
    Backtest a portfolio with optional rebalancing, returning asset and portfolio normalized values.

    Parameters:
    - prices: DataFrame of asset prices (rows = dates, columns = tickers)
    - weights: List or array of asset weights (same order as columns)
    - rebalance: Whether to rebalance periodically (default = False)
    - rebalance_freq: Frequency of rebalancing in trading days (default = 252 = yearly)
    - initial_value: Initial portfolio value (default = 1.0)

    Returns:
    - DataFrame of normalized values per asset and total portfolio
    """
    
    # Step 1: Normalize prices
    norm_prices = prices / prices.iloc[0]
    weights = np.array(weights)

    # Step 2: Rebalance dates
    if rebalance:
        idx_rebalance = np.arange(0, len(norm_prices), rebalance_freq)
    else:
        idx_rebalance = [0]

    # Step 3: Prepare output DataFrame
    asset_vals = pd.DataFrame(index=norm_prices.index, columns=norm_prices.columns, dtype='float64')
    portfolio_vals = pd.Series(index=norm_prices.index, dtype='float64')

    portfolio_value = initial_value

    for i in range(len(idx_rebalance)):
        start_idx = idx_rebalance[i]
        end_idx = idx_rebalance[i + 1] if i + 1 < len(idx_rebalance) else len(norm_prices)

        period_prices = norm_prices.iloc[start_idx:end_idx]
        start_prices = period_prices.iloc[0]

        # Allocate value per asset
        money_allocation = portfolio_value * weights
        units_held = money_allocation / start_prices

        # Calculate values for each asset and total portfolio
        period_asset_vals = period_prices.multiply(units_held, axis=1)
        period_portfolio_vals = period_asset_vals.sum(axis=1)

        # Store results
        asset_vals.iloc[start_idx:end_idx] = period_asset_vals
        portfolio_vals.iloc[start_idx:end_idx] = period_portfolio_vals

        # Update portfolio value for next rebalance period
        portfolio_value = period_portfolio_vals.iloc[-1]

        if not rebalance:
            break

    # Combine into one DataFrame
    result = asset_vals.copy()
    result['Portfolio'] = portfolio_vals
    return result

def portfolio_performance(pf: pd.Series, risk_free_rate: float = 0.0) -> pd.Series:
    """
    Calculates portfolio performance indicators from a time series of daily portfolio values.

    Parameters:
    - pf: pd.Series with portfolio values indexed by date
    - risk_free_rate: Annual risk-free rate (default = 0.0)

    Returns:
    - pd.Series with Total Return, Annual Return, Annual Volatility, and Sharpe Ratio
    """
    indicators = ['Total_Return', 'Annual_Return', 'Annual_Volatility', 'Sharpe']
    perf = pd.Series(index=indicators, dtype='float64')

    # Length of backtest in years
    pf_years = (pf.index[-1] - pf.index[0]) / pd.to_timedelta('365.25D')

    # Daily returns
    daily_returns = pf.pct_change().dropna()

    # Metrics
    total_return = pf.iloc[-1] / pf.iloc[0] - 1
    annual_return = (1 + total_return) ** (1 / pf_years) - 1
    annual_volatility = daily_returns.std() * np.sqrt(252)

    # Sharpe Ratio (excess return over volatility)
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_volatility if annual_volatility != 0 else np.nan

    # Fill output
    perf['Total_Return'] = total_return
    perf['Annual_Return'] = annual_return
    perf['Annual_Volatility'] = annual_volatility
    perf['Sharpe'] = sharpe_ratio

    return perf

def drawdown_metrics(ticker: pd.Series) -> pd.DataFrame:
    """
    Calculates drawdown metrics for a given time series of portfolio or asset values.

    Identifies drawdown periods where the ticker value is below its cumulative max, 
    and returns the top 10 by maximum depth.

    Parameters:
    - ticker: pd.Series of daily portfolio values (indexed by date)

    Returns:
    - pd.DataFrame with columns:
        - 'drawdown': Maximum drawdown (as negative decimal)
        - 'start': Start date of drawdown
        - 'valley': Date of drawdown minimum
        - 'end': Recovery date (NaT if not recovered)
    """
    # Calculate drawdown as percent decline from peak
    cummax = ticker.cummax()
    dd = ticker / cummax - 1
    underwater = dd < 0

    # Identify start and end of drawdown periods
    start_idxs, end_idxs = [], []
    for i in range(1, len(underwater)):
        if underwater.iloc[i] and not underwater.iloc[i - 1]:
            start_idxs.append(i)
        elif not underwater.iloc[i] and underwater.iloc[i - 1]:
            end_idxs.append(i)

    # Handle case where drawdown hasn't recovered yet
    if len(start_idxs) > len(end_idxs):
        end_idxs.append(len(ticker) - 1)

    # Calculate drawdown stats
    data = []
    for start, end in zip(start_idxs, end_idxs):
        valley_idx = dd.iloc[start:end + 1].idxmin()
        drawdown = dd.loc[valley_idx]
        data.append({
            'drawdown': drawdown,
            'start': ticker.index[start],
            'valley': valley_idx,
            'end': ticker.index[end] if not underwater.iloc[end] else pd.NaT
        })

    # Return top 10 deepest drawdowns
    df = pd.DataFrame(data)
    df = df.sort_values('drawdown').head(10).reset_index(drop=True)
    return df

def backtest_sim(ticker_prices, initial_capital, initial_weights, rebalance_frequency=None):
    """
    Simulates a portfolio backtest with discrete shares and optional rebalancing.

    Parameters:
    - ticker_prices (pd.DataFrame): Daily prices of each asset (indexed by date)
    - initial_capital (float): Starting capital
    - initial_weights (np.array): Portfolio weights
    - rebalance_frequency (int or None): Rebalance every N steps. If None, no rebalancing.

    Returns:
    - shares (pd.DataFrame): Number of shares held per asset over time
    - portfolio_values (pd.Series): Portfolio value over time
    """
    shares = pd.DataFrame(0, index=ticker_prices.index, columns=ticker_prices.columns, dtype=np.int64)
    portfolio_values = pd.Series(index=ticker_prices.index, name='Portfolio', dtype=np.float64)

    # Buy initial shares
    prices_0 = ticker_prices.iloc[0]
    shares.iloc[0] = (initial_weights * initial_capital) // prices_0
    portfolio_values.iloc[0] = (shares.iloc[0] * prices_0).sum()

    # Define rebalancing dates (indices)
    rebalance_dates = set(np.arange(0, len(ticker_prices), rebalance_frequency)) if rebalance_frequency else set()

    for t in range(1, len(ticker_prices)):
        date = ticker_prices.index[t]
        prev_date = ticker_prices.index[t - 1]

        # Default: carry forward previous shares
        shares.loc[date] = shares.loc[prev_date]

        # Portfolio value before (or after) rebalance
        portfolio_values.at[date] = (shares.loc[date] * ticker_prices.loc[date]).sum()

        # Rebalance if needed
        if t in rebalance_dates:
            new_shares = (initial_weights * portfolio_values.at[date]) // ticker_prices.loc[date]
            shares.loc[date] = new_shares
            portfolio_values.at[date] = (new_shares * ticker_prices.loc[date]).sum()

    return shares, portfolio_values

def backtest_vect(ticker_prices, initial_capital, initial_weights, rebalance_frequency=None):
    """
    Vectorized backtesting simulation of a portfolio with optional rebalancing.

    This function simulates portfolio performance using vectorized operations for efficiency.

    Args:
        ticker_prices (pd.DataFrame): DataFrame containing ticker prices,
        with dates as index and tickers as columns.
        initial_capital (float): The initial capital for the portfolio.
        initial_weights (np.array): array containing initial weights for each ticker.
        rebalance_frequency (int, optional): Frequency of rebalancing (e.g., 30 for monthly rebalancing).
        If None, no rebalancing occurs.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - shares (pd.DataFrame): DataFrame showing the number of shares held of each ticker at each time step.
            - portfolio_values (pd.DataFrame): DataFrame showing the value of each ticker at each time step.
    """

    if rebalance_frequency == None:
        shares = pd.DataFrame(index=ticker_prices.index, columns=ticker_prices.columns, dtype=np.int64)
        shares.iloc[0] = ((initial_weights*initial_capital)//ticker_prices.iloc[0]).astype('int')
        shares = shares.ffill()
        portfolio_values = shares*ticker_prices
        return shares, portfolio_values

    # Identify rebalance index positions in the historical series DataFrame
    idx_rebalance = [i for i in range(len(ticker_prices)) if i % rebalance_frequency == 0]
    # Initialize shares value tracking
    shares = pd.DataFrame(0, index=ticker_prices.iloc[idx_rebalance].index,
                        columns=ticker_prices.columns, dtype=np.int64)
    shares.iloc[0] = ((initial_weights * initial_capital) // ticker_prices.iloc[idx_rebalance[0]]).astype('int')

    for t in range(1,len(idx_rebalance)):
        portfolio_val = (shares.iloc[t-1] * ticker_prices.iloc[idx_rebalance[t]]).sum()
        shares.iloc[t] = ((initial_weights * portfolio_val) // ticker_prices.iloc[idx_rebalance[t]]).astype('int')

    # create the daily shares
    shares = shares.reindex(ticker_prices.index, method='ffill')
    portfolio_values = shares*ticker_prices

    return shares, portfolio_values