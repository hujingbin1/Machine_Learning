import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_features(df):
    # Price-related features
    df['midpx_ma'] = df['midpx'].rolling(window=12).mean()
    df['midpx_std'] = df['midpx'].rolling(window=12).std()
    df['lastpx_ma'] = df['lastpx'].rolling(window=12).mean()
    df['lastpx_std'] = df['lastpx'].rolling(window=12).std()

    df['log_return_midpx'] = np.log(df['midpx']).diff()
    df['log_return_lastpx'] = np.log(df['lastpx']).diff()
    df['realized_volatility_midpx'] = df.groupby('symbol')['log_return_midpx'].apply(lambda x: np.sqrt(np.sum(x ** 2)))
    df['realized_volatility_lastpx'] = df.groupby('symbol')['log_return_lastpx'].apply(
        lambda x: np.sqrt(np.sum(x ** 2)))

    # Trading-related features
    df['tradeBuyQty_ma'] = df['tradeBuyQty'].rolling(window=12).mean()
    df['tradeBuyQty_std'] = df['tradeBuyQty'].rolling(window=12).std()
    df['tradeSellQty_ma'] = df['tradeSellQty'].rolling(window=12).mean()
    df['tradeSellQty_std'] = df['tradeSellQty'].rolling(window=12).std()

    df['tradeBuyTurnover_ma'] = df['tradeBuyTurnover'].rolling(window=12).mean()
    df['tradeBuyTurnover_std'] = df['tradeBuyTurnover'].rolling(window=12).std()
    df['tradeSellTurnover_ma'] = df['tradeSellTurnover'].rolling(window=12).mean()
    df['tradeSellTurnover_std'] = df['tradeSellTurnover'].rolling(window=12).std()

    # Market strength and liquidity features
    df['bid_ask_spread'] = df['ask0'] - df['bid0']
    df['bid_ask_spread_ma'] = df['bid_ask_spread'].rolling(window=12).mean()
    df['bid_ask_spread_std'] = df['bid_ask_spread'].rolling(window=12).std()

    df['volume_imbalance'] = abs((df['bsize0'] + df['bsize0_4']) - (df['asize0'] + df['asize0_4']))
    df['volume_imbalance_ma'] = df['volume_imbalance'].rolling(window=12).mean()
    df['volume_imbalance_std'] = df['volume_imbalance'].rolling(window=12).std()

    # Time features
    df['hour'] = pd.to_datetime(df['interval'], unit='ms').dt.hour
    df['minute'] = pd.to_datetime(df['interval'], unit='ms').dt.minute

    # Aggregated features over different time windows
    for time in [12, 24, 48]:
        df[f'midpx_ma_{time}'] = df['midpx'].rolling(window=time).mean()
        df[f'midpx_std_{time}'] = df['midpx'].rolling(window=time).std()
        df[f'lastpx_ma_{time}'] = df['lastpx'].rolling(window=time).mean()
        df[f'lastpx_std_{time}'] = df['lastpx'].rolling(window=time).std()

        df[f'tradeBuyQty_ma_{time}'] = df['tradeBuyQty'].rolling(window=time).mean()
        df[f'tradeBuyQty_std_{time}'] = df['tradeBuyQty'].rolling(window=time).std()
        df[f'tradeSellQty_ma_{time}'] = df['tradeSellQty'].rolling(window=time).mean()
        df[f'tradeSellQty_std_{time}'] = df['tradeSellQty'].rolling(window=time).std()

        df[f'bid_ask_spread_ma_{time}'] = df['bid_ask_spread'].rolling(window=time).mean()
        df[f'bid_ask_spread_std_{time}'] = df['bid_ask_spread'].rolling(window=time).std()
        df[f'volume_imbalance_ma_{time}'] = df['volume_imbalance'].rolling(window=time).mean()
        df[f'volume_imbalance_std_{time}'] = df['volume_imbalance'].rolling(window=time).std()

    return df


# Example usage
df = pd.read_csv('data1.csv')
df = add_features(df)

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop(columns=['symbol', 'interval']))
df_scaled = pd.DataFrame(df_scaled, columns=df.drop(columns=['symbol', 'interval']).columns)


# Prepare LSTM input
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]['fret12']  # assuming fret12 is the target
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 12
X, y = create_sequences(df_scaled, seq_length)

# Example of creating LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, df_scaled.shape[1])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
