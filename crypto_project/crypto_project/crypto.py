import os
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_FILE = "BTCUSDT-trades-2026-10-11.csv"

FEATURES = ["ret_1", "ret_5", "vol_20", "flow", "flow_5", "log_qty"]
LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}


def load_trades(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"CSV file not found: {filename}")

    print("Loading historical data...")

    df = pd.read_csv(filename, header=None, nrows=500000)
    if df.shape[1] != 7:
        raise ValueError("Expected a 7-column Binance trades CSV.")

    df.columns = [
        "id", "price", "quantity", "quote_qty",
        "time", "maker", "best_match"
    ]

    for col in ["price", "quantity", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price", "quantity", "time"])

    # Detect timestamp unit: seconds, milliseconds, or microseconds
    median_time = df["time"].median()
    unit = "us" if median_time > 1e14 else "ms" if median_time > 1e11 else "s"

    df["timestamp"] = pd.to_datetime(
        df["time"], unit=unit, errors="coerce", utc=True
    )

    maker = df["maker"].astype(str).str.lower().str.strip()
    df["side"] = np.where(maker == "true", "sell", "buy")

    df = df.dropna(subset=["timestamp"])
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]

    return df[["timestamp", "price", "quantity", "side"]].sort_values(
        "timestamp"
    )


def create_bars(df):
    df = df.copy()
    df["buy_qty"] = np.where(df["side"] == "buy", df["quantity"], 0.0)

    grouped = df.set_index("timestamp").resample("1s")

    bars = pd.DataFrame({
        "price": grouped["price"].last(),
        "qty": grouped["quantity"].sum(),
        "buy": grouped["buy_qty"].sum(),
        "trades": grouped["price"].count()
    })

    bars = bars[bars["trades"] > 0].reset_index()
    print(f"{len(bars)} bars loaded from {DATA_FILE}")

    if len(bars) < 100:
        raise ValueError("Not enough bars to train and test.")

    return bars


def create_features(df):
    df = df.copy()
    df["ret_1"] = df["price"].pct_change()
    df["ret_5"] = df["price"].pct_change(5)
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["flow"] = (
        2 * df["buy"] / df["qty"].replace(0, np.nan) - 1
    )
    df["flow_5"] = df["flow"].rolling(5).mean()
    df["log_qty"] = np.log1p(df["qty"])

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df


def train_models(df):
    df = df.copy()
    horizon = 10
    future_return = df["price"].shift(-horizon) / df["price"] - 1

    split = int(len(df) * 0.70)
    training_returns = future_return.iloc[:split].dropna()

    low = training_returns.quantile(0.25)
    high = training_returns.quantile(0.75)

    df["target"] = np.select(
        [future_return < low, future_return > high],
        [0, 2],
        default=1
    )
    df.loc[future_return.isna(), "target"] = np.nan

    valid = df.dropna(subset=FEATURES + ["target"]).copy()
    valid["target"] = valid["target"].astype(int)

    train = valid[valid.index < split]
    test = valid[valid.index >= split]

    if len(train) < 50 or len(test) < 10:
        raise ValueError("Not enough valid data to train and test.")

    if train["target"].nunique() < 2:
        raise ValueError("Training data has fewer than two target classes.")

    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    baseline = y_test.value_counts(normalize=True).max()

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10,
            class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8,
            min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"{name:<15} accuracy={accuracy:.3f}  baseline={baseline:.3f}")

    return models, valid


def main():
    try:
        trades = load_trades(DATA_FILE)
        bars = create_bars(trades)
        featured = create_features(bars)
        models, valid = train_models(featured)

        latest = featured.dropna(subset=FEATURES).tail(1)
        if latest.empty:
            raise ValueError("No valid features for prediction.")

        model = models["Random Forest"]
        probabilities = model.predict_proba(latest[FEATURES])[0]
        classes = model.classes_

        index = int(np.argmax(probabilities))
        signal = LABELS[int(classes[index])]
        confidence = probabilities[index]
        price = float(latest["price"].iloc[0])

        print(f"\nLive: {signal} (confidence {confidence:.0%}) at {price:,.2f}")

        if signal == "BUY":
            spend = 100.00
            coin = spend / price
            print(f"Paper BUY: spend {spend:.2f} USDT = {coin:.6f} coin")
        elif signal == "SELL":
            print("Paper SELL: sell signal generated")
        else:
            print("Paper HOLD: no trade placed")

    except Exception as error:
        print(f"Data or model error: {error}")


if __name__ == "__main__":
    main()