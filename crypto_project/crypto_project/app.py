import os
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Crypto Signal Dashboard",
    page_icon="₿",
    layout="centered"
)

DATA_FILE = "BTCUSDT-trades-2026-10-11.csv"

FEATURES = [
    "ret_1", "ret_5", "vol_20",
    "flow", "flow_5", "log_qty"
]

LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename, header=None, nrows=500000)

    if df.shape[1] < 7:
        raise ValueError("The CSV must contain 7 columns.")

    df = df.iloc[:, :7].copy()
    df.columns = [
        "id", "price", "quantity", "quote_qty",
        "time", "maker", "best_match"
    ]

    for col in ["price", "quantity", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price", "quantity", "time"])

    median_time = df["time"].median()

    if median_time > 1e14:
        unit = "us"
    elif median_time > 1e11:
        unit = "ms"
    else:
        unit = "s"

    df["timestamp"] = pd.to_datetime(
        df["time"], unit=unit, utc=True, errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])
    df = df[(df["price"] > 0) & (df["quantity"] > 0)].copy()

    maker = df["maker"].astype(str).str.lower().str.strip()

    # Buyer-maker trades are classified as aggressive sells.
    df["side"] = maker.map({
        "true": "sell",
        "false": "buy"
    })

    df = df.dropna(subset=["side"]).sort_values("timestamp")

    df["buy_qty"] = df["quantity"].where(df["side"] == "buy", 0)
    df["sell_qty"] = df["quantity"].where(df["side"] == "sell", 0)

    grouped = df.set_index("timestamp").resample("1s")

    bars = pd.DataFrame({
        "price": grouped["price"].last(),
        "qty": grouped["quantity"].sum(),
        "buy": grouped["buy_qty"].sum(),
        "sell": grouped["sell_qty"].sum(),
        "trades": grouped["price"].count()
    })

    bars = bars[bars["trades"] > 0].reset_index()

    if len(bars) < 100:
        raise ValueError(
            f"Only {len(bars)} price bars were created. "
            "Check your CSV timestamps and data."
        )

    bars["ret_1"] = bars["price"].pct_change()
    bars["ret_5"] = bars["price"].pct_change(5)
    bars["vol_20"] = bars["ret_1"].rolling(20).std()

    total_qty = bars["buy"] + bars["sell"]

    bars["flow"] = (
        (bars["buy"] - bars["sell"]) /
        total_qty.replace(0, float("nan"))
    )

    bars["flow_5"] = bars["flow"].rolling(5).mean()
    bars["log_qty"] = __import__("numpy").log1p(bars["qty"])

    bars[FEATURES] = bars[FEATURES].replace(
        [float("inf"), -float("inf")], float("nan")
    )

    return bars


# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

def train_models(bars):
    import numpy as np

    data = bars.copy().reset_index(drop=True)

    horizon = 10
    future_return = data["price"].shift(-horizon) / data["price"] - 1

    split = int(len(data) * 0.70)

    low = future_return.iloc[:split].dropna().quantile(0.25)
    high = future_return.iloc[:split].dropna().quantile(0.75)

    data["target"] = np.select(
        [future_return < low, future_return > high],
        [0, 2],
        default=1
    )

    data.loc[future_return.isna(), "target"] = np.nan

    valid = data.dropna(subset=FEATURES + ["target"]).copy()
    valid["target"] = valid["target"].astype(int)

    train = valid[valid.index < split]
    test = valid[valid.index >= split]

    if len(train) < 50 or len(test) < 10:
        raise ValueError("Not enough valid data to train and test.")

    if train["target"].nunique() < 2:
        raise ValueError("Training data contains fewer than two classes.")

    X_train = train[FEATURES]
    y_train = train["target"]
    X_test = test[FEATURES]
    y_test = test["target"]

    baseline = y_test.value_counts(normalize=True).max()

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, predictions), 3),
            "Baseline": round(baseline, 3)
        })

        trained_models[name] = model

    return trained_models, pd.DataFrame(results), valid


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

st.title("Crypto signal: Decision Tree / Random Forest")
st.caption("Supervised learning on Binance historical data. Paper money only.")

if not os.path.exists(DATA_FILE):
    st.error(f"CSV not found: {DATA_FILE}")
    st.info("Place the CSV in the same folder as app.py.")
    st.stop()

try:
    bars = load_data(DATA_FILE)
except Exception as error:
    st.error(f"Could not load data: {error}")
    st.stop()

st.write(f"Loaded {len(bars):,} bars")

chart = px.line(
    bars,
    x="timestamp",
    y="price",
    title="BTCUSDT Historical Price",
    labels={"timestamp": "Time (UTC)", "price": "Price (USDT)"}
)

chart.update_traces(line_color="#ff4b4b", line_width=1.2)
chart.update_layout(
    height=350,
    margin=dict(l=10, r=10, t=50, b=10)
)

st.plotly_chart(chart, use_container_width=True)


# --------------------------------------------------
# TRAIN BUTTON
# --------------------------------------------------

if st.button("Train models", type="primary", use_container_width=True):
    with st.spinner("Training models..."):
        try:
            models, results, valid = train_models(bars)

            st.session_state["models"] = models
            st.session_state["results"] = results
            st.session_state["valid"] = valid

            st.success("Models trained successfully!")

        except Exception as error:
            st.error(f"Training failed: {error}")


# --------------------------------------------------
# MODEL RESULTS
# --------------------------------------------------

if "results" in st.session_state:
    st.subheader("Model comparison")
    st.dataframe(
        st.session_state["results"],
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Accuracy should beat the baseline to show improvement "
        "over the majority-class prediction."
    )

    st.subheader("Model for live signal")

    selected_model = st.selectbox(
        "Choose a model",
        ["Random Forest", "Decision Tree"]
    )

    if st.button("Get live signal", use_container_width=True):
        model = st.session_state["models"][selected_model]
        valid = st.session_state["valid"]

        latest = valid.dropna(subset=FEATURES).tail(1)

        if latest.empty:
            st.error("No valid features available.")
        else:
            probabilities = model.predict_proba(latest[FEATURES])[0]
            classes = model.classes_

            best = int(probabilities.argmax())
            signal = LABELS[int(classes[best])]
            confidence = float(probabilities[best])
            price = float(latest["price"].iloc[0])

            st.subheader("Signal")

            if signal == "BUY":
                st.success(signal)
            elif signal == "SELL":
                st.error(signal)
            else:
                st.info(signal)

            st.write(
                f"Confidence: {confidence:.0%} | "
                f"Price: {price:,.2f} USDT"
            )

            st.caption(
                "This is a model prediction from historical data, "
                "not a guaranteed profitable or truly live trading signal."
            )

            st.subheader("Paper trading")

            if signal == "BUY":
                spend = 100.0
                coin = spend / price
                st.info(
                    f"Paper BUY: spend {spend:.2f} USDT = "
                    f"{coin:.6f} BTC"
                )
            elif signal == "SELL":
                st.info("Paper SELL signal generated. No real order placed.")
            else:
                st.info("Paper HOLD: no trade placed.")

else:
    st.info("Click 'Train models' to train and compare the models.")