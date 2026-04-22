import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Financial Copilot", layout="centered")
st.title("💰 AI Financial Copilot (Hackathon Ready)")

# ---------------- SESSION INIT ----------------
defaults = {
    "analyzed": False,
    "results": None,
    "ai_advice": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    else:
        data = pd.read_csv("data.csv")

        data["total_expense"] = data["food"] + data["rent"] + data["travel"] + data["others"]
        data = data[data["total_expense"] <= data["income"]]
        data["savings"] = data["income"] - data["total_expense"]

        X = data[["income", "food", "rent", "travel", "others"]]
        y = data["savings"]

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)

        joblib.dump(model, "model.pkl")
        return model

model = load_model()

# ---------------- INPUT FORM ----------------
st.subheader("User Profile & Inputs")

with st.form("finance_form"):
    user_type = st.selectbox(
        "User Type",
        ["General User", "Gig Worker (Driver/Delivery Partner)"]
    )

    income = st.number_input("Income (₹)", min_value=0)
    food = st.number_input("Food (₹)", min_value=0)
    rent = st.number_input("Rent (₹)", min_value=0)
    travel = st.number_input("Travel (₹)", min_value=0)
    others = st.number_input("Others (₹)", min_value=0)

    submitted = st.form_submit_button("Analyze")

# ---------------- ANALYSIS ----------------
if submitted:
    if income == 0:
        st.error("Income must be greater than 0")
    else:
        total = food + rent + travel + others
        savings = income - total

        ml_pred = model.predict(np.array([[income, food, rent, travel, others]]))[0]

        # Hybrid prediction
        final_pred = 0.7 * savings + 0.3 * ml_pred
        final_pred = max(final_pred, 0)  # Safety

        expense_ratio = total / income

        # Context
        if user_type == "Gig Worker (Driver/Delivery Partner)":
            context_note = "Income varies daily. Higher savings buffer needed."
            target_ratio = 0.25
        else:
            context_note = "Stable income assumed."
            target_ratio = 0.2

        # Improved Score
        score = 100

        if expense_ratio > 0.85:
            score -= 25
        elif expense_ratio > 0.7:
            score -= 15
        elif expense_ratio > 0.6:
            score -= 10

        if savings < 0:
            score -= 40

        score = max(score, 0)

        # Store results
        st.session_state.results = {
            "income": income,
            "food": food,
            "rent": rent,
            "travel": travel,
            "others": others,
            "total": total,
            "savings": savings,
            "final_pred": final_pred,
            "expense_ratio": expense_ratio,
            "score": score,
            "context_note": context_note,
            "target_ratio": target_ratio,
            "user_type": user_type
        }

        st.session_state.analyzed = True

# ---------------- DISPLAY ----------------
if st.session_state.analyzed and st.session_state.results:

    r = st.session_state.results

    # Summary
    st.subheader("📊 Financial Summary")
    st.write("Total Expenses:", r["total"])
    st.write("Actual Savings:", r["savings"])
    st.write("AI Predicted Savings:", int(r["final_pred"]))

    # Metrics
    st.subheader("📈 Financial Metrics")
    st.write("Expense Ratio:", round(r["expense_ratio"], 2))

    # Score
    st.subheader("💯 Financial Health Score")
    st.metric("Score", f"{r['score']}/100")
    st.progress(r["score"] / 100)

    # Context
    st.subheader("🧠 Context")
    st.info(r["context_note"])

    # Daily Mode
    st.subheader("📅 Daily Survival Mode")
    daily_budget = r["income"] / 30
    daily_spent = r["total"] / 30

    st.write("Daily Budget:", int(daily_budget))
    st.write("Daily Spending:", int(daily_spent))

    if daily_spent > daily_budget:
        st.error("Overspending daily")
    else:
        st.success("Under control")

    # Future Projection
    st.subheader("🔮 3-Month Projection")
    future = r["final_pred"] * 3
    st.write("Projected Savings:", int(future))

    # Goal Planning
    st.subheader("🎯 Goal Planning")
    goal = st.number_input("Enter Savings Goal (₹)", min_value=0)

    if goal > 0:
        months_actual = goal / max(r["savings"], 1)
        months_ai = goal / max(r["final_pred"], 1)

        st.write(f"Months (Current Savings): {round(months_actual, 1)}")
        st.write(f"Months (AI Prediction): {round(months_ai, 1)}")

    # Savings Gap
    st.subheader("📉 Savings Gap Analysis")
    target_savings = r["income"] * r["target_ratio"]
    gap = target_savings - r["savings"]

    st.write("Target Monthly Savings:", int(target_savings))
    st.write("Savings Gap:", int(gap))

    # Chart
    st.subheader("📊 Expense Breakdown")
    chart_data = {
        "Food": r["food"],
        "Rent": r["rent"],
        "Travel": r["travel"],
        "Others": r["others"]
    }
    st.bar_chart(chart_data)

    # Alerts
    st.subheader("⚠️ Alerts")
    alerts = []

    if r["total"] > r["income"]:
        alerts.append("Expenses exceed income")

    if r["expense_ratio"] > 0.85:
        alerts.append("Very high spending")

    if r["savings"] < r["target_ratio"] * r["income"]:
        alerts.append("Savings below target")

    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success("No major risks")

    # ---------------- AI ADVICE ----------------
    st.subheader("🤖 AI Advice")

    if st.button("Generate AI Advice"):

        prompt = f"""
You are an expert financial advisor.

User Type: {r['user_type']}
Income: ₹{r['income']}
Expenses: ₹{r['total']}
Savings: ₹{r['final_pred']}
Expense Ratio: {r['expense_ratio']:.2f}

Give:
1. Financial health status (1 line)
2. Top 3 problems
3. Exact ₹ fixes
4. Weekly plan
5. One smart saving hack
"""

        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            res = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            st.session_state.ai_advice = res.choices[0].message.content

        except Exception as e:
            st.error(f"API Error: {e}")

            # Fallback advice
            advice = []

            if r["expense_ratio"] > 0.8:
                advice.append("Reduce non-essential expenses immediately.")

            if r["food"] > 0.3 * r["income"]:
                advice.append("Cut food delivery and outside eating.")

            if r["rent"] > 0.4 * r["income"]:
                advice.append("Reduce rent via sharing or relocation.")

            if r["savings"] < 0:
                advice.append("Critical: You are overspending. Fix urgently.")

            if not advice:
                advice.append("Finances stable. Start SIP or investments.")

            st.session_state.ai_advice = "\n\n".join(advice)

    if st.session_state.ai_advice:
        st.write(st.session_state.ai_advice)

    # Reset
    if st.button("🔄 Reset"):
        st.session_state.analyzed = False
        st.session_state.results = None
        st.session_state.ai_advice = None
        st.rerun()
