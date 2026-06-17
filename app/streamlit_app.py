import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prédiction des rendements boursiers", layout="wide")
st.title("Prédiction des rendements boursiers basé sur les informations")

results_path = "data/processed/results.csv"
if not os.path.exists(results_path):
    st.warning("Lance d'abord `python -m src.main` pour générer les résultats.")
    st.stop()

results = pd.read_csv(results_path)
metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc", "total_return", "excess_return", "sharpe", "max_drawdown"]
available_metrics = [col for col in metric_cols if col in results.columns]

files = [f for f in os.listdir("data/processed") if f.startswith("backtest_") and f.endswith(".csv")]
if not files:
    st.warning("Aucun fichier de backtest trouvé.")
    st.stop()

left, right = st.columns([2, 1])

with right:
    tickers = sorted(results["ticker"].dropna().unique()) if "ticker" in results else []
    selected_ticker = st.selectbox("Ticker", tickers) if tickers else None
    sort_metric = st.selectbox("Métrique de classement", available_metrics, index=available_metrics.index("sharpe") if "sharpe" in available_metrics else 0)

filtered = results if selected_ticker is None else results[results["ticker"] == selected_ticker]
leaderboard = filtered.sort_values(sort_metric, ascending=False).reset_index(drop=True)

with left:
    st.subheader("Classement des modèles")
    st.dataframe(leaderboard, width="stretch")

best = leaderboard.iloc[0]
cols = st.columns(4)
cols[0].metric("Meilleur modèle", best["model"])
cols[1].metric("Expérience", best["experiment"])
cols[2].metric("Sharpe", f"{best.get('sharpe', 0):.2f}")
cols[3].metric("Rendement total", f"{best.get('total_return', 0):.1%}")

model_file = st.selectbox("Choisir un backtest", sorted(files))
bt = pd.read_csv(f"data/processed/{model_file}")

st.subheader("Equity curve")
fig, ax = plt.subplots()
ax.plot(pd.to_datetime(bt["date"]), bt["market_equity"], label="Buy & Hold")
ax.plot(pd.to_datetime(bt["date"]), bt["strategy_equity"], label="Strategy")
ax.legend()
ax.set_title("Performance cumulée")
ax.set_xlabel("Date")
ax.set_ylabel("Capital normalisé")
st.pyplot(fig)

euronext_path = "data/processed/euronext_microstructure.csv"
if not os.path.exists(euronext_path):
    euronext_path = "data/processed/euronext_microstructure_20240104.csv"
if os.path.exists(euronext_path):
    st.divider()
    st.subheader("Données Euronext")

    euronext = pd.read_csv(euronext_path)
    euronext["label"] = (
        euronext["symbol"].fillna(euronext["isin"]).astype(str)
        + " - "
        + euronext["name"].fillna("").astype(str)
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Instruments", f"{euronext['isin'].nunique():,}")
    metric_cols[1].metric("Dates", f"{pd.to_datetime(euronext['date']).nunique():,}")
    metric_cols[2].metric("Trades", f"{euronext['n_trades'].sum():,.0f}")
    metric_cols[3].metric("Valeur echangée", f"{euronext['traded_value'].sum():,.0f}")

    selected_instrument = st.selectbox(
        "Instrument Euronext",
        euronext.sort_values("traded_value", ascending=False)["label"].tolist(),
    )
    instrument = euronext[euronext["label"] == selected_instrument].iloc[0]
    st.dataframe(
        pd.DataFrame([instrument.drop(labels=["label"])]),
        width="stretch",
    )

    top = euronext.sort_values("traded_value", ascending=False).head(20)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(top["symbol"].fillna(top["isin"]), top["traded_value"])
    ax2.set_title("Top 20 par valeur echangée")
    ax2.set_xlabel("Instrument")
    ax2.set_ylabel("Valeur echangée")
    ax2.tick_params(axis="x", rotation=70)
    st.pyplot(fig2)

euronext_prediction_path = "data/processed/euronext_prediction_results.csv"
euronext_status_path = "data/processed/euronext_prediction_status.csv"
euronext_predictions_path = "data/processed/euronext_predictions.csv"
euronext_backtest_path = "data/processed/euronext_backtest.csv"
euronext_importance_path = "data/processed/euronext_feature_importance.csv"
if os.path.exists(euronext_prediction_path):
    st.divider()
    st.subheader("Pipeline Prédiction Euronext")

    if os.path.exists(euronext_status_path):
        euronext_status = pd.read_csv(euronext_status_path).iloc[0]
        status_cols = st.columns(4)
        status_cols[0].metric("Mode", euronext_status["mode"])
        status_cols[1].metric("Dates", f"{euronext_status['n_dates']:,.0f}")
        status_cols[2].metric("Instruments", f"{euronext_status['n_instruments']:,.0f}")
        status_cols[3].metric("Taux positif", f"{euronext_status['target_positive_rate']:.1%}")
        st.info(euronext_status["note"])
        if "news_rows" in euronext_status:
            news_cols = st.columns(5)
            news_cols[0].metric("News Euronext", f"{euronext_status['news_rows']:,.0f}")
            news_cols[1].metric("News reliées", f"{euronext_status['matched_news_rows']:,.0f}")
            news_cols[2].metric("ISIN avec sentiment", f"{euronext_status['sentiment_instruments']:,.0f}")
            news_cols[3].metric("Lignes ML sentiment", f"{euronext_status.get('dataset_sentiment_rows', 0):,.0f}")
            news_cols[4].metric("ISIN ML sentiment", f"{euronext_status.get('dataset_sentiment_instruments', 0):,.0f}")

    euronext_results = pd.read_csv(euronext_prediction_path)
    if not euronext_results.empty:
        st.dataframe(euronext_results, width="stretch")

        plot_cols = st.columns(2)
        with plot_cols[0]:
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            labels = euronext_results["experiment"] + "\n" + euronext_results["model"]
            x = range(len(labels))
            ax3.bar(x, euronext_results["balanced_accuracy"], label="Balanced accuracy")
            ax3.plot(x, euronext_results["auc"], marker="o", color="black", label="AUC")
            ax3.set_xticks(list(x))
            ax3.set_xticklabels(labels, rotation=35, ha="right")
            ax3.set_ylim(0, 1)
            ax3.set_title("Performance des modèles Euronext")
            ax3.legend()
            st.pyplot(fig3)

        with plot_cols[1]:
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            ax4.bar(labels, euronext_results["f1"])
            ax4.set_ylim(0, 1)
            ax4.set_title("F1-score par modèle")
            ax4.tick_params(axis="x", rotation=35)
            st.pyplot(fig4)

    if os.path.exists(euronext_backtest_path):
        euronext_backtest = pd.read_csv(euronext_backtest_path)
        if not euronext_backtest.empty:
            st.subheader("Mini-backtest Euronext")
            backtest_choice = st.selectbox(
                "Stratégie Euronext",
                sorted((euronext_backtest["experiment"] + " / " + euronext_backtest["model"]).unique()),
            )
            bt_experiment, bt_model = backtest_choice.split(" / ")
            bt_view = euronext_backtest[
                (euronext_backtest["experiment"] == bt_experiment)
                & (euronext_backtest["model"] == bt_model)
            ].copy()
            fig_bt, ax_bt = plt.subplots(figsize=(9, 4))
            ax_bt.plot(pd.to_datetime(bt_view["date"]), bt_view["long_equity"], marker="o", label="Long top probas")
            ax_bt.plot(pd.to_datetime(bt_view["date"]), bt_view["long_short_equity"], marker="o", label="Long/short")
            ax_bt.plot(pd.to_datetime(bt_view["date"]), bt_view["benchmark_equity"], marker="o", label="Benchmark égal-pondéré")
            ax_bt.set_title("Equity curve du mini-backtest Euronext")
            ax_bt.set_xlabel("Date")
            ax_bt.set_ylabel("Capital normalisé")
            ax_bt.legend()
            st.pyplot(fig_bt)

    if os.path.exists(euronext_importance_path):
        euronext_importance = pd.read_csv(euronext_importance_path)
        if not euronext_importance.empty:
            st.subheader("Importance des variables Euronext")
            importance_choice = st.selectbox(
                "Importance modèle",
                sorted((euronext_importance["experiment"] + " / " + euronext_importance["model"]).unique()),
            )
            imp_experiment, imp_model = importance_choice.split(" / ")
            imp = euronext_importance[
                (euronext_importance["experiment"] == imp_experiment)
                & (euronext_importance["model"] == imp_model)
            ].sort_values("importance", ascending=True)
            fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
            ax_imp.barh(imp["feature"], imp["importance"])
            ax_imp.set_title("Variables les plus importantes")
            ax_imp.set_xlabel("Importance normalisée")
            st.pyplot(fig_imp)

    if os.path.exists(euronext_predictions_path):
        euronext_predictions = pd.read_csv(euronext_predictions_path)
        if not euronext_predictions.empty:
            selected_prediction_run = st.selectbox(
                "Prédictions Euronext",
                sorted((euronext_predictions["experiment"] + " / " + euronext_predictions["model"]).unique()),
            )
            pred_experiment, selected_model = selected_prediction_run.split(" / ")
            model_predictions = euronext_predictions[
                (euronext_predictions["experiment"] == pred_experiment)
                & (euronext_predictions["model"] == selected_model)
            ].sort_values("probability_positive", ascending=False)

            pred_cols = st.columns(2)
            with pred_cols[0]:
                fig5, ax5 = plt.subplots(figsize=(8, 4))
                ax5.hist(model_predictions["probability_positive"], bins=20, edgecolor="white")
                ax5.axvline(0.5, color="black", linestyle="--", linewidth=1)
                ax5.set_title("Distribution des probabilités de hausse")
                ax5.set_xlabel("Probabilité prédite")
                ax5.set_ylabel("Nombre d'instruments")
                st.pyplot(fig5)

            with pred_cols[1]:
                top_pred = model_predictions.head(15).sort_values("probability_positive")
                labels = top_pred["symbol"].fillna(top_pred["isin"]).astype(str)
                fig6, ax6 = plt.subplots(figsize=(8, 5))
                ax6.barh(labels, top_pred["probability_positive"])
                ax6.set_xlim(0, 1)
                ax6.set_title("Top probabilités de hausse")
                ax6.set_xlabel("Probabilité prédite")
                st.pyplot(fig6)

            st.dataframe(model_predictions.head(25), width="stretch")

st.divider()
st.subheader("Conclusion")
conclusion_cols = st.columns(2)
with conclusion_cols[0]:
    st.markdown(
        """
        **Pipeline Yahoo US**

        Le pipeline principal compare les modèles avec et sans sentiment sur des tickers Yahoo.
        Il reste le plus adapté pour mesurer l'apport du sentiment sur une série longue.
        """
    )
with conclusion_cols[1]:
    st.markdown(
        """
        **Pipeline Euronext**

        Le pipeline Euronext utilise 4 journées de données haute fréquence agrégées.
        Les résultats sont utiles pour une preuve de concept, mais doivent être interprétés avec prudence.
        """
    )

st.markdown(
    """
    **Limites et prochaines étapes**

    - Les données Euronext couvrent seulement 4 jours.
    - Le sentiment Euronext dépend du fichier optionnel `data/raw/euronext_news.csv`.
    - La comparaison `microstructure_only` vs `microstructure_plus_sentiment` devient vraiment informative dès que des news Euronext datées et reliées aux ISIN sont disponibles.
    - Le mini-backtest illustre la transformation des probabilités en stratégie, mais il reste court à cause de la fenêtre temporelle.
    """
)
