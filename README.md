# Sentimental-Analysis
Predictive analysis platform for sentimental analysis 


## ✨ Features

- **Batch sentiment analysis** using `transformers.pipeline("sentiment-analysis")`
- **Aspect keyword tagging** (e.g., service, food, price, parking…)
- **Dark UI theme** with custom CSS
- **Metrics**: total entries, net sentiment, unique aspects
- **Visuals**: bar chart of sentiment distribution + aspect breakdown table
- **Quick single-text check** in the sidebar
- **Sample dataset** included for instant demo

---

## 🧭 App Flow (High-Level)

flowchart LR
<img width="2174" height="1140" alt="image" src="https://github.com/user-attachments/assets/6b179156-43c0-4e3e-b096-f96e5cbadf8d" />

Usage

Choose data

Use the sample dataset (enabled by default), or

Upload your own CSV with a text column (optional created_at, source columns supported).

Configure aspects
In the sidebar, edit the comma-separated keywords (e.g., service, food, price, parking, staff, ambience, delivery).
The app tags each text with any matching aspect; if none match, it assigns general.

Run analysis
Click “Run Sentiment Analysis”. You’ll get:

Metrics: Total Entries, Net Sentiment (%), Unique Aspects

Bar chart: Positive/Negative/Neutral counts (labels come from the model)

Aspect table: Counts per aspect by sentiment

Quick single-text check
In the sidebar, paste a single sentence and click “Check sentiment” for an instant label and score

