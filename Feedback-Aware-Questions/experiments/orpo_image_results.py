import json
import plotly.graph_objects as go

filenames = {
    f"Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_feedback_aware_orpo_smart_duplicates.json": "FA ORPO (ours)",
    f"Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_feedback_aware_sft_smart_duplicates.json": "FA SFT (ours)",
    f"Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_baseline_single_answer_orpo_smart_duplicates.json": "SSG ORPO",
    f"Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_baseline_single_answer_sft_smart_duplicates.json": "SSG SFT",
}

fig_precision = go.Figure()

for filename in filenames:
    print(f"Processing {filename}")
    dataset = json.load(open(filename, 'r'))        

    precision_at_k = []

    for data in dataset:
        if len(data) == 0:
            continue

        precision_at_k_dict = {}

        for i in range(1, 26):
            precision_at_k_dict[i] = len([1 for x in data[:i] if x['label'] == 'GOOD']) / i

        precision_at_k.append(precision_at_k_dict)

    mean_precision_at_k_dict = {}

    for i in range(1, 26):
        mean_precision_at_k_dict[i] = sum([x[i] for x in precision_at_k]) / len(precision_at_k)

    fig_precision.add_trace(go.Scatter(x=list(mean_precision_at_k_dict.keys()), y=list(mean_precision_at_k_dict.values()), mode='lines+markers', name=filenames[filename], line=dict(width=7), marker=dict(size=15)))


fig_precision.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="K",
    yaxis_title="Precision",
    width=1200,
    height=600,
    xaxis=dict(tickfont=dict(size=22)),
    yaxis=dict(tickfont=dict(size=22)),
    xaxis_title_font=dict(size=26),
    yaxis_title_font=dict(size=26),
    legend=dict(
        x=0.01,  # Position the legend inside, adjust as needed
        y=0.01,
        traceorder="normal",
        bgcolor="rgba(255, 255, 255, 0.5)",  # Optional: Add background transparency
        bordercolor="Black",
        borderwidth=2,
        font=dict(size=26),
    ),
)

fig_precision.write_image("Feedback-Aware-Questions/experiments/questions_orpo_results.png")
