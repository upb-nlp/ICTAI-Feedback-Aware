import json
import plotly.graph_objects as go

filenames = {
    f"Feedback-Aware-Keywords/experiments/test_keywords_feedback_aware_orpo_smart_duplicates.json": "FA ORPO (ours)",
    f"Feedback-Aware-Keywords/experiments/test_keywords_feedback_aware_sft_smart_duplicates.json": "FA SFT (ours)",
    f"Feedback-Aware-Keywords/experiments/test_keywords_baseline_single_keyword_orpo_smart_duplicates.json": "SSG ORPO",
    f"Feedback-Aware-Keywords/experiments/test_keywords_baseline_single_keyword_sft_smart_duplicates.json": "SSG SFT",
}
ground_truth = json.load(open("Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_test.json", 'r'))

lens = [len(gt['keywords']) for gt in ground_truth]
print(f"95% din abstracte au mai putin de X keywords. X = {sorted(lens)[int(len(lens) * 0.95)]}")


fig_precision = go.Figure()

for filename in filenames:
    print(f"Processing {filename}")
    dataset = json.load(open(filename, 'r'))        

    # Precision at K
    precision_at_k = []
    for data in dataset:
        if len(data) == 0:
            continue

        precision_at_k_dict = {}
        for i in range(1, 11):
            precision_at_k_dict[i] = len([1 for x in data[:i] if x['label'] == 'GOOD']) / i
        precision_at_k.append(precision_at_k_dict)

    mean_precision_at_k_dict = {}
    for i in range(1, 11):
        mean_precision_at_k_dict[i] = sum([x[i] for x in precision_at_k]) / len(precision_at_k)

    # Recall at K
    recall_at_k = []
    for data, gt in zip(dataset, ground_truth):
        if len(data) == 0:
            continue

        recall_at_k_dict = {}
        for i in range(1, 11):
            recall_at_k_dict[i] = len([1 for x in data[:i] if x['label'] == 'GOOD']) / len(gt['keywords'])
        recall_at_k.append(recall_at_k_dict)

    mean_recall_at_k_dict = {}
    for i in range(1, 11):
        mean_recall_at_k_dict[i] = sum([x[i] for x in recall_at_k]) / len(recall_at_k)

    # F1 at K
    mean_f1_at_k_dict = {i: 2 * (mean_precision_at_k_dict[i] * mean_recall_at_k_dict[i]) / (mean_precision_at_k_dict[i] + mean_recall_at_k_dict[i]) for i in range(1, 11)}


    fig_precision.add_trace(go.Scatter(x=list(mean_f1_at_k_dict.keys()), y=list(mean_f1_at_k_dict.values()), mode='lines+markers', name=filenames[filename], line=dict(width=7), marker=dict(size=15)))

fig_precision.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="K",
    yaxis_title="F1",
    width=1200,
    height=600,
    xaxis=dict(tickfont=dict(size=22)),
    yaxis=dict(tickfont=dict(size=22)),
    xaxis_title_font=dict(size=26),
    yaxis_title_font=dict(size=26),
    legend=dict(
        x=0.01,  # Position the legend inside, adjust as needed
        y=0.98,
        traceorder="normal",
        bgcolor="rgba(255, 255, 255, 0.5)",  # Optional: Add background transparency
        bordercolor="Black",
        borderwidth=2,
        font=dict(size=26),
    ),
)
fig_precision.write_image("Feedback-Aware-Keywords/experiments/keywords_orpo_results.png")