import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Define standard colors
COLORS = sns.color_palette("Set2", 10)

def main():
    df = pd.read_csv('all_models_aggregated.csv')
    
    # Merge Gemini models
    df['model'] = df['model'].replace({'gemini-3.6-flash': 'Google Gemini', 'gemini-3.8-flash': 'Google Gemini'})
    
    # We want average across all datasets per model for a fair comparison
    agg_df = df.groupby('model').agg({
        'avg_rouge_l': 'mean',
        'avg_bertscore': 'mean',
        'avg_judge': 'mean',
        'avg_latency': 'mean'
    }).reset_index()

    # Drop any extra if we have more than 9 models. Currently we should have 9:
    # Mistral 7B, Orca Mini 3B, qwen2.5:3b, Google Gemini, gemma2:2b, openai/gpt-oss-120b, 
    # qwen2.5:1.5b, deepseek-r1:1.5b, llama3.2:3b
    print(f"Models to plot: {agg_df['model'].unique()}, Total: {len(agg_df)}")

    os.makedirs('graphs', exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Bar Chart for 3 Metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('9-Model Comparison: Core Metrics', fontsize=16, fontweight='bold', y=1.05)
    
    metrics = [('avg_rouge_l', 'ROUGE-L F1'), ('avg_bertscore', 'BERTScore F1'), ('avg_judge', 'LLM Judge Score (1-5)')]
    
    for i, (col, title) in enumerate(metrics):
        sorted_df = agg_df.sort_values(by=col, ascending=False)
        sns.barplot(data=sorted_df, x=col, y='model', ax=axes[i], palette='viridis')
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('')
        if i > 0:
            axes[i].set_yticks([])
    plt.tight_layout()
    plt.savefig('graphs/comparison_9models_metrics_bars.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Radar Chart
    categories = ['ROUGE-L', 'BERTScore', 'Judge (Normalized)']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, weight='bold')

    for idx, row in agg_df.iterrows():
        # Normalize judge to 0-1
        j_norm = (row['avg_judge'] - 1) / 4.0
        values = [row['avg_rouge_l'], row['avg_bertscore'], j_norm]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['model'], alpha=0.7)
        ax.fill(angles, values, alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('9-Model Multi-Metric Radar Footprint', size=16, weight='bold', y=1.1)
    plt.savefig('graphs/comparison_9models_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Heatmap
    heatmap_df = agg_df.set_index('model')[['avg_rouge_l', 'avg_bertscore', 'avg_judge']]
    heatmap_df['avg_judge'] = heatmap_df['avg_judge'] / 5.0 # normalize for color scale
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_df, annot=agg_df.set_index('model')[['avg_rouge_l', 'avg_bertscore', 'avg_judge']], 
                fmt=".3f", cmap="YlGnBu", linewidths=.5)
    plt.title('9-Model Score Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Model', fontsize=12)
    plt.xticks([0.5, 1.5, 2.5], ['ROUGE-L', 'BERTScore', 'Judge Score'], fontsize=11)
    plt.tight_layout()
    plt.savefig('graphs/comparison_9models_heatmap.png', dpi=300)
    plt.close()

    # 4. Latency Bar Chart
    plt.figure(figsize=(10, 6))
    sorted_df = agg_df.sort_values(by='avg_latency', ascending=True)
    sns.barplot(data=sorted_df, x='avg_latency', y='model', palette='magma')
    plt.title('Generation Latency Comparison (Average seconds)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Average Latency (s)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    for index, value in enumerate(sorted_df['avg_latency']):
        plt.text(value + 0.1, index, f"{value:.1f}s", va='center')
    plt.tight_layout()
    plt.savefig('graphs/comparison_9models_latency.png', dpi=300)
    plt.close()

    # 5. Scatter Plot (BERTScore vs Judge)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=agg_df, x='avg_bertscore', y='avg_judge', hue='model', s=200, palette='tab10')
    plt.title('BERTScore vs. Judge Score Alignment', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('BERTScore F1', fontsize=12)
    plt.ylabel('Judge Score (1-5)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('graphs/comparison_9models_scatter.png', dpi=300)
    plt.close()

    # 6. Combined Score Rank
    agg_df['combined_score'] = (agg_df['avg_rouge_l'] + agg_df['avg_bertscore'] + ((agg_df['avg_judge']-1)/4)) / 3
    sorted_df = agg_df.sort_values(by='combined_score', ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sorted_df, x='model', y='combined_score', palette='coolwarm')
    plt.title('Overall Model Ranking (Combined Score)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Combined Score (0-1)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('graphs/comparison_9models_ranking.png', dpi=300)
    plt.close()

    print("Successfully generated 6 comparison graphs in 'graphs/' folder.")

if __name__ == '__main__':
    main()
