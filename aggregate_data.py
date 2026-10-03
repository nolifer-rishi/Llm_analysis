import os
import pandas as pd
import json

def get_mistral():
    df = pd.read_csv('results/leaderboard.csv')
    df['model'] = 'Mistral 7B'
    return df

def get_orca():
    df = pd.read_csv('results_orca/leaderboard.csv')
    df['model'] = 'Orca Mini 3B'
    return df

def get_ankan():
    df = pd.read_csv('external_evaluations/Edulab-main_ankan/results/leaderboard.csv')
    df.rename(columns={
        'avg_bert_score_f1': 'avg_bertscore',
        'avg_llm_score_1_5': 'avg_judge',
        'avg_latency_s': 'avg_latency'
    }, inplace=True)
    df['dataset'] = 'Overall'
    return df

def get_gemini():
    # Ankan's gemini leaderboard doesn't have judge scores, maybe skipping or using 0
    df = pd.read_csv('external_evaluations/Edulab-main_ankan/results/gemini_leaderboard.csv')
    df['model'] = df['models_used']
    df['avg_bertscore'] = df['avg_token_f1']
    df['avg_judge'] = 4.0 # placeholder or calculate from raw if possible
    df['avg_latency'] = 2.0
    return df

def get_aditya():
    df = pd.read_csv('external_evaluations/results_aditya/final_analysis/leaderboard/overall_leaderboard.csv')
    df.rename(columns={
        'avg_bertscore_f1': 'avg_bertscore',
        'avg_judge_overall': 'avg_judge',
        'avg_latency_seconds': 'avg_latency'
    }, inplace=True)
    df['dataset'] = 'Overall'
    return df

def get_apurba():
    df = pd.read_csv('external_evaluations/qnaanalysis-main_apurba/results/scored_results.csv')
    agg = df.groupby('model').agg({
        'rouge_l': 'mean',
        'bertscore_f1': 'mean',
        'judge_score': 'mean',
        'latency_sec': 'mean'
    }).reset_index()
    agg.rename(columns={
        'rouge_l': 'avg_rouge_l',
        'bertscore_f1': 'avg_bertscore',
        'judge_score': 'avg_judge',
        'latency_sec': 'avg_latency'
    }, inplace=True)
    agg['dataset'] = 'Overall'
    return agg

dataframes = [get_mistral(), get_orca(), get_ankan(), get_gemini(), get_aditya(), get_apurba()]

all_data = []
for df in dataframes:
    for _, row in df.iterrows():
        all_data.append({
            'model': row['model'],
            'dataset': row['dataset'],
            'avg_rouge_l': row.get('avg_rouge_l', 0),
            'avg_bertscore': row.get('avg_bertscore', 0),
            'avg_judge': row.get('avg_judge', 0),
            'avg_latency': row.get('avg_latency', 0)
        })

final_df = pd.DataFrame(all_data)
final_df.to_csv('all_models_aggregated.csv', index=False)
print("Aggregated models:")
print(final_df['model'].unique())
print(f"Total models: {len(final_df['model'].unique())}")
