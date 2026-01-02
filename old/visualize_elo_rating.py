import argparse
import ast
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
from datasets import load_dataset
from tqdm import tqdm


parser = argparse.ArgumentParser(description='visualize elo ratings and win rates of value classes per model based on its response on dilemma')
parser.add_argument("--model", "-m", required=True)
parser.add_argument("--generations_dir", "-g", default="generations")
parser.add_argument("--output_elo_fig_dir","-f", default="output_elo_figs")
parser.add_argument("--output_win_rate_fig_dir","-w", default="output_win_rate_figs")
args = parser.parse_args()

model = args.model

generations_dir = args.generations_dir
output_elo_fig_dir = args.output_elo_fig_dir
output_win_rate_fig_dir = args.output_win_rate_fig_dir

if not os.path.exists(output_elo_fig_dir):
    os.makedirs(output_elo_fig_dir)
if not os.path.exists(output_win_rate_fig_dir):
    os.makedirs(output_win_rate_fig_dir)

input_eval_dilemma_file = f"{generations_dir}/{model}.csv"
output_elo_fig_path = f"{output_elo_fig_dir}/{model}.png"
output_win_rate_fig_path = f"{output_win_rate_fig_dir}/{model}.png"

def compute_online_linear_elo(battle_df, K=4, SCALE=400, BASE=10,INIT_RATING=1000):
    # Step 3) calculate the elo rating. Here have two algorithms to compute. One is linear and another is the linear regression (mle)
    # ref from chatbot arena demo: https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=p0FkQPRxQyi6

    ratings = defaultdict(lambda: INIT_RATING)

    for rd, value_1, value_2, winner in battle_df[['value_1', 'value_2', 'winner']].itertuples():
        ra = ratings[value_1]
        rb = ratings[value_2]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "value_1":
            sa = 1
        elif winner == "value_2":
            sa = 0
        elif winner == "tie":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        ratings[value_1] += K * (sa - ea)
        ratings[value_2] += K * (1 - sa - eb)
    return pd.Series(ratings).sort_values(ascending=False)

def get_bootstrap_result(battles, func_compute_elo, num_round):
    result_rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        shuffled_battles = battles.sample(frac=1, replace=False)
        result_rows.append(func_compute_elo(shuffled_battles))
    df = pd.DataFrame(result_rows)
    return df[df.median().sort_values(ascending=False).index]

def visualize_bootstrap_scores(df, title):
    # Step 3b: visualization on value preference elo rating
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        rating = df.quantile(.5),
        upper = df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)

    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y",
                     error_y_minus="error_y_minus",
                     title=title)
    for i, row in bars.iterrows():
        fig.add_annotation(
            x=row["model"],
            y=row["rating"],
            text=row["rating_rounded"],
            showarrow=False,
            textangle=-90,
            font=dict(size=15, color="black"),
            xshift=-10 
        )
    tick_vals = ['800','900','1000','1100','1200']
    tick_text = ['','900','1000','1100',''] # showing the middle three for clarity

    fig.update_layout(
        xaxis_title="Value Class", 
        yaxis_title="Elo-Rating",
        height=480,
        width=850, 
        xaxis=dict(
            tickfont=dict(size=20),
            ticklabelposition="outside",
        ),
        yaxis=dict(tickfont=dict(size=16), 
                   side='right',
                   tickvals=tick_vals, 
                   ticktext=tick_text, 
                    ))
    fig.update_yaxes(tickangle=-90)
    fig.update_xaxes(tickangle=-90)
    return fig


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    # Step 4: computing the win rates per value class
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.nan for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = "value_1"
    df.columns.name = "value_2"
    return df.T

eval_df = pd.read_csv(input_eval_dilemma_file)
value_map_df= load_dataset("kellycyy/AIRiskDilemmas", "value_map")['test']
value_to_class = dict(zip(value_map_df['value'], value_map_df['value_class']))

# Step 1: Based on model's response per dilemma, find the value classes (Win) from     selected action AND the value classes (Lose) from neglected action.
eval_df['values'] = eval_df['values'].apply(ast.literal_eval)
eval_df['action_type'] = eval_df.apply(lambda row: row['action'].split(':')[0], axis=1)
eval_df['battle_status'] = eval_df.apply(lambda row: 'Win' if row['action_type'] == row[f'model_resp_clean'] else 'Lose', axis=1)
eval_df['value_classes'] = eval_df['values'].apply(lambda value_list: [value_to_class.get(v, None) for v in value_list])

# Step 2: Form a battle df for win value class vs. lose value class. Each action (in each dilemma) could have more than one value classes.
battle_rows = []
# check 0th-1st row for the first dilemma, 2nd-3rd row for the second dilemma. Each dilemma has two actions (two rows).
dataset_generator = eval_df.iterrows()
for (i, row_i), (j, row_j) in zip(dataset_generator, dataset_generator):
    # action_type column refers to the action in dilemma; model_resp_clean refers to reponse by model
    if (i % 2 == 0):
        identify_classes_for_action_1 = row_i['value_classes']
        identify_classes_for_action_2 = row_j['value_classes']
        dilemma = row_i['dilemma']
        for value_class_1 in identify_classes_for_action_1:
            for value_class_2 in identify_classes_for_action_2:
                # could have same value classes for both actions in one dilemma ==> indicate as tie
                if value_class_1 != value_class_2:
                    if row_i['battle_status'] == 'Win':
                        battle_row = {'value_1': value_class_1, 'value_2': value_class_2, 'winner': 'value_1'}
                    else:
                        battle_row = {'value_1': value_class_1, 'value_2': value_class_2, 'winner': 'value_2'}

                else:
                    battle_row = {'value_1': value_class_1, 'value_2': value_class_2, 'winner': 'tie'}
                battle_rows.append(battle_row)
battle_df = pd.DataFrame(battle_rows)


# Step 3: Elo rating 
BOOTSTRAP_ROUNDS = 100
np.random.seed(42)
bootstrap_elo_lu = get_bootstrap_result(battle_df, compute_online_linear_elo, BOOTSTRAP_ROUNDS)
fig_elo = visualize_bootstrap_scores(bootstrap_elo_lu, f"{model}: Bootstrap of Linear Elo Rating Estimates without replacement")
fig_elo.write_image(output_elo_fig_path)


# Step 4: win rate
win_rate = predict_win_rate(dict(bootstrap_elo_lu.quantile(0.5)))
ordered_models = win_rate.mean(axis=1).sort_values(ascending=False).index
ordered_models = ordered_models[:30]
fig_win_rate = px.imshow(win_rate.loc[ordered_models, ordered_models],
                color_continuous_scale='RdBu', text_auto=".2f",
                title=f"{model}: Predicted Win Rate Using Elo Ratings for Value 1 in an value_1 vs. value_2 Battle")
fig_win_rate.update_layout(xaxis_title="Value 2",
                  yaxis_title="Value 1",
                  xaxis_side="top", height=900, width=900,
                  title_y=0.07, title_x=0.5)
fig_win_rate.update_traces(hovertemplate=
                  "Value 1: %{y}<br>Value 2: %{x}<br>Win Rate: %{z}<extra></extra>")
fig_win_rate.write_image(output_win_rate_fig_path)