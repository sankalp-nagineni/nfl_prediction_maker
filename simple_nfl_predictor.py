"""
Simple NFL Win Predictor - Beginner Version
Uses basic stats to predict which team will win
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step 1: Load the games data
print("Loading NFL games data...")
games = pd.read_csv('/Users/sankalp/ml/nfl_games.csv')

# Filter to only 2025 season for most current predictions
games = games[games['season'] == 2025]
print(f"Using season: 2025 only")

# Step 2: Create simple features
# We'll use: point differential, home/away, and rest days
games['home_won'] = (games['home_score'] > games['away_score']).astype(int)
games['point_diff'] = games['home_score'] - games['away_score']
games['total_points'] = games['home_score'] + games['away_score']

# Filter to only games with scores (remove future games)
games = games[games['home_score'].notna()]
print(f"Total games: {len(games)}")

# Step 3: Calculate each team's average performance
# Group by home team to get home stats
home_stats = games.groupby('home_team').agg({
    'home_score': 'mean',
    'away_score': 'mean', 
    'home_won': 'mean'
}).rename(columns={
    'home_score': 'avg_home_points_for',
    'away_score': 'avg_home_points_against',
    'home_won': 'home_win_rate'
})

# Group by away team to get away stats
away_stats = games.groupby('away_team').agg({
    'away_score': 'mean',
    'home_score': 'mean'
}).rename(columns={
    'away_score': 'avg_away_points_for',
    'home_score': 'avg_away_points_against'
})

# Combine stats
team_stats = home_stats.join(away_stats)
print(f"\nTeam stats calculated for {len(team_stats)} teams")
print(team_stats.head())

# Step 4: Prepare training data
# For each game, get the stats of both teams
X_list = []
y_list = []

for _, game in games.iterrows():
    home = game['home_team']
    away = game['away_team']
    
    if home in team_stats.index and away in team_stats.index:
        # Features: home team's offense, defense, and away team's offense, defense
        features = {
            'home_avg_points': team_stats.loc[home, 'avg_home_points_for'],
            'home_avg_allowed': team_stats.loc[home, 'avg_home_points_against'],
            'away_avg_points': team_stats.loc[away, 'avg_away_points_for'],
            'away_avg_allowed': team_stats.loc[away, 'avg_away_points_against'],
            'home_win_rate': team_stats.loc[home, 'home_win_rate'],
        }
        X_list.append(features)
        y_list.append(game['home_won'])

X = pd.DataFrame(X_list)
y = pd.Series(y_list)

print(f"\nFeatures shape: {X.shape}")
print(f"Sample features:\n{X.head()}")

# Step 5: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Check accuracy
accuracy = model.score(X_test, y_test)
print(f"\n✅ Model Accuracy: {accuracy:.1%}")

# Step 8: Show feature importance
print("\n Feature Importance:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"  {feature}: {coef:.3f}")


# Team abbreviations for reference
TEAMS = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 
         'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
         'LA', 'LAC', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 
         'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS']

# Step 9: Make a prediction
def predict_winner(home_team, away_team):
    """Predict who wins: home or away team"""
    home_team = home_team.upper()
    away_team = away_team.upper()
    

    #features are the stats of the home and away teams,we'll use this to predict the winner
    features = pd.DataFrame([{
        'home_avg_points': team_stats.loc[home_team, 'avg_home_points_for'],
        'home_avg_allowed': team_stats.loc[home_team, 'avg_home_points_against'],
        'away_avg_points': team_stats.loc[away_team, 'avg_away_points_for'],
        'away_avg_allowed': team_stats.loc[away_team, 'avg_away_points_against'],
        'home_win_rate': team_stats.loc[home_team, 'home_win_rate'],
    }])
    
    prob = model.predict_proba(features)[0]
    home_prob = prob[1]
    away_prob = prob[0]
    
    print(f"\n {away_team} @ {home_team}")
    print(f"   {home_team}: {home_prob:.0%} chance to win")
    print(f"   {away_team}: {away_prob:.0%} chance to win")
    
    if home_prob > away_prob:
        print(f"   🏆 Predicted Winner: {home_team}")
    else:
        print(f"   🏆 Predicted Winner: {away_team}")




predict_winner('KC', 'SF')
predict_winner('PHI', 'DAL')
predict_winner('BUF', 'MIA')
predict_winner('NE', 'SEA')
predict_winner('BUF', 'NE')