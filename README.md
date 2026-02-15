# 🏈 NFL Game Predictor

A simple machine learning project that predicts NFL game winners. Built this to learn how ML actually works with real sports data.

## What it does

Feed it two teams, and it tells you who's probably going to win. That's it. No fancy stuff.

```
🏈 SF @ KC
   KC: 62% chance to win
   SF: 38% chance to win
   🏆 Predicted Winner: KC
```

## How it works

1. Looks at 2025 season game data
2. Calculates each team's average points scored, points allowed, and home win rate
3. Uses logistic regression (basically fancy pattern matching) to learn what predicts wins
4. Spits out win probabilities for any matchup

## The features it uses

- How many points the home team usually scores at home
- How many points they usually give up
- Same stats for the away team
- The home team's win rate at home

Simple stuff, but it works surprisingly well.

## Try it yourself

```bash
# Make sure you have pandas and scikit-learn
pip install pandas scikit-learn

# Run it
python simple_nfl_predictor.py
```

Then add your own predictions at the bottom:

```python
predict_winner('KC', 'SF')    # Super Bowl rematch anyone?
predict_winner('BUF', 'MIA')  # AFC East rivalry
```

## Team codes

Use these 3-letter codes:

```
ARI ATL BAL BUF CAR CHI CIN CLE 
DAL DEN DET GB  HOU IND JAX KC
LA  LAC LV  MIA MIN NE  NO  NYG 
NYJ PHI PIT SEA SF  TB  TEN WAS
```

## Current accuracy

Around **60-65%** on test data. Not bad for a simple model! Vegas uses way more features and still doesn't hit 100%, so I'll take it.

## What I learned

- Machine learning isn't magic, it's just math finding patterns
- Home field advantage is real (the model picks it up)
- More data = better predictions
- Sometimes simple models are good enough

---

*Built while learning ML. Feel free to fork and improve it!*
