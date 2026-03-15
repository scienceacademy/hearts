# ML Feature Engineering Guide

Building a Hearts bot through feature engineering.

---

## Part 1: What You're Building

Your goal is to build a Hearts bot that beats RuleBot — a heuristic bot that plays a decent game of Hearts using hand-coded rules (pass high cards, duck to avoid taking points, dump the Queen of Spades when safe).

You won't write game logic or design a model architecture. Instead, you'll work on the critical link between the two: **feature engineering**. You decide what information from the game state to translate into numbers that a machine learning model can learn from.

The model (a Random Forest classifier) is fixed. The training pipeline is fixed. Your one job: edit `hearts/ml/features.py` to give the model better information. Better features → better decisions → more wins.

Think of it this way: the model sees a list of numbers, not cards. Your `StudentFeatureExtractor` is the translator. A good translator highlights what matters — "the Queen of Spades is still out there and you hold the King." A bad one buries the signal in noise — 200 numbers where most are zero.

---

## Part 2: ML Concepts You Need

### Classification

The model looks at a game state and picks one card to play. During training, it studies thousands of examples of what RuleBot chose in similar situations. Each example is a pair: (game state as numbers, card that was played). The model finds patterns — "when these numbers look like this, card 37 tends to get picked."

At play time, the model sees a new game state, converts it to numbers using your feature extractor, and predicts probabilities for each of the 52 cards. It then picks the highest-probability card that's actually legal to play.

### Features

A feature is a single number that describes something about the game state. For example:
- `1.0` if hearts have been broken, `0.0` if not — one feature
- Points taken by each player — four features (one per player)
- Whether each of the 52 cards is in your hand — fifty-two features (each 0 or 1)

The same game state encoded as different features produces a different model. If your features don't mention whether the Queen of Spades has been played, the model can't learn to care about it.

### Training and Testing

We split the training data: 80% for learning, 20% for testing. The model trains on the 80% and then we check how well it does on the 20% it has never seen.

- **Train accuracy**: how often the model picks the right card on data it trained on
- **Test accuracy**: how often it picks the right card on data it has *never seen*

If train accuracy is much higher than test accuracy, the model has **overfit** — it memorized the training examples instead of learning general patterns. This usually means you have too many features relative to how much useful signal they contain.

Don't expect perfect accuracy. Hearts has many valid plays in any situation, and RuleBot itself isn't optimal. A test accuracy of 75–85% is typical, and even modest accuracy improvements can produce a bot that wins more games.

### Feature Importance

After training, the Random Forest can report which features it relied on most. This is your primary debugging tool. If you add a feature and it shows up with high importance, the model found it useful. If it has near-zero importance, either the feature isn't informative or it's redundant with something already there.

---

## Part 3: The Workflow

Your workflow is a loop:

1. **Edit features** in `hearts/ml/features.py` by modifying `StudentFeatureExtractor`
2. **Train** the model:
   ```bash
   python scripts/train_model.py
   ```
3. **Evaluate** against RuleBot:
   ```bash
   python scripts/evaluate_model.py
   ```
4. Read the report, think about what to try next, repeat.

### First-Time Setup

Before your first training run, generate the training data (this takes a few minutes):

```bash
python scripts/generate_data.py --games 3000 --seed 42
```

This plays 3,000 games of 4 RuleBots against each other and records every decision. The data is saved to `data/output/play_decisions.jsonl`. More games = better training data, but 3,000 is a good balance between quality and generation speed.

Note: You can also generate games with other bots playing. Look at the `generate_data.py` script for details.

### Reading the Evaluation Report

The evaluation script prints three sections:

```
=== Training Metrics ===
Feature vector size:  223 features       ← how many numbers your extractor produces
Train accuracy:       82.4%              ← how well the model fits training data
Test accuracy:        81.8%              ← how well it generalizes (the one that matters)

=== Feature Importance (top 10) ===
 1. trick_position       0.0354          ← the model relies on knowing if you're leading vs following
 2. legal_3C             0.0320
 3. legal_4C             0.0289
...

=== Tournament Results (200 games) ===
Player          Win Rate    Avg Score
MLBot           13.0%       90.7            ← your bot's performance
RuleBot-1       28.5%       67.1
RuleBot-2       25.0%       70.5
RuleBot-3       33.5%       68.0
```

---

## Part 4: Understanding BasicFeatureExtractor

Before adding features, understand what's already there. `BasicFeatureExtractor` produces ~223 features in these groups:

### Hand (52 features)

A binary vector: `1` if the card is in your hand, `0` otherwise.

```python
hand_2C = 1.0   # you hold the 2 of clubs
hand_QS = 0.0   # you don't hold the queen of spades
...
```

This tells the model your options but doesn't directly say anything *about* those cards.

### Legal Plays (52 features)

A binary vector: `1` if the card is a legal play right now, `0` otherwise. This encodes the rules of Hearts (follow suit, first-trick restrictions, hearts leading) so the model knows which cards it's choosing between.

### Cards Played This Round (52 features)

A binary vector: `1` if the card has been played in a completed trick, `0` otherwise.

```python
played_QS = 1.0  # queen of spades already played — spades are safe now
played_AH = 0.0  # ace of hearts still out there
```

This is raw information — the model has to figure out *why* certain cards being gone matters.

### Current Trick (52 features)

A positional encoding: each card played in the current trick gets a value based on play order (0.25 for first, 0.5 for second, 0.75 for third). Cards not in the trick are 0.

This tells the model what's been played in this trick and in what order.

### Suit Counts (4 features)

Number of cards of each suit in your hand, normalized by 13. Gives the model direct access to suit lengths — short suits mean you're close to voiding a suit and being able to dump point cards.

### Trick Position (1 feature)

How many cards have been played in the current trick (0–3), normalized by 3. This is one of the most important features: leading (0.0) vs. playing last (1.0) is a huge strategic difference.

### Hearts Broken (1 feature)

`1.0` if hearts have been played, `0.0` otherwise. Until hearts are broken, you can't lead with a heart.

### Points Taken by Player (4 features)

Points accumulated by each player this round, in relative order: your points first, then left, across, right. The relative ordering lets the model learn "I have too many points" regardless of which seat you're in.

### Trick Number (1 feature)

Normalized 0.0–1.0. Trick 1 = 0.0, trick 13 = 1.0. Early-round play is different from late-round play.

### Cumulative Scores (4 features)

Game-level scores divided by 100, in relative order (your score first, then left, across, right). Tells the model who's close to losing.

### What's Missing

This is where your work begins. `BasicFeatureExtractor` **cannot** see:

- **Which suits you're void in.** If you can't follow suit, you can dump hearts or the Queen of Spades — this is a huge strategic advantage. The suit counts tell the model how many cards you have, but a binary "void yes/no" flag makes this explicit.
- **Whether dangerous cards (Q♠, K♠, A♠) have been played.** The entire spade game changes once the Queen of Spades is gone.
- **How many hearts are still out there.** Leading hearts when 10 are unplayed is very different from when only 2 remain.
- **What other players might be void in.** If someone didn't follow suit earlier, they're probably void — and can dump points on you.

---

## Part 5: Feature Engineering Exercises

Work through these in order. Each one adds information the model can use to make better decisions.

### Exercise 1: Suit Void Tracking (4 features)

**Motivation:** You hold the King of Diamonds and someone leads diamonds. Annoying but manageable — you might take the trick. Now imagine you're void in clubs and someone leads clubs. Suddenly you can throw away a heart or the Queen of Spades. Being void in a suit is a massive strategic advantage that the basic features don't encode.

**What to encode:** For each of the 4 suits, a binary flag: `1.0` if you have zero cards of that suit, `0.0` otherwise.

**Implementation hint:** Iterate over `Suit`, check `view.hand` for cards of each suit.

```python
# In extract():
for suit in Suit:
    cards_of_suit = [c for c in view.hand if c.suit == suit]
    features.append(1.0 if len(cards_of_suit) == 0 else 0.0)

# In feature_names():
for suit in Suit:
    names.append(f"void_{suit.name}")
```

**Sanity check:** After training, `void_HEARTS` and `void_SPADES` should appear in the feature importance list. Being void in those suits matters most strategically.

---

### Exercise 2: Dangerous Cards Remaining (3 features)

**Motivation:** You hold the King of Spades. Should you be worried? That depends entirely on whether the Queen of Spades has already been played. If she's gone, your King is safe. If she's still out there, playing your King could cost you 13 points.

**What to encode:** Three binary flags — one each for Q♠, K♠, and A♠. `1.0` if the card has **not** been played yet (still dangerous), `0.0` if it's been played (safe).

**Implementation hint:** Check `view.cards_played_this_round` for each card. Also check `view.trick_so_far` (the card might have been played in the current trick). Also check if it's in your own hand.

**Sanity check:** The Q♠ flag should rank high in importance — it's one of the most decision-relevant facts in Hearts.

---

### Exercise 3: Hearts Remaining (1 feature)

**Motivation:** Leading hearts is risky when 10 hearts are still unplayed — someone will take those points. But when only 1 or 2 hearts remain, leading one is almost free. The basic features know *whether* hearts have been broken, but not *how many* are left.

**What to encode:** Count of unplayed hearts, normalized (divide by 13 so the value is 0.0–1.0).

**Implementation hint:** Count hearts in `view.cards_played_this_round` and `view.trick_so_far`. Subtract from 13.

**Sanity check:** This feature should show moderate importance. It refines the binary `hearts_broken` flag with more granular information.

---

### Exercise 4: Opponent Void Inference (4–8 features)

**Motivation:** The player to your left didn't follow suit when clubs were led two tricks ago. That means they're void in clubs — so if you lead clubs, they can dump hearts or the Queen of Spades on you. Knowing who's void in what completely changes your lead strategy.

**What to encode:** For each player (or each opponent), track which suits they've failed to follow. For each of the 4 suits, a binary flag per opponent: `1.0` if that opponent has been observed not following that suit, `0.0` otherwise.

**Implementation hint:** This is the hardest exercise. You need to iterate over completed tricks in `view.cards_played_this_round`:
1. For each completed trick, find the lead suit (first card played).
2. For each subsequent card in that trick, check if it matches the lead suit.
3. If a player played a non-lead-suit card, they're void in the lead suit.

Start simple — maybe just 4 features (one per suit: "is *any* opponent void in this suit?") before trying per-opponent tracking.

**Sanity check:** These features are subtle but should show up with moderate importance, especially for the spade and club voids. If they don't show up at all, double-check your logic — make sure you're looking at the lead suit of each *completed* trick, not the current trick.

---

## Part 6: Interpreting Your Results

### Performance Targets

| Level | Win Rate | Avg Score | What It Means |
|-------|----------|-----------|---------------|
| RandomBot baseline | ~1% | ~104 | Playing randomly — loses badly |
| Basic features only | ~10% | ~93 | Model learned something, but still far behind RuleBot |
| Good features | ~15% | ~87 | Noticeably better — feature engineering is paying off |
| Great features | >18% | <85 | Approaching competitive play with RuleBot |

Note: RuleBots average ~25–30% win rate each and ~65–70 avg score. In Hearts, the bot with the most points loses, so lower avg score is better. A perfect RuleBot clone would win ~25% of the time (one of four equal players). Getting close to that with an ML model is a significant achievement.

### Common Issues

**Win rate dropped below ~1%**
You have a bug in your feature code. Most likely: `extract()` and `feature_names()` are returning different-length lists, a feature is returning NaN or infinity, or you're appending features in a different order than names.

**Train accuracy much higher than test accuracy**
The model is overfitting. You may have added too many features without enough signal. Try removing features that show zero importance — they add noise without value.

**New features don't appear in importance**
Either the features aren't informative (they don't vary enough or are redundant with existing features) or there's a bug (they're always returning the same value). Print some values during training to verify.

**Features added but win rate didn't change**
This can happen if the features are correct but don't affect the model's decisions in practice. The feature might be informative but redundant with other features the model already uses. Try removing some basic features and keeping only your new ones to see if they carry signal.

---

## Part 7: Extensions

For students who finish the exercises and want to explore further.

### Train a Pass Model

The training data includes `pass_decisions.jsonl` with every passing decision. You could:
1. Create a `PassFeatureExtractor` that extracts features from `PassView`
2. Modify `train.py` to train a pass model (predict which 3 cards to pass)
3. Wire it into `MLBot` to replace the RuleBot pass strategy

### Try Different Models

The pipeline uses `RandomForestClassifier`, but scikit-learn has many other classifiers. Copy `train.py` and try:
- `GradientBoostingClassifier` — often more accurate but slower to train
- `MLPClassifier` — a simple neural network
- Compare the same features across different models

### Mixed Training Data

The default training data is 4 RuleBots playing each other. What happens if you generate data with different populations?
- 2× RuleBot + 2× RandomBot
- 4× different rule strategies
- Your own trained MLBot playing against RuleBots (iterative improvement)

### Shoot-the-Moon Detection

Can you add features that help the model recognize when an opponent is attempting to shoot the moon? Look for a single player accumulating all the points. If detected early enough, the right play is to take one point card to block them.
