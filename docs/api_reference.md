# API Reference

Reference for the types you'll use when writing features: `PlayerView`, `PassView`, `Card`, `Suit`, and the `FeatureExtractor` interface.

---

## `PlayerView` — What Your Bot Sees

`PlayerView` is the snapshot of game state passed to your bot's `play_card()` method. Every feature you write in `StudentFeatureExtractor.extract()` reads from a `PlayerView` instance.

Defined in `hearts/state.py`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `hand` | `set[Card]` | Cards currently in the player's hand |
| `player_index` | `int` (0–3) | Which seat this player occupies |
| `trick_so_far` | `list[tuple[int, Card]]` | Cards played in the current trick so far — each entry is `(player_index, card)` in play order. Empty if you're leading. |
| `lead_suit` | `Suit` or `None` | The suit of the first card played in this trick. `None` if you're leading. |
| `hearts_broken` | `bool` | `True` if any heart has been played in a previous trick this round |
| `is_first_trick` | `bool` | `True` if this is the first trick of the round (special rules apply) |
| `cards_played_this_round` | `list[tuple[int, int, Card]]` | All cards from completed tricks — each entry is `(trick_number, player_index, card)`. Trick numbers are 1-based. Does **not** include the current in-progress trick. |
| `tricks_won_by_player` | `list[int]` | How many tricks each player has won so far (length 4) |
| `points_taken_by_player` | `list[int]` | Points accumulated by each player this round (length 4) |
| `pass_direction` | `PassDirection` | Which direction cards were passed this round (`LEFT`, `RIGHT`, `ACROSS`, or `NONE`) |
| `cards_received_in_pass` | `list[Card]` | The 3 cards this player received during passing (empty list during No Pass rounds) |
| `legal_plays` | `list[Card]` | Pre-computed list of cards you're allowed to play right now |
| `cumulative_scores` | `list[int]` | Game-level scores entering this round (length 4). These are the total points across all previous rounds. |

### Usage Examples

```python
# Check if you're void in a suit
clubs_in_hand = [c for c in view.hand if c.suit == Suit.CLUBS]
is_void_clubs = len(clubs_in_hand) == 0

# Check if Q♠ has been played already
qs_played = any(
    card.suit == Suit.SPADES and card.rank == 12
    for _, _, card in view.cards_played_this_round
)

# Your position in the trick (0 = leading, 3 = last)
position = len(view.trick_so_far)

# Count unplayed hearts
hearts_played = sum(
    1 for _, _, card in view.cards_played_this_round
    if card.suit == Suit.HEARTS
)
# Also count hearts in the current trick
hearts_played += sum(
    1 for _, card in view.trick_so_far
    if card.suit == Suit.HEARTS
)
hearts_remaining = 13 - hearts_played
```

---

## `PassView` — What Your Bot Sees During Passing

`PassView` is passed to your bot's `pass_cards()` method at the start of rounds that have a pass phase.

Defined in `hearts/state.py`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `hand` | `set[Card]` | All 13 cards in the player's hand (before passing) |
| `player_index` | `int` (0–3) | Which seat this player occupies |
| `pass_direction` | `PassDirection` | `LEFT`, `RIGHT`, or `ACROSS` (never `NONE` — no PassView is created during No Pass rounds) |
| `cumulative_scores` | `list[int]` | Game-level scores entering this round (length 4) |
| `round_number` | `int` | Which round of the game (1-based) |

---

## `Card` — A Playing Card

Defined in `hearts/types.py`. Cards are frozen dataclasses — immutable and hashable.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `suit` | `Suit` | One of `CLUBS`, `DIAMONDS`, `SPADES`, `HEARTS` |
| `rank` | `int` | 2–14, where 11=J, 12=Q, 13=K, 14=A |

### Key Functions

```python
from hearts.types import Card, Suit, card_points, is_heart, is_queen_of_spades

card = Card(Suit.SPADES, 12)  # Queen of Spades
str(card)                      # "Q♠"
card_points(card)              # 13
is_queen_of_spades(card)       # True
is_heart(card)                 # False

heart = Card(Suit.HEARTS, 7)
card_points(heart)             # 1
is_heart(heart)                # True
```

---

## `Suit` — Card Suits

Defined in `hearts/types.py`. `Suit` is an `IntEnum` with values:

| Suit | Value | Symbol |
|------|-------|--------|
| `Suit.CLUBS` | 0 | ♣ |
| `Suit.DIAMONDS` | 1 | ♦ |
| `Suit.SPADES` | 2 | ♠ |
| `Suit.HEARTS` | 3 | ♥ |

You can iterate over all suits with `for suit in Suit:`.

---

## Card Index Mapping (0–51)

When cards appear in training data (JSONL files), they're stored as integers:

```
index = suit.value * 13 + (rank - 2)
```

| Index | Card | Index | Card | Index | Card | Index | Card |
|-------|------|-------|------|-------|------|-------|------|
| 0     | 2♣   | 13    | 2♦   | 26    | 2♠   | 39    | 2♥   |
| 1     | 3♣   | 14    | 3♦   | 27    | 3♠   | 40    | 3♥   |
| ...   | ...  | ...   | ...  | ...   | ...  | ...   | ...  |
| 10    | Q♣   | 23    | Q♦   | 36    | A♠   | 49    | Q♥   |
| 11    | K♣   | 24    | K♦   | 37    | Q♠   | 50    | K♥   |
| 12    | A♣   | 25    | A♦   | 38    | K♠   | 51    | A♥   |

Convert between representations with:

```python
from hearts.data.schema import card_to_index, index_to_card

card_to_index(Card(Suit.SPADES, 12))  # 37 (Q♠)
index_to_card(37)                      # Card(SPADES, 12)
```

---

## `FeatureExtractor` — The Interface You Implement

Defined in `hearts/ml/features.py`.

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `extract` | `(view: PlayerView) -> np.ndarray` | Convert a game state into a numeric vector. Called once per play decision during training and during gameplay. |
| `feature_names` | `() -> list[str]` | Return human-readable names for each dimension. Used in the evaluation report to show feature importance. |

### Contract

1. `extract()` must always return an array of the **same length**, regardless of the game state.
2. `feature_names()` must return a list with **exactly as many names** as the length of the array from `extract()`.
3. Feature values should be numeric (no NaN or inf).
4. Normalize large values when possible (e.g., divide scores by 100, divide trick number by 12).

### Example: Adding a Feature

```python
def extract(self, view: PlayerView) -> np.ndarray:
    features, _ = _extract_basic_features(view)

    # === YOUR FEATURES BELOW ===
    # Am I void in each suit?
    for suit in Suit:
        cards_of_suit = [c for c in view.hand if c.suit == suit]
        features.append(1.0 if len(cards_of_suit) == 0 else 0.0)

    return np.array(features, dtype=np.float32)

def feature_names(self) -> list[str]:
    # ... get base names ...
    names = [...]

    # === YOUR FEATURE NAMES BELOW ===
    for suit in Suit:
        names.append(f"void_{suit.name}")

    return names
```

---

## `TrickResult` — Completed Trick Summary

Passed to `Bot.on_trick_complete()`. Defined in `hearts/state.py`.

| Field | Type | Description |
|-------|------|-------------|
| `cards_played` | `list[tuple[int, Card]]` | All 4 cards played, as `(player_index, card)` pairs in play order |
| `lead_suit` | `Suit` | The suit that was led |
| `winner` | `int` | Player index who won the trick |
| `points` | `int` | Total points in the trick (0–14) |
| `trick_number` | `int` | Which trick (1–13) |

## `RoundResult` — Completed Round Summary

Passed to `Bot.on_round_complete()`. Defined in `hearts/state.py`.

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `list[int]` | Points taken by each player this round (length 4) |
| `tricks` | `list[TrickResult]` | All 13 tricks |
| `shoot_the_moon` | `bool` | Whether someone shot the moon |
| `moon_shooter` | `int` or `None` | Player index of the moon shooter |
| `pass_direction` | `PassDirection` | The pass direction used this round |
| `hands_dealt` | `list[set[Card]]` | Original hands before passing |

## `PassDirection` — Pass Direction Enum

Defined in `hearts/state.py`.

| Value | Meaning |
|-------|---------|
| `PassDirection.LEFT` | Pass 3 cards to the player on your left |
| `PassDirection.RIGHT` | Pass 3 cards to the player on your right |
| `PassDirection.ACROSS` | Pass 3 cards to the player across from you |
| `PassDirection.NONE` | No passing this round |

Rotation order: LEFT → RIGHT → ACROSS → NONE → LEFT → ...
