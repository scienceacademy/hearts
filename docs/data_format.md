# Data Format Reference

This document describes the JSON schemas for the training data files produced by `scripts/generate_data.py`.

---

## Card Index Mapping

Cards are represented as integers 0–51:

```
index = suit_ordinal * 13 + (rank - 2)
```

| Suit     | Ordinal | Index Range |
|----------|---------|-------------|
| CLUBS    | 0       | 0–12        |
| DIAMONDS | 1       | 13–25       |
| SPADES   | 2       | 26–38       |
| HEARTS   | 3       | 39–51       |

Ranks: 2=2, 3=3, ..., 10=10, 11=J, 12=Q, 13=K, 14=A.

**Common cards:**

| Card | Index |
|------|-------|
| 2♣   | 0     |
| A♣   | 12    |
| 2♦   | 13    |
| Q♠   | 36    |
| K♠   | 37    |
| A♠   | 38    |
| 2♥   | 39    |
| A♥   | 51    |

---

## Play Decisions (`play_decisions.jsonl`)

Each line is a JSON object representing one card-play decision. This is the primary training data file.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `player_index` | `int` (0–3) | Which seat this player occupies |
| `trick_number` | `int` (1–13) | Current trick number within the round |
| `hand` | `list[int]` | Card indices currently in the player's hand (sorted) |
| `legal_plays` | `list[int]` | Card indices the player is allowed to play (sorted) |
| `trick_so_far` | `list[[int, int]]` | Cards already played in the current trick — each entry is `[player_index, card_index]` in play order |
| `lead_suit` | `string` or `null` | Name of the lead suit (`"CLUBS"`, `"DIAMONDS"`, `"SPADES"`, `"HEARTS"`), or `null` if this player is leading |
| `hearts_broken` | `bool` | Whether hearts have been played in this round |
| `is_first_trick` | `bool` | Whether this is the first trick of the round |
| `cards_played_this_round` | `list[list[[int, int]]]` | Completed tricks — outer list is tricks (index = trick number - 1), inner list is 4 `[player_index, card_index]` pairs in play order. Does **not** include the current in-progress trick. |
| `tricks_won_by_player` | `list[int]` | Count of tricks won by each player so far (length 4) |
| `points_taken_by_player` | `list[int]` | Points accumulated by each player so far (length 4) |
| `pass_direction` | `string` | `"LEFT"`, `"RIGHT"`, `"ACROSS"`, or `"NONE"` |
| `cards_received_in_pass` | `list[int]` | Card indices received during passing (3 cards, or `[]` during No Pass rounds) |
| `cumulative_scores` | `list[int]` | Game-level scores entering this round (length 4) |
| `round_number` | `int` | Which round of the game (1-based) |
| **`card_played`** | **`int`** | **Label — the card index that was played** |
| `points_this_trick` | `int` | Outcome — total points in the completed trick |
| `round_score` | `int` | Outcome — this player's total points for the round |
| `game_rank` | `int` (1–4) | Outcome — final ranking (1 = best/lowest score) |

### Example Record

```json
{
  "player_index": 2,
  "trick_number": 4,
  "hand": [0, 3, 7, 15, 22, 28, 31, 40, 44, 51],
  "legal_plays": [0, 3, 7],
  "trick_so_far": [[0, 19], [1, 23]],
  "lead_suit": "DIAMONDS",
  "hearts_broken": true,
  "is_first_trick": false,
  "cards_played_this_round": [
    [[0, 0], [1, 12], [2, 25], [3, 38]],
    [[3, 39], [0, 11], [1, 50], [2, 24]],
    [[1, 19], [3, 48], [0, 36], [2, 10]]
  ],
  "tricks_won_by_player": [1, 1, 1, 0],
  "points_taken_by_player": [0, 1, 0, 0],
  "pass_direction": "LEFT",
  "cards_received_in_pass": [3, 15, 44],
  "cumulative_scores": [22, 45, 18, 31],
  "round_number": 4,

  "card_played": 7,

  "points_this_trick": 0,
  "round_score": 3,
  "game_rank": 1
}
```

**Reading this example:** Player 2 is in trick 4, holding 10 cards. Two players have already played into the trick (player 0 played card 19 and player 1 played card 23). The lead suit is diamonds, so player 2 must follow suit if possible. Three previous tricks have been completed. Player 2 chose to play card 7. After the round ended, player 2 scored 3 points and finished the game with the best rank.

---

## Pass Decisions (`pass_decisions.jsonl`)

Each line is a JSON object representing one card-passing decision. This file is smaller (only 3 of every 4 rounds involve passing) and is used for the optional extension of training a pass model.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `player_index` | `int` (0–3) | Which seat this player occupies |
| `hand` | `list[int]` | All 13 card indices in the player's hand before passing (sorted) |
| `pass_direction` | `string` | `"LEFT"`, `"RIGHT"`, or `"ACROSS"` (never `"NONE"`) |
| `cumulative_scores` | `list[int]` | Game-level scores entering this round (length 4) |
| `round_number` | `int` | Which round of the game (1-based) |
| **`cards_passed`** | **`list[int]`** | **Label — the 3 card indices that were passed** |
| `round_score` | `int` | Outcome — this player's total points for the round |
| `game_rank` | `int` (1–4) | Outcome — final ranking (1 = best/lowest score) |

### Example Record

```json
{
  "player_index": 2,
  "hand": [0, 3, 7, 12, 15, 22, 28, 31, 40, 44, 47, 49, 51],
  "pass_direction": "LEFT",
  "cumulative_scores": [22, 45, 18, 31],
  "round_number": 4,

  "cards_passed": [47, 49, 51],

  "round_score": 3,
  "game_rank": 1
}
```

---

## Notes

- **Outcome annotations** (`points_this_trick`, `round_score`, `game_rank`) are added after the game completes, not during play. They reflect what actually happened, not what the bot predicted.
- **`round_score`** values across all 4 players in a round always sum to 26 (or 78 if someone shot the moon — the shooter scores 0 and the other three each score 26).
- **`game_rank`** uses 1 = lowest final score (winner), 4 = highest. Ties share the better rank.
- Records are written in play order, so consecutive lines often represent the same trick from different players' perspectives.
