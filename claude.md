# Hearts Bot ML Project — Full Plan

## Game Overview (Hearts Rules Reference)

Hearts is a 4-player trick-taking card game where the goal is to **avoid points**. Hearts are worth 1 point each, and the Queen of Spades is worth 13. The player with the **lowest** score wins. The exception is "shooting the moon" — if one player takes all 26 points in a round, they score 0 and everyone else gets 26. (This project uses the "add 26 to others" variant exclusively — the shooter always scores 0 and the other three players each receive 26 points. There is no option for the shooter to subtract from their own score.)

**Tie-breaking:** If two or more players share the lowest final score, they share the victory (all are considered winners). No additional rounds are played and no other tiebreaker is used.

**Reproducibility:** All sources of randomness (deck shuffling, bot decisions) must accept an optional `random.Random` instance (or seed) to enable deterministic, reproducible results for testing and ML data generation. When no RNG is provided, use the default `random` module.

---

## Tier 1: Game Engine & Bot Framework

### Step 0: Project Setup

Create project scaffolding:

- **Python version**: Requires Python 3.11+
- **`pyproject.toml`**: project metadata, dependencies, and tool configuration
  - Core dependencies: none (Tier 1 is pure Python stdlib)
  - ML dependencies (optional install group `[ml]`): `numpy`, `scikit-learn`, `matplotlib`
  - Dev dependencies (optional install group `[dev]`): `pytest`, `ruff`
- **`.gitignore`**: Python defaults, `.venv/`, `data/output/`, `models/`, `__pycache__/`
- **`hearts/__init__.py`**: package init
- Install in editable mode: `pip install -e ".[dev,ml]"`

---

### Step 1: Core Data Types & Card Representation

Create `hearts/types.py`:

- `Suit` enum: CLUBS, DIAMONDS, SPADES, HEARTS (with ordering)
- `Card` dataclass: suit + rank (2–14 where 11=J, 12=Q, 13=K, 14=A)
- `Deck` class: standard 52-card deck, shuffle, deal into 4 hands of 13
- Helper functions: `card_points(card) -> int`, `is_heart(card) -> bool`, `is_queen_of_spades(card) -> bool`
- `Card` should have clean `__str__` (e.g., "Q♠", "10♥") and be hashable/comparable

**Tests:** Card creation, point values, deck completeness, dealing produces 4 hands of 13 with no duplicates.

---

### Step 2: Game State & Trick Logic

Create `hearts/state.py`:

- `TrickState` dataclass:
  - `cards_played`: list of (player_index, Card) tuples
  - `lead_suit`: Suit
  - `winner() -> int`: determines trick winner (highest card of lead suit)

- `RoundState` dataclass (the state of one 13-trick round):
  - `hands`: list of 4 sets of Cards (each player's current hand)
  - `current_trick`: TrickState
  - `cards_taken`: list of 4 lists of Cards (cards won per player — the raw card lists)
  - `hearts_broken`: bool
  - `lead_player`: int
  - `scores`: per-round point totals (computed from cards_taken)

- `GameState` dataclass (multi-round game):
  - `cumulative_scores`: list of 4 ints
  - `round_history`: list of completed RoundStates
  - `current_round`: RoundState
  - `game_over`: bool (score >= 100 threshold)
  - `pass_direction`: cycles through Left, Right, Across, No Pass

- `PassView` dataclass (what bots see during the passing phase):
  - `hand`: the player's 13 cards
  - `player_index`: which seat (0–3)
  - `pass_direction`: Left, Right, or Across (never constructed during No Pass rounds)
  - `cumulative_scores`: list of 4 ints (game-level scores entering this round)
  - `round_number`: int

- `TrickResult` dataclass (summary of a completed trick):
  - `cards_played`: list of (player_index, Card) tuples in play order
  - `lead_suit`: Suit
  - `winner`: int (player index who won the trick)
  - `points`: int (total points in this trick)
  - `trick_number`: int (1–13)

- `RoundResult` dataclass (summary of a completed round):
  - `scores`: list of 4 ints (points taken this round per player)
  - `tricks`: list of TrickResult (all 13 tricks)
  - `shoot_the_moon`: bool (whether someone shot the moon)
  - `moon_shooter`: int or None (player index, if applicable)
  - `pass_direction`: the pass direction used this round
  - `hands_dealt`: list of 4 sets of Cards (original hands before passing)

**Tests:** Trick winner determination for all edge cases, hearts broken tracking, score computation, shoot-the-moon detection.

---

### Step 3: Rules Engine (Legal Move Validation)

Create `hearts/rules.py`:

- `get_legal_plays(hand, trick, hearts_broken, is_first_trick) -> list[Card]`:
  - Must follow lead suit if possible
  - First trick: cannot play hearts or Q♠ (unless hand is all point cards)
  - Cannot lead hearts until hearts are broken (unless only hearts remain)
  - 2♣ leads the very first trick

- `validate_play(card, hand, trick, hearts_broken, is_first_trick) -> bool`
- `validate_pass(cards, hand) -> bool`: checks that exactly 3 cards are provided, all are in the player's hand, and no duplicates

**Tests:** Exhaustive rule coverage — must-follow-suit, hearts leading restriction, first-trick restrictions, edge cases (hand is all hearts, etc.).

---

### Step 4: Round Engine (Single Round Execution)

Create `hearts/round.py`:

- `Round` class:
  - Constructor takes 4 bot instances and dealt hands
  - Executes the card-passing phase (calls each bot's `pass_cards`)
  - Executes 13 tricks sequentially:
    - Determines play order (2♣ leads first trick, trick winner leads thereafter)
    - For each player in order: builds `PlayerView`, calls bot's `play_card`, validates move
    - Resolves trick winner, updates state
  - Detects shoot-the-moon
  - Returns round scores

- `PlayerView` dataclass — **this is what bots see** (critical for ML later):
  - `hand`: the player's current cards
  - `player_index`: which seat (0–3)
  - `trick_so_far`: cards played in current trick (list of (player_index, Card))
  - `lead_suit`: Suit or None
  - `hearts_broken`: bool
  - `is_first_trick`: bool
  - `cards_played_this_round`: all cards played so far (list of (trick_num, player_index, Card))
  - `tricks_won_by_player`: list of 4 ints (count of tricks won per player)
  - `points_taken_by_player`: list of 4 ints (points accumulated per player)
  - `pass_direction`: which direction cards were passed
  - `cards_received_in_pass`: the 3 cards this player received
  - `legal_plays`: list of legal cards (pre-computed by engine)

**Tests:** Full round execution with deterministic bots, correct score tallying, shoot-the-moon scoring, proper trick sequencing.

---

### Step 5: Game Engine (Multi-Round to 100)

Create `hearts/game.py`:

- `Game` class:
  - Takes 4 bot instances
  - Runs rounds until a player hits ≥100 points
  - Pass direction rotates: Left → Right → Across → None → Left → ...
  - Returns final scores and full game history

- `GameResult` dataclass:
  - `final_scores`: list of 4 ints
  - `winner`: player index (lowest score)
  - `rounds`: list of round results
  - `num_rounds`: int

**Tests:** Game termination condition, pass direction rotation, winner determination, tie-breaking.

---

### Step 6: Bot Interface & Simple Bots

Create `hearts/bot.py` (abstract interface) and `hearts/bots/` directory:

**Bot ABC:**

```python
class Bot(ABC):
    @abstractmethod
    def pass_cards(self, view: PassView) -> list[Card]:
        """Select 3 cards to pass."""

    @abstractmethod
    def play_card(self, view: PlayerView) -> Card:
        """Select a card to play."""

    def on_trick_complete(self, trick_result: TrickResult) -> None:
        """Optional callback — observe completed trick."""

    def on_round_complete(self, round_result: RoundResult) -> None:
        """Optional callback — observe completed round."""
```

**RandomBot** (`hearts/bots/random_bot.py`):

- Passes 3 random cards
- Plays a random legal card
- Serves as baseline and testing tool

**RuleBot** (`hearts/bots/rule_bot.py`):

- Heuristic strategy that students should aim to beat:
  - Pass: pass highest hearts, Q♠, K♠, A♠ when possible
  - Play: "duck" (play low to avoid taking tricks with points), void a suit early, dump Q♠ when safe
- Doesn't need to be brilliant — just clearly better than random

**Tests:** Both bots always return legal moves across many random games (fuzz testing). RuleBot consistently beats RandomBot in tournaments.

---

### Step 7: Tournament Runner

Create `hearts/tournament.py`:

- `Tournament` class:
  - Takes N bot factories (callables that create bot instances)
  - Runs configurable number of games
  - Rotates seat positions to reduce positional advantage
  - Collects statistics:
    - Win rates
    - Average scores
    - Score distributions
    - Per-round average points
  - Returns `TournamentResult` with all stats

- Console output: summary table of results

**Tests:** Correct rotation logic, stats computation, deterministic results with seeded RNG.

---

### Step 8: Documentation & Final Tier 1 Polish

- Docstrings on every public class and method
- `README.md`: project overview, rules summary, how to create a bot, how to run a tournament
- `examples/my_first_bot.py`: minimal bot example with comments
- Type hints throughout
- Run full test suite, ensure 100% pass rate

---

## Tier 2: Classical ML Framework

### Step 9: Game Data Generation & Storage

Create `hearts/data/generator.py` and `hearts/data/schema.py`:

- **Data schema** — for each decision point, store:
  - **Common features** (shared context):
    - Round number, pass direction
    - Current scores (all 4 players, both this round and cumulative)
    - Cards already played this round (could be encoded as 52-bit vector)
    - Hearts broken flag
  - **For `play_card` decisions:**
    - Hand (52-bit binary vector: 1 if card in hand)
    - Current trick cards and positions
    - Legal plays (52-bit mask)
    - Trick number (1–13)
    - Player's seat position relative to lead
    - **Label/target**: card that was played
    - **Outcome** (for reward signals): points taken this trick, final round score, game result
  - **For `pass_cards` decisions:**
    - Hand before passing (52-bit vector)
    - **Label/target**: the 3 cards passed
    - **Outcome**: round score, game result

- **Generator**: runs N games with RuleBot (or mixed bots), serializes every decision point to JSON Lines format
- **Output location**: `data/output/` directory (gitignored). Default filenames: `play_decisions.jsonl` and `pass_decisions.jsonl`
- Generate a provided dataset: ~10,000 games → ~520,000 play decisions + ~40,000 pass decisions

**Tests:** Schema validation, generated data can be loaded and shapes are correct.

---

### Step 10: Feature Engineering Scaffold

Create `hearts/ml/features.py`:

- `FeatureExtractor` base class with method `extract(view: PlayerView) -> np.ndarray`
- **Provided example** `BasicFeatureExtractor`:
  - Hand as 52-dim binary vector
  - Cards played this round as 52-dim binary vector
  - Current trick as 52-dim binary vector (with positional encoding for play order)
  - Hearts broken: 1-dim
  - Points taken per player: 4-dim
  - Trick number: 1-dim (normalized)
  - Cumulative scores: 4-dim (normalized)
  - Total: ~165 features (students are told this is a starting point to improve)

- **Student task stub** `CustomFeatureExtractor`:
  - Skeleton with TODOs and hints:
    - "Consider: which suits have you voided?"
    - "Consider: which high cards (Q♠, K♠, A♠) are still unplayed?"
    - "Consider: how many hearts are still out?"
    - "Consider: what can you infer about other players' hands from their plays?"

**Tests:** Feature dimensions are consistent, no NaN/inf values, feature values in expected ranges.

---

### Step 11: Training Pipeline Scaffold

Create `hearts/ml/train.py`:

- **Data loading**: reads generated game data, applies feature extractor, produces X (features) and y (labels) matrices
- **Label encoding**:
  - For `play_card`: card index 0–51 (classification over legal moves)
  - For `pass_cards`: multi-label or treated as 3 sequential decisions
- **Model scaffold** using scikit-learn (low barrier to entry):
  - Provided working example with `RandomForestClassifier`
  - Clear interface: `train(X, y) -> model`, `predict(model, x) -> card`
  - Students can swap in: gradient boosting, SVM, simple neural net via `MLPClassifier`
- **Train/validation split** with evaluation metrics:
  - Classification accuracy on held-out data
  - **But more importantly**: tournament performance vs RuleBot

- **Training script** (`scripts/train_model.py`):
  - Loads data, extracts features, trains model, saves to `models/` directory
  - CLI args via `argparse`: `--data-dir` (default `data/output/`), `--output` (default `models/model.pkl`), `--features` (extractor class name), `--n-estimators`, `--test-size`

**Tests:** Pipeline runs end-to-end on small data sample, model can be saved/loaded, predictions are valid card indices.

---

### Step 12: ML Bot Wrapper

Create `hearts/bots/ml_bot.py`:

- `MLBot(Bot)`:
  - Constructor takes a trained model, feature extractor, and an optional `pass_strategy: Bot` (defaults to a `RuleBot` instance via composition — **not** inheritance)
  - `play_card`: extracts features from PlayerView → model predicts → masks to legal moves → selects card
  - `pass_cards`: delegates to `self.pass_strategy.pass_cards()` (composition pattern). Students can replace this with a trained pass model later.
  - Handles edge cases: model predicts illegal move → fall back to highest-probability legal move

- Factory function: `load_ml_bot(model_path, feature_extractor) -> Bot`

**Tests:** MLBot always plays legal cards, gracefully handles all game situations, can complete full games.

---

### Step 13: Evaluation & Iteration Tools

Create `hearts/ml/evaluate.py` and update tournament:

- **Head-to-head evaluation**: run ML bot vs RuleBots in tournament, report win rate and average score margin
- **Learning curve**: train on increasing data sizes, plot tournament performance
- **Feature importance**: for tree-based models, report which features matter most
- **Comparison dashboard** (optional simple script): table of different student models' tournament results

**Tests:** Head-to-head evaluation produces correct win rates, learning curve function runs on small data, feature importance extraction works for tree-based models.

---

### Step 14: Student-Facing Documentation for Tier 2

- `docs/ml_guide.md`:
  - Explains the feature engineering task and why feature choice matters
  - Walks through the provided `BasicFeatureExtractor`
  - Lists the suggested improvements with increasing difficulty
  - Explains training pipeline and how to evaluate
  - Rubric: beat RandomBot (baseline) → beat RuleBot (good) → beat other students (excellent)

- `docs/data_format.md`: schema of generated game data
- `docs/api_reference.md`: full reference of PlayerView fields and what information is available

---

## File Structure

```
hearts-project/
├── pyproject.toml         # Step 0
├── .gitignore             # Step 0
├── README.md              # Step 8
├── hearts/
│   ├── __init__.py
│   ├── __main__.py        # CLI entry point: `python -m hearts`
│   ├── types.py           # Step 1
│   ├── state.py           # Step 2
│   ├── rules.py           # Step 3
│   ├── round.py           # Step 4
│   ├── game.py            # Step 5
│   ├── bot.py             # Step 6 (ABC)
│   ├── tournament.py      # Step 7
│   ├── bots/
│   │   ├── __init__.py
│   │   ├── random_bot.py  # Step 6
│   │   ├── rule_bot.py    # Step 6
│   │   └── ml_bot.py      # Step 12
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py   # Step 9
│   │   └── schema.py      # Step 9
│   └── ml/
│       ├── __init__.py
│       ├── features.py    # Step 10
│       ├── train.py       # Step 11
│       └── evaluate.py    # Step 13
├── tests/
│   ├── test_types.py      # Step 1
│   ├── test_state.py      # Step 2
│   ├── test_rules.py      # Step 3
│   ├── test_round.py      # Step 4
│   ├── test_game.py       # Step 5
│   ├── test_bots.py       # Step 6
│   ├── test_tournament.py # Step 7
│   ├── test_generator.py  # Step 9
│   ├── test_features.py   # Step 10
│   ├── test_train.py      # Step 11
│   ├── test_ml_bot.py     # Step 12
│   └── test_evaluate.py   # Step 13
├── scripts/
│   ├── generate_data.py   # Step 9
│   ├── train_model.py     # Step 11
│   └── run_tournament.py  # Step 7
├── data/
│   └── output/            # Step 9: generated game data (gitignored)
├── models/                # Trained model files (gitignored)
├── docs/
│   ├── ml_guide.md        # Step 14
│   ├── data_format.md     # Step 14
│   └── api_reference.md   # Step 14
├── examples/
│   └── my_first_bot.py    # Step 8
└── README.md              # Step 8
```

---

## Key Design Decisions & Rationale

**Why `PlayerView` is so detailed:** This is the bridge between Tier 1 and Tier 2. Everything a student might want as an ML feature must be accessible through `PlayerView`. By making this rich from the start, students don't need to modify the engine — they just need to decide which information to encode.

**Why RuleBot matters:** It sets the bar. Random is too easy to beat; a decent heuristic bot forces students to actually learn something useful from the data. It also generates better training data than RandomBot would.

**Why scikit-learn first (not PyTorch):** Lower barrier. Students can focus on feature engineering, which is where the real learning happens at this level. A good feature vector with a random forest will beat a bad feature vector with a neural net. A future extension could introduce PyTorch for students who want to go deeper.

**Why classification over legal moves, not regression:** Predicting "which card to play" as a classification task with masking to legal moves is cleaner than trying to predict card values. The masking step is a great teaching moment about constrained prediction.

**Why composition for MLBot pass strategy:** `MLBot` holds a `pass_strategy: Bot` instance (defaulting to `RuleBot`) rather than inheriting from `RuleBot` or duplicating logic. This keeps the architecture clean and lets students swap in a trained pass model without modifying `MLBot` itself.
