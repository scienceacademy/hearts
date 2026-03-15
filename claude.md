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

**Student context:** This is students' first ML course. They understand Python but have not used scikit-learn, numpy, or ML concepts before. The documentation is a teaching tool, not just a reference. Students will **only** modify `hearts/ml/features.py` — the model (RandomForestClassifier) and training pipeline are fixed. Their sole task is feature engineering: deciding what information from `PlayerView` to encode as numbers, and how.

### Step 9: Game Data Generation & Storage

Create `hearts/data/schema.py`, `hearts/data/generator.py`, and `scripts/generate_data.py`.

#### Card Index Mapping

Cards are represented as integers 0–51 using the mapping: `index = suit_ordinal * 13 + (rank - 2)`, where suit ordinals are CLUBS=0, DIAMONDS=1, SPADES=2, HEARTS=3 and ranks are 2–14 (J=11, Q=12, K=13, A=14).

Examples: 0 = 2♣, 12 = A♣, 13 = 2♦, 37 = Q♠, 51 = A♥.

`schema.py` provides `card_to_index(card: Card) -> int` and `index_to_card(index: int) -> Card`.

#### Play Decision Record (`play_decisions.jsonl`)

Each line is a JSON object mirroring `PlayerView` fields, plus a label and outcome annotations. Example:

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

Field notes:
- `trick_so_far`: list of `[player_index, card_index]` pairs in play order — the in-progress trick
- `cards_played_this_round`: list of completed tricks, each a list of 4 `[player_index, card_index]` pairs in play order. Trick number is implicit (array index + 1). Does **not** include the current in-progress trick (that's `trick_so_far`)
- `lead_suit`: string name of suit enum, or `null` if this player is leading
- `pass_direction`: string — `"LEFT"`, `"RIGHT"`, `"ACROSS"`, or `"NONE"`
- `cards_received_in_pass`: list of 3 card indices, or `[]` during No Pass rounds
- **Label**: `card_played` — the card index (0–51) that was played
- **Outcome annotations** (added after game completes): `points_this_trick`, `round_score`, `game_rank` (1–4, 1 = best)

#### Pass Decision Record (`pass_decisions.jsonl`)

Each line mirrors `PassView` fields. Example:

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

#### `hearts/data/schema.py`

- `card_to_index(card: Card) -> int` and `index_to_card(index: int) -> Card`
- `serialize_player_view(view: PlayerView) -> dict` — converts to JSON-safe dict using index mapping. `cards_played_this_round` is restructured from flat `(trick_num, player_index, Card)` tuples into the nested-by-trick format shown above.
- `deserialize_player_view(data: dict) -> PlayerView` — reconstructs a `PlayerView`, flattening the nested trick structure back to `(trick_num, player_index, Card)` tuples.
- `serialize_pass_view(view: PassView) -> dict`
- `deserialize_pass_view(data: dict) -> PassView`

#### `hearts/data/generator.py`

- `RecordingBot(Bot)` — wraps any `Bot` instance. Intercepts `pass_cards` and `play_card` calls: records the serialized view and the chosen action, then delegates to the inner bot. Stores records in memory as a list per game.
- `generate_game_data(bots: list[Bot], seed) -> tuple[GameResult, list[dict], list[dict]]` — wraps the 4 bots in `RecordingBot`s, runs one `Game`, then annotates each recorded decision with outcome data (`points_this_trick`, `round_score`, `game_rank`) from the `GameResult` and `RoundResult` objects. Returns the game result plus the annotated play and pass records.
- `generate_dataset(n_games, bot_factory, output_dir, seed)` — runs N games, writes `play_decisions.jsonl` and `pass_decisions.jsonl` line by line. Prints progress to stdout.

#### `scripts/generate_data.py`

CLI entry point:
- `--games` (default 10000)
- `--output-dir` (default `data/output/`)
- `--seed` (default: none, for reproducibility)
- Runs 4× RuleBot by default
- Prints summary on completion: number of games, total play decisions, total pass decisions, output file paths

**Tests (`tests/test_schema.py` and `tests/test_generator.py`):**
- **Schema round-trip**: serialize a `PlayerView` → JSON string → deserialize → compare all fields to original
- **Schema round-trip for `PassView`**
- **Card index mapping**: `index_to_card(card_to_index(card)) == card` for all 52 cards
- **Generator record count**: run a small game (e.g., 2 games with seeded RNG), verify number of play records = total cards played (52 per round × num_rounds per game × num_games), number of pass records = expected count (accounting for No Pass rounds)
- **Outcome annotation correctness**: verify `round_score` sums to 26 (or 78 on shoot-the-moon) per round across all 4 players, `game_rank` values are 1–4 with correct ordering
- **JSONL format**: records can be loaded with `json.loads()` line by line

---

### Step 10: Feature Engineering Scaffold

Create `hearts/ml/features.py`:

This is **the file students edit**. It must be clear, well-commented, and approachable for someone who has never written ML code.

- `FeatureExtractor` base class:
  - `extract(view: PlayerView) -> np.ndarray`: convert a player view into a numeric vector
  - `feature_names() -> list[str]`: return human-readable names for each dimension (used by evaluation to report feature importance). Must match length of `extract()` output.

- **`BasicFeatureExtractor`** (provided, working, ~165 features):
  - Hand as 52-dim binary vector (1 if card in hand, 0 otherwise)
  - Cards played this round as 52-dim binary vector
  - Current trick as 52-dim binary vector (with positional encoding: multiply by 0.25/0.5/0.75/1.0 based on play order)
  - Hearts broken: 1-dim (0 or 1)
  - Points taken per player: 4-dim (raw ints)
  - Trick number: 1-dim (normalized to 0.0–1.0)
  - Cumulative scores: 4-dim (divided by 100 for normalization)
  - **Every feature group should have a block comment explaining what it captures and why**
  - **End with a block comment listing what this extractor does NOT capture** — this is the launching pad for student work

- **`StudentFeatureExtractor`** (the student's working copy):
  - Starts as an exact copy of `BasicFeatureExtractor`
  - Students modify this class to add/change features
  - Includes clearly marked `# === YOUR FEATURES BELOW ===` section at the end
  - Pre-populated with commented-out example stubs for the first exercise (suit void tracking) to show the pattern:
    ```python
    # === YOUR FEATURES BELOW ===
    # Example: suit void tracking (uncomment and complete)
    # for suit in Suit:
    #     cards_of_suit = [c for c in view.hand if c.suit == suit]
    #     features.append(1.0 if len(cards_of_suit) == 0 else 0.0)
    #     names.append(f"void_{suit.name}")
    ```

**Tests:** Both extractors produce consistent dimensions, `feature_names()` length matches `extract()` output length, no NaN/inf values, feature values in expected ranges. Test with multiple different `PlayerView` snapshots (early trick, late trick, hearts broken, etc.).

---

### Step 11: Training Pipeline

Create `hearts/ml/train.py`:

This is a **fixed pipeline that students do not modify**. It should be readable (students are encouraged to read it to understand what's happening) but not a task for them.

- **`load_data(data_dir) -> list[dict]`**: reads `play_decisions.jsonl`, returns list of records
- **`build_dataset(records, extractor: FeatureExtractor) -> tuple[np.ndarray, np.ndarray]`**:
  - Reconstructs `PlayerView` from each record's serialized JSON
  - Calls `extractor.extract(view)` to get feature vector
  - Label is the card index (0–51) from the record
  - Returns X (n_samples × n_features) and y (n_samples,)
- **`train_model(X, y, seed) -> tuple[RandomForestClassifier, dict]`**:
  - Fixed model: `RandomForestClassifier(n_estimators=100, random_state=seed)`
  - 80/20 train/test split
  - Trains on training set
  - Returns the trained model and a metrics dict containing:
    - `train_accuracy`: float
    - `test_accuracy`: float
    - `feature_importances`: list of (feature_name, importance) tuples, sorted descending
  - No model selection, no hyperparameter flags — the model is a constant
- **`save_model(model, extractor, path)`**: pickles model + extractor class name together so evaluation can reload them as a pair

- **Script** (`scripts/train_model.py`):
  - CLI args: `--data-dir` (default `data/output/`), `--output` (default `models/model.pkl`), `--features` (extractor class name, default `StudentFeatureExtractor`), `--seed`
  - Prints to stdout:
    - Number of training samples
    - Feature vector dimensionality
    - Train/test accuracy
    - Top 10 features by importance
  - Saves model to disk

**Tests:** Pipeline runs end-to-end on a small generated dataset (~100 games), model can be saved and reloaded, predictions are valid card indices in range 0–51, metrics dict contains all expected keys.

---

### Step 12: ML Bot Wrapper

Create `hearts/bots/ml_bot.py`:

- `MLBot(Bot)`:
  - Constructor takes a trained model, feature extractor, and an optional `pass_strategy: Bot` (defaults to a `RuleBot` instance via composition — **not** inheritance)
  - `play_card`: extracts features from PlayerView → calls `model.predict_proba()` → masks to legal moves only (zero out probabilities for illegal cards) → selects highest-probability legal card. If the model doesn't support `predict_proba`, falls back to `predict()` and checks legality.
  - `pass_cards`: delegates to `self.pass_strategy.pass_cards()` (composition pattern)
  - Edge case handling: if all legal moves have zero probability after masking (shouldn't happen but defensive), fall back to random legal card

- Factory function: `load_ml_bot(model_path) -> Bot` — loads the pickled model + extractor class name, instantiates the extractor, returns an MLBot

**Tests:** MLBot always plays legal cards across many random games (fuzz test), gracefully handles edge cases, can complete full games without errors. Test with a deliberately bad model (random weights) to verify fallback behavior.

---

### Step 13: Evaluation Script

Create `hearts/ml/evaluate.py` and **script** `scripts/evaluate_model.py`:

This is the second half of the student workflow. After training, students run this to see how their bot actually plays.

- **`evaluate_model.py`** script:
  - CLI args: `--model` (default `models/model.pkl`), `--games` (default 200), `--seed`
  - Loads the saved model via `load_ml_bot()`
  - Runs a tournament: 1× MLBot vs 3× RuleBot, with seat rotation
  - Prints a unified results report to stdout:

    ```
    === Training Metrics ===
    Feature vector size:  169 features
    Train accuracy:       34.2%
    Test accuracy:        31.8%

    === Feature Importance (top 10) ===
    1. hand_QS              0.0823
    2. trick_num            0.0614
    3. hearts_broken        0.0487
    ...

    === Tournament Results (200 games) ===
    Player          Win Rate    Avg Score   Avg Rank
    MLBot           28.5%       24.3        2.1
    RuleBot-1       24.0%       25.1        2.4
    RuleBot-2       23.5%       25.8        2.5
    RuleBot-3       24.0%       25.4        2.4

    Baseline (RandomBot):  ~5% win rate, ~32 avg score
    Target (beat RuleBot): >25% win rate consistently
    ```

  - The report is designed so students can immediately see:
    - Whether their features are helping (compare to baseline numbers)
    - Which features the model is actually using (importance)
    - Whether they're overfitting (train vs test accuracy gap)

- **Evaluation functions** in `hearts/ml/evaluate.py` (used by the script but also importable):
  - `run_evaluation(model_path, n_games, seed) -> EvaluationResult`
  - `print_report(result: EvaluationResult)` — formats the stdout report
  - `EvaluationResult` dataclass: holds training metrics, feature importance, and tournament stats

**Tests:** Evaluation runs end-to-end with a trained model, report contains all expected sections, win rates sum to ~100%, tournament uses seat rotation.

---

### Step 14: Student-Facing Documentation

This is the centerpiece of Tier 2. The `ml_guide.md` functions as a **lab handout** — it teaches concepts, walks through the codebase, and structures the assignment.

#### `docs/ml_guide.md` — structure:

**Part 1: What You're Building** (~1 page)
- Your goal: build a Hearts bot that beats RuleBot by teaching a model to pick good cards
- The model (RandomForest) is fixed — your job is to decide **what information to give it**
- Analogy: the model sees a list of numbers, not cards. Your feature extractor is the translator. A good translator tells the model what matters; a bad one buries the signal in noise.

**Part 2: ML Concepts You Need** (~2 pages)
- **Classification**: the model sees a game state and picks one of 52 cards. It's trained on thousands of examples of what RuleBot chose in similar situations.
- **Features**: the numeric representation of a game state. Same game state, different features = different model behavior.
- **Training and testing**: we split the data so we can check if the model memorized or actually learned. Explain train/test split in plain terms.
- **Feature importance**: the model can tell us which numbers it relied on most. This is your debugging tool.
- Keep it concrete — every concept illustrated with a Hearts example, not abstract math.

**Part 3: The Workflow** (~1 page)
- Two-step process, documented as copy-paste terminal commands:
  1. `python scripts/train_model.py` — trains the model using your features, saves to `models/model.pkl`
  2. `python scripts/evaluate_model.py` — loads the model, plays 200 games vs RuleBot, prints results
- Edit `hearts/ml/features.py` → run step 1 → run step 2 → read the report → repeat
- Explain what each section of the output report means

**Part 4: Understanding BasicFeatureExtractor** (~2 pages)
- Full code walkthrough, feature group by feature group
- For each group: what it encodes, what a value of 0 vs 1 means, why it might help
- End with an explicit list: "Here's what the basic extractor **cannot** see" — this seeds Part 5:
  - It doesn't know which suits you're void in
  - It doesn't know if dangerous cards (Q♠, K♠, A♠) have been played yet
  - It doesn't know how many hearts are still out there
  - It doesn't know where you are in the trick (leading? last to play?)
  - It doesn't know what other players are likely to be holding

**Part 5: Feature Engineering Exercises** (~3 pages)
- Structured as a progression. Each exercise has:
  - **Motivation**: a scenario that shows why this information matters (e.g., "You hold the K♠. Should you be worried? That depends entirely on whether the Q♠ has already been played.")
  - **What to encode**: description of the feature(s) to add, with encoding hints (binary flag? count? normalized value?)
  - **Implementation hint**: which `PlayerView` fields to use
  - **Sanity check**: after training, what feature importance should you expect? (e.g., "If Q♠-still-out shows up in the top 10, you've done it right.")

- **Exercise progression:**
  1. **Suit void tracking** (4 features) — am I void in each suit? Binary flags. Uses `view.hand`. *Motivation: if you're void in diamonds, you can dump hearts or Q♠ when diamonds are led.*
  2. **Dangerous cards remaining** (3 features) — have Q♠, K♠, A♠ been played yet? Binary flags. Uses `view.cards_played_this_round`. *Motivation: the entire spade game changes once Q♠ is gone.*
  3. **Hearts remaining** (1 feature) — how many of the 13 hearts haven't appeared yet? Normalized count. Uses `view.cards_played_this_round`. *Motivation: leading hearts late in a round when 10 hearts are still out is very different from when only 2 remain.*
  4. **Trick position** (1–2 features) — where am I in the current trick? Am I leading, or playing last? Uses `len(view.trick_so_far)`. *Motivation: if you play last, you know exactly what you'll take. If you lead, you're guessing.*
  5. **Opponent void inference** (4–8 features) — which players appear to be void in which suits, based on them not following suit earlier? Uses `view.cards_played_this_round`. *Motivation: if the player to your left is void in clubs, leading clubs lets them dump points on you. This is the hardest exercise — it requires iterating over past tricks and reasoning about what other players' plays reveal.*

**Part 6: Interpreting Your Results** (~1 page)
- How to read the evaluation report
- What "good" looks like: concrete targets
  - Baseline (RandomBot): ~5% win rate, ~32 avg score
  - Basic features only: ~15–20% win rate
  - Good features: >25% win rate (consistently beating RuleBot)
  - Great features: >30% win rate
- Common failure modes:
  - Features that don't change (constant value = zero information)
  - Features that duplicate existing ones (redundant = wasted capacity)
  - Train accuracy much higher than test accuracy (overfitting — too many features, not enough signal)
  - Bug check: if win rate drops below RandomBot, you have a bug in your feature code

**Part 7: Extensions** (optional, for advanced students, ~1 page)
- Train a model for `pass_cards` using `pass_decisions.jsonl` and write a `PassFeatureExtractor`
- Experiment with different models: swap `RandomForestClassifier` for `GradientBoostingClassifier` or `MLPClassifier` in a copy of `train.py`
- Generate training data with mixed bot populations (e.g., 2× RuleBot + 2× RandomBot) and compare model quality
- Feature engineering for "shooting the moon" detection

#### `docs/data_format.md`:
- Schema of `play_decisions.jsonl` — every field, its type, and an example record
- Schema of `pass_decisions.jsonl` (brief, since it's optional)
- How records map to `PlayerView` fields

#### `docs/api_reference.md`:
- Full reference for `PlayerView`: every field, its type, what it contains, when it's populated
- Full reference for `PassView`
- Card index mapping (0–51 → which card)
- `FeatureExtractor` interface: methods to implement, contract (dimensions must be consistent, names must match)

---

## File Structure

```
hearts-project/
├── pyproject.toml         # Step 0
├── .gitignore             # Step 0
├── hearts/
│   ├── __init__.py
│   ├── __main__.py        # Step 8: CLI entry point
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
│       ├── features.py    # Step 10 ← STUDENTS EDIT THIS FILE
│       ├── train.py       # Step 11 (fixed, students read but don't modify)
│       └── evaluate.py    # Step 13
├── tests/
│   ├── test_types.py      # Step 1
│   ├── test_state.py      # Step 2
│   ├── test_rules.py      # Step 3
│   ├── test_round.py      # Step 4
│   ├── test_game.py       # Step 5
│   ├── test_bots.py       # Step 6
│   ├── test_tournament.py # Step 7
│   ├── test_schema.py     # Step 9
│   ├── test_generator.py  # Step 9
│   ├── test_features.py   # Step 10
│   ├── test_train.py      # Step 11
│   ├── test_ml_bot.py     # Step 12
│   └── test_evaluate.py   # Step 13
├── scripts/
│   ├── generate_data.py   # Step 9
│   ├── train_model.py     # Step 11
│   ├── evaluate_model.py  # Step 13
│   └── run_tournament.py  # Step 7
├── data/
│   └── output/            # Step 9: generated game data (gitignored)
├── models/                # Trained model files (gitignored)
├── docs/
│   ├── ml_guide.md        # Step 14: student lab handout
│   ├── data_format.md     # Step 14: data schema reference
│   └── api_reference.md   # Step 14: PlayerView / PassView / FeatureExtractor reference
├── examples/
│   └── my_first_bot.py    # Step 8
└── README.md              # Step 8
```

---

## Key Design Decisions & Rationale

**Why `PlayerView` is so detailed:** This is the bridge between Tier 1 and Tier 2. Everything a student might want as an ML feature must be accessible through `PlayerView`. By making this rich from the start, students don't need to modify the engine — they just need to decide which information to encode.

**Why students only edit `features.py`:** This is their first ML course. Giving them one file to modify with a clear contract (implement `extract()` and `feature_names()`) keeps the scope manageable. The fixed model and pipeline let them focus entirely on the interesting question: what information matters?

**Why the model is fixed (RandomForest):** A good feature vector with a random forest will beat a bad feature vector with a neural net. By fixing the model, students learn that *representation is the bottleneck* — the most important lesson in applied ML. Model choice is available as an optional extension for advanced students.

**Why RuleBot matters:** It sets the bar. Random is too easy to beat; a decent heuristic bot forces students to actually learn something useful from the data. It also generates better training data than RandomBot would.

**Why the evaluation report is all-in-one:** Students need immediate, interpretable feedback. Printing accuracy, feature importance, and tournament results in a single report means they can see the full picture after every iteration without juggling multiple scripts or outputs.

**Why classification over legal moves, not regression:** Predicting "which card to play" as a classification task with masking to legal moves is cleaner than trying to predict card values. The masking step is a great teaching moment about constrained prediction.

**Why composition for MLBot pass strategy:** `MLBot` holds a `pass_strategy: Bot` instance (defaulting to `RuleBot`) rather than inheriting from `RuleBot` or duplicating logic. This keeps the architecture clean and lets advanced students swap in a trained pass model without modifying `MLBot` itself.
