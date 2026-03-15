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
  - Neural network dependencies (optional install group `[nn]`): `torch` (CPU-only)
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

### Step 14: Student-Facing Documentation for Tier 2

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

## Tier 3: Neural Networks & Learned Representations

**Prerequisite:** Students have completed Tier 2 — they have a working `StudentFeatureExtractor`, understand the train/evaluate workflow, and have tournament results for their hand-crafted features with RandomForest.

**Goal:** Explore whether a neural network can match or beat hand-crafted features by learning its own representations. Students will progress from a drop-in sklearn MLP (minimal new code) to a PyTorch model where they control the architecture and input representation.

**New dependency:** `torch` (CPU only — added to `pyproject.toml` as optional install group `[nn]`). No GPU required. The problem is small enough (~20k parameters, ~500k training samples) to train in under a minute on any modern laptop CPU.

### Step 15: sklearn MLP Baseline

Create `scripts/train_mlp.py` (a copy of `train_model.py` with the model swapped):

This step introduces neural networks with **zero new dependencies or APIs**. Students swap one line — `RandomForestClassifier` → `MLPClassifier` — and compare results.

- **Script** (`scripts/train_mlp.py`):
  - Identical to `train_model.py` except the model is `MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=seed)`
  - Uses the same `FeatureExtractor`, same data, same `save_model` format
  - The saved model works with the existing `evaluate_model.py` and `MLBot` unchanged (sklearn's `MLPClassifier` supports `predict_proba`)
  - CLI args: same as `train_model.py`

- **Student task**: run `train_mlp.py` with their `StudentFeatureExtractor`, then `evaluate_model.py`. Compare results to their RandomForest. Expected outcome: similar or marginally different performance, reinforcing the Tier 2 lesson that features matter more than model choice when features are hand-crafted.

- No new tests needed — existing `test_train.py` and `test_ml_bot.py` cover the pipeline. Optionally add a smoke test that `train_mlp.py` runs end-to-end.

---

### Step 16: PyTorch Training Scaffold

Create `hearts/ml/neural.py` and `scripts/train_neural.py`:

This step introduces PyTorch with a **fully provided training scaffold**. Students read and understand the code but do not need to write the training loop. The scaffold is deliberately simple and heavily commented.

#### `hearts/ml/neural.py`:

- **`HeartsDataset(torch.utils.data.Dataset)`**: loads `play_decisions.jsonl`, applies a `FeatureExtractor`, returns `(feature_tensor, label, legal_moves_mask)` tuples
- **`HeartsNet(nn.Module)`**: a simple feedforward network
  - Default architecture: `input_dim → 128 → ReLU → 64 → ReLU → 52`
  - Forward pass returns raw logits (52-dim)
  - The architecture is defined in a single `nn.Sequential` block that students will later modify
- **`MaskedCrossEntropyLoss`**: applies legal-move masking before computing cross-entropy. Sets logits for illegal moves to `-inf` before softmax. This is a key teaching moment — the model only learns from legal moves.
- **`train_neural(data_dir, extractor, net, epochs, batch_size, lr, seed) -> tuple[HeartsNet, dict]`**:
  - Standard PyTorch training loop: DataLoader → forward → loss → backward → step
  - 80/20 train/test split
  - Tracks loss per epoch and test accuracy
  - Returns trained model and metrics dict (same keys as Tier 2: `train_accuracy`, `test_accuracy`, `feature_importances` set to `None` since neural nets don't have built-in importance)
  - **Heavily commented** — every block explains what's happening and why

#### `hearts/bots/neural_bot.py`:

- **`NeuralBot(Bot)`**: like `MLBot` but wraps a PyTorch model instead of sklearn
  - `play_card`: extracts features → `torch.tensor` → `net.forward()` → mask illegal moves → argmax → card
  - `pass_cards`: delegates to `pass_strategy` (same composition pattern as `MLBot`)
  - Handles `torch.no_grad()` context for inference

#### `scripts/train_neural.py`:

- CLI args: `--data-dir`, `--output` (default `models/neural.pt`), `--features` (default `StudentFeatureExtractor`), `--epochs` (default 20), `--batch-size` (default 512), `--lr` (default 0.001), `--seed`
- Prints per-epoch training loss and final train/test accuracy
- Saves model state dict + extractor class name

#### `scripts/evaluate_neural.py`:

- Same interface as `evaluate_model.py` but loads a PyTorch model via `NeuralBot`
- CLI args: `--model` (default `models/neural.pt`), `--games` (default 200), `--seed`
- Prints the same unified report format as Tier 2 (but without feature importance section, replaced with a note that neural nets don't provide it directly)

- **Student task at this step**: run `train_neural.py` with their `StudentFeatureExtractor`, evaluate, and compare to their RandomForest and sklearn MLP results. All three models use the same hand-crafted features. Expected outcome: comparable performance, establishing the baseline before Step 17 changes the input representation.

**Tests:** `HeartsDataset` returns correct shapes, `HeartsNet` forward pass produces 52-dim output, `MaskedCrossEntropyLoss` correctly zeros illegal moves, `NeuralBot` always plays legal cards (fuzz test), training loop runs end-to-end on small data.

---

### Step 17: Raw Input Experiments

Create `hearts/ml/raw_features.py`:

This is the conceptual payoff. Students replace hand-crafted features with minimal raw inputs and see if the network can learn useful representations on its own.

#### `hearts/ml/raw_features.py`:

- **`RawFeatureExtractor(FeatureExtractor)`**: a minimal extractor with no domain engineering:
  - Hand as 52-dim binary vector (same as basic)
  - All cards played this round as 52-dim binary vector (same as basic)
  - Current trick as 52-dim binary vector (no positional encoding — just 0/1)
  - Hearts broken: 1-dim
  - Trick number: 1-dim (normalized)
  - Points taken per player: 4-dim
  - Cumulative scores: 4-dim
  - Total: ~165 features — similar count to `BasicFeatureExtractor`, but **no derived features** (no suit voids, no dangerous card tracking, nothing the student added in Tier 2). The raw binary vectors contain the same underlying information, but the network must learn to extract it.
  - `feature_names()` returns generic names (`hand_0`, `hand_1`, ..., `played_0`, ...) since there's no semantic meaning at this level.

- **Student task**: modify the `HeartsNet` architecture in their copy and experiment:
  - Does a wider network (256 → 128 → 52) help?
  - Does a deeper network (128 → 128 → 64 → 32 → 52) help?
  - Does adding dropout change anything?
  - How does `RawFeatureExtractor` + neural net compare to `StudentFeatureExtractor` + RandomForest?

- **Key question for students to answer**: can the network discover features like "void in a suit" on its own? Or does hand-crafting still win? (The honest answer: for this problem size and data volume, hand-crafted features are likely competitive or better. But the exercise demonstrates the *concept* of learned representations.)

No new tests needed beyond verifying `RawFeatureExtractor` produces correct shapes.

---

### Step 18: Comparison Analysis & Documentation

Create `docs/nn_guide.md`:

This is the Tier 3 lab handout, structured similarly to Tier 2's `ml_guide.md`.

#### `docs/nn_guide.md` — structure:

**Part 1: From Features to Learned Representations** (~1 page)
- Recap of Tier 2: you chose what information to give the model
- The neural network proposition: what if the model could figure out what information matters on its own?
- Intuition for hidden layers: each layer transforms the input into a more useful representation

**Part 2: The sklearn MLP** (~1 page)
- What it is: a neural network in sklearn's familiar API
- How to run it: copy-paste commands
- Expected result and what it tells us (model choice alone doesn't help much)

**Part 3: PyTorch Walkthrough** (~2 pages)
- Guided code walkthrough of `neural.py`: Dataset, Model, Loss, Training loop
- What each PyTorch concept maps to: tensors = numpy arrays on steroids, `nn.Module` = a function with learnable parameters, backpropagation = how the model updates
- The masked cross-entropy loss: why we need it, what would happen without it

**Part 4: Architecture Experiments** (~2 pages)
- How to modify the network architecture (literally which line to change)
- Experiments to try, with predictions:
  - Wider first layer: might help capture more card interactions
  - Deeper network: more abstraction, but diminishing returns on small data
  - Dropout: regularization to prevent overfitting
- How to read the training loss curve: what underfitting and overfitting look like

**Part 5: The Big Comparison** (~1 page)
- Students should have results for at least 4 configurations:
  1. RandomForest + hand-crafted features (Tier 2 best)
  2. sklearn MLP + hand-crafted features
  3. PyTorch net + hand-crafted features
  4. PyTorch net + raw features
- Guiding questions: which won? Why? What does this tell us about when feature engineering matters vs. when learned representations can substitute?

**Final reflection prompt**: "If you had 100× more training data and a GPU, would you expect the raw-feature neural net to overtake hand-crafted features? Why or why not?" (This connects to the broader deep learning narrative without requiring students to actually scale up.)

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
│   │   ├── ml_bot.py      # Step 12
│   │   └── neural_bot.py  # Step 16
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py   # Step 9
│   │   └── schema.py      # Step 9
│   └── ml/
│       ├── __init__.py
│       ├── features.py    # Step 10 ← STUDENTS EDIT THIS FILE
│       ├── train.py       # Step 11 (fixed, students read but don't modify)
│       ├── evaluate.py    # Step 13
│       ├── neural.py      # Step 16: PyTorch scaffold
│       └── raw_features.py # Step 17
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
│   ├── test_evaluate.py   # Step 13
│   ├── test_neural.py     # Step 16
│   └── test_neural_bot.py # Step 16
├── scripts/
│   ├── generate_data.py   # Step 9
│   ├── train_model.py     # Step 11
│   ├── evaluate_model.py  # Step 13
│   ├── run_tournament.py  # Step 7
│   ├── train_mlp.py       # Step 15
│   ├── train_neural.py    # Step 16
│   └── evaluate_neural.py # Step 16
├── data/
│   └── output/            # Step 9: generated game data (gitignored)
├── models/                # Trained model files (gitignored)
├── docs/
│   ├── ml_guide.md        # Step 14: student lab handout (Tier 2)
│   ├── nn_guide.md        # Step 18: neural network lab handout (Tier 3)
│   ├── data_format.md     # Step 14: data schema reference
│   └── api_reference.md   # Step 14: PlayerView / PassView / FeatureExtractor reference
├── examples/
│   └── my_first_bot.py    # Step 8
└── README.md              # Step 8
```

---

## Key Design Decisions & Rationale

**Why `PlayerView` is so detailed:** This is the bridge between Tier 1 and Tier 2. Everything a student might want as an ML feature must be accessible through `PlayerView`. By making this rich from the start, students don't need to modify the engine — they just need to decide which information to encode.

**Why students only edit `features.py` in Tier 2:** This is their first ML course. Giving them one file to modify with a clear contract (implement `extract()` and `feature_names()`) keeps the scope manageable. The fixed model and pipeline let them focus entirely on the interesting question: what information matters?

**Why the model is fixed (RandomForest) in Tier 2:** A good feature vector with a random forest will beat a bad feature vector with a neural net. By fixing the model, students learn that *representation is the bottleneck* — the most important lesson in applied ML. Model choice is available as an optional extension for advanced students, and becomes the focus in Tier 3.

**Why RuleBot matters:** It sets the bar. Random is too easy to beat; a decent heuristic bot forces students to actually learn something useful from the data. It also generates better training data than RandomBot would.

**Why the evaluation report is all-in-one:** Students need immediate, interpretable feedback. Printing accuracy, feature importance, and tournament results in a single report means they can see the full picture after every iteration without juggling multiple scripts or outputs.

**Why classification over legal moves, not regression:** Predicting "which card to play" as a classification task with masking to legal moves is cleaner than trying to predict card values. The masking step is a great teaching moment about constrained prediction.

**Why composition for MLBot pass strategy:** `MLBot` holds a `pass_strategy: Bot` instance (defaulting to `RuleBot`) rather than inheriting from `RuleBot` or duplicating logic. This keeps the architecture clean and lets advanced students swap in a trained pass model without modifying `MLBot` itself.

**Why Tier 3 starts with sklearn MLP:** It isolates the "neural network" variable from the "PyTorch" variable. Students see that switching to a neural net with the same features and familiar API doesn't magically improve results. This prevents the misconception that neural nets are universally better, and sets up the real question of Tier 3: can neural nets compensate for less feature engineering by learning representations?

**Why Tier 3 is CPU-only:** The problem is genuinely small — ~20k parameters, ~500k samples. Training takes under a minute on CPU. Requiring a GPU would create access barriers and distract from the pedagogical goals. Students who want to scale up can do so on their own.
