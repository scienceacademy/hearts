# Hearts Bot ML Project

A Hearts card game engine with a bot framework and ML pipeline for training
card-playing agents.

## Setup

Requires Python 3.11+.

```bash
pip install -e ".[dev,ml]"
```

## Rules Summary

Hearts is a 4-player trick-taking game where the goal is to **avoid points**.
Hearts are worth 1 point each and the Queen of Spades is worth 13. The player
with the lowest score when someone reaches 100 wins.

**Shoot the moon:** if one player takes all 26 points in a round, they score 0
and everyone else gets 26.

## Quick Start

Run a default tournament:

```bash
python -m hearts
```

Run with options:

```bash
python scripts/run_tournament.py --games 200 --seed 42
```

## Creating a Bot

Subclass `hearts.bot.Bot` and implement `pass_cards` and `play_card`:

```python
from hearts.bot import Bot
from hearts.state import PassView, PlayerView
from hearts.types import Card

class MyBot(Bot):
    def pass_cards(self, view: PassView) -> list[Card]:
        # Pass 3 cards — view.hand has your 13 cards
        return sorted(view.hand, key=lambda c: c.rank)[-3:]

    def play_card(self, view: PlayerView) -> Card:
        # Pick from view.legal_plays
        return min(view.legal_plays, key=lambda c: c.rank)
```

See `examples/my_first_bot.py` for a complete example.

## Running a Tournament

```python
from hearts.tournament import Tournament
from hearts.bots.rule_bot import RuleBot

t = Tournament(
    bot_factories={"MyBot": MyBot, "RuleBot": RuleBot},
    num_games=100,
)
result = t.run()
print(result.summary())
```

## Running Tests

```bash
pytest
```

## Project Structure

```
hearts/
  types.py        # Card, Suit, Deck
  state.py        # GameState, PlayerView, RoundResult, etc.
  rules.py        # Legal move validation
  round.py        # Single round engine
  game.py         # Multi-round game engine
  bot.py          # Bot ABC
  tournament.py   # Tournament runner
  bots/
    random_bot.py  # Random baseline
    rule_bot.py    # Heuristic bot
```
