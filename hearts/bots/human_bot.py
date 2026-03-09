"""Interactive human player for terminal play."""

from __future__ import annotations

from hearts.bot import Bot
from hearts.state import (
    PassDirection,
    PassView,
    PlayerView,
    RoundResult,
    TrickResult,
)
from hearts.types import Card, Suit, card_points


def _sort_hand(cards: set[Card] | list[Card]) -> list[Card]:
    """Sort cards by suit then rank for display."""
    return sorted(cards, key=lambda c: (c.suit, c.rank))


def _format_card(card: Card) -> str:
    suit_colors = {
        Suit.HEARTS: "\033[91m",
        Suit.DIAMONDS: "\033[91m",
        Suit.SPADES: "\033[0m",
        Suit.CLUBS: "\033[0m",
    }
    reset = "\033[0m"
    return f"{suit_colors[card.suit]}{card}{reset}"


def _format_hand(
    cards: list[Card], highlight: set[Card] | None = None
) -> str:
    """Format cards with index numbers for selection."""
    parts = []
    for i, card in enumerate(cards):
        label = _format_card(card)
        if highlight and card in highlight:
            label = f"\033[1m{label}\033[0m"
        parts.append(f"[{i}] {label}")
    return "  ".join(parts)


def _read_choices(
    prompt: str,
    cards: list[Card],
    count: int,
    valid: set[Card] | None = None,
) -> list[Card]:
    """Prompt the user to pick card(s) by index."""
    while True:
        try:
            raw = input(prompt).strip()
            if count == 1:
                indices = [int(raw)]
            else:
                indices = [int(x) for x in raw.replace(",", " ").split()]
            if len(indices) != count:
                print(f"Please select exactly {count} card(s).")
                continue
            if any(i < 0 or i >= len(cards) for i in indices):
                print(f"Invalid index. Use 0-{len(cards) - 1}.")
                continue
            if len(set(indices)) != count:
                print("No duplicate selections.")
                continue
            selected = [cards[i] for i in indices]
            if valid and not all(c in valid for c in selected):
                bad = [c for c in selected if c not in valid]
                print(f"Illegal choice: {', '.join(str(c) for c in bad)}")
                continue
            return selected
        except (ValueError, IndexError):
            print("Enter card number(s) separated by spaces.")
        except EOFError:
            raise SystemExit(1)


class HumanBot(Bot):
    """Interactive human player via terminal input."""

    def pass_cards(self, view: PassView) -> list[Card]:
        """Prompt the human to select 3 cards to pass."""
        direction = view.pass_direction
        hand = _sort_hand(view.hand)
        print()
        print(f"=== Passing {direction.value} ===")
        print(f"Scores: {view.cumulative_scores}")
        print(f"Your hand: {_format_hand(hand)}")
        return _read_choices(
            "Choose 3 cards to pass (e.g. 0 5 12): ",
            hand, 3,
        )

    def play_card(self, view: PlayerView) -> Card:
        """Prompt the human to select a card to play."""
        hand = _sort_hand(view.hand)
        legal = set(view.legal_plays)

        print()
        if view.trick_so_far:
            played_str = "  ".join(
                f"P{p}: {_format_card(c)}"
                for p, c in view.trick_so_far
            )
            print(f"  Trick so far: {played_str}")
        else:
            print("  You lead this trick.")

        pts = view.points_taken_by_player
        print(f"  Points taken: {pts}")
        if view.hearts_broken:
            print("  Hearts are broken.")

        print(f"  Your hand: {_format_hand(hand, legal)}")

        legal_indices = [
            i for i, c in enumerate(hand) if c in legal
        ]
        legal_str = ", ".join(str(i) for i in legal_indices)
        print(f"  Legal plays: [{legal_str}]")

        selected = _read_choices(
            "  Play card: ", hand, 1, valid=legal,
        )
        return selected[0]

    def on_trick_complete(self, result: TrickResult) -> None:
        """Show trick result."""
        cards_str = "  ".join(
            f"P{p}: {_format_card(c)}"
            for p, c in result.cards_played
        )
        pts = f" ({result.points} pts)" if result.points else ""
        print(
            f"  -> P{result.winner} wins trick "
            f"{result.trick_number}: {cards_str}{pts}"
        )

    def on_round_complete(self, result: RoundResult) -> None:
        """Show round summary."""
        print()
        print(f"--- Round complete ---")
        print(f"  Scores this round: {result.scores}")
        if result.shoot_the_moon:
            print(
                f"  P{result.moon_shooter} shot the moon!"
            )
