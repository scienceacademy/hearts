"""Interactive human player for terminal play."""

from __future__ import annotations
import os

from hearts.bot import Bot
from hearts.state import (
    PassDirection,
    PassView,
    PlayerView,
    RoundResult,
    TrickResult,
)
from hearts.types import Card, Suit, card_points

RESET = '\033[0m'
GREY = '\033[90m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
PURPLE = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
BGGREY = '\033[100m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def _sort_hand(cards: set[Card] | list[Card]) -> list[Card]:
    """Sort cards by suit then rank for display."""
    return sorted(cards, key=lambda c: (c.suit, c.rank))


def _format_card(card: Card) -> str:
    return f"{RED if card.suit in [Suit.HEARTS, Suit.DIAMONDS] else WHITE}{card}{RESET}" + (" " if card.rank != 10 else "")


def _format_hand(
    cards: list[Card], highlight: set[Card] | None = None
) -> str:
    """Format cards with index numbers for selection."""
    string = ""
    for i, card in enumerate(cards):
        label = _format_card(card)
        if highlight and card in highlight:
            string += f"[{GREEN}{i:2}{RESET}] {label}  "
        else:
            string += f"[{i:2}] {label}  "
        if i != len(cards) - 1:
            if card.suit != cards[i + 1].suit:
                string += "\n"
    return string


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
            clear()
            return selected
        except (ValueError, IndexError):
            print("Enter card number(s) separated by spaces.")
        except EOFError:
            raise SystemExit(1)


class HumanBot(Bot):
    """Interactive human player via terminal input."""
    lastwinner = 4
    history = [f" {GREY}Past tricks{RESET}", f" {GREY}will go here{RESET}"]

    def __init__(self, seat: int):
        self.seat = seat

    def pass_cards(self, view: PassView) -> list[Card]:
        """Prompt the human to select 3 cards to pass."""
        direction = view.pass_direction
        hand = _sort_hand(view.hand)

        if view.round_number == 0:
            print(f"{BOLD}=== Passing {direction.value} ==={RESET}")
        else:
            clear()
            print(f"{BOLD}=== Passing {direction.value} ==={RESET}")
            scores = [(i, n) for i, n in enumerate(view.cumulative_scores)]
            scores = sorted(scores, key=lambda x: x[1])
            print("Scores:")
            print(f"1st: {CYAN if scores[0][0] == self.seat else WHITE}P{scores[0][0]}{RESET} - {scores[0][1]}")
            print(f"2nd: {CYAN if scores[1][0] == self.seat else WHITE}P{scores[1][0]}{RESET} - {scores[1][1]}")
            print(f"3rd: {CYAN if scores[2][0] == self.seat else WHITE}P{scores[2][0]}{RESET} - {scores[2][1]}")
            print(f"4th: {CYAN if scores[3][0] == self.seat else WHITE}P{scores[3][0]}{RESET} - {scores[3][1]}\n")
        print(f"Your hand: \n{_format_hand(hand)}")
        return _read_choices(
            "Choose 3 cards to pass (e.g. 0 5 12): ",
            hand, 3,
        )

    def play_card(self, view: PlayerView) -> Card:
        """Prompt the human to select a card to play."""
        hand = _sort_hand(view.hand)
        legal = set(view.legal_plays)

        trick_so_far: list[str] = ["   "] * 4
        for p, c in view.trick_so_far:
            trick_so_far[p] = _format_card(c)
        print(
            f"{view.points_taken_by_player[0]:2} pts   {view.points_taken_by_player[1]:2} pts"
            f"     {CYAN if self.seat == 0 else RESET}P0{RESET}  {CYAN if self.seat == 1 else RESET}P1{RESET}  {CYAN if self.seat == 2 else RESET}P2{RESET}  {CYAN if self.seat == 3 else RESET}P3{RESET}\n"

            f"  {CYAN if self.seat == 0 else RESET}P0{RESET}       {CYAN if self.seat == 1 else RESET}P1{RESET}  "
            f"     {self.history[0] if len(self.history) > 0 else ''}\n"

             "   ╭───────╮   "
            f"     {self.history[1] if len(self.history) > 1 else ''}\n"

            f"   │{trick_so_far[0]} {trick_so_far[1]}│   "
            f"     {self.history[2] if len(self.history) > 2 else ''}\n"

             "   │       │   "
            f"     {self.history[3] if len(self.history) > 3 else ''}\n"

            f"   │{trick_so_far[3]} {trick_so_far[2]}│   "
            f"     {self.history[4] if len(self.history) > 4 else ''}\n"

             "   ╰───────╯   "
            f"     {self.history[5] if len(self.history) > 5 else ''}\n"

            f"  {CYAN if self.seat == 3 else RESET}P3{RESET}       {CYAN if self.seat == 2 else RESET}P2{RESET}  "
            f"     {self.history[6] if len(self.history) > 6 else ''}\n"

            f"{view.points_taken_by_player[3]:2} pts   {view.points_taken_by_player[2]:2} pts"
            f"     {self.history[7] if len(self.history) > 7 else ''}\n"
        )
        print(f"{CYAN if self.lastwinner == self.seat else RESET}P{self.lastwinner}{RESET} won trick {self.lasttrick} for {self.lastpoints} pts" if self.lastwinner != 4 else "")

        print(f"Your hand: \n{_format_hand(hand, legal)}")

        selected = _read_choices(
            "Card selection: ", hand, 1, valid=legal,
        )
        return selected[0]

    def on_trick_complete(self, result: TrickResult) -> None:
        """Show trick result."""
        self.lastwinner = result.winner
        self.lastpoints = result.points
        self.lasttrick = result.trick_number
        leader = result.cards_played[0][0]
        historystr = ""
        for i, c in sorted(result.cards_played, key=lambda x: x[0]):
            historystr += f"{UNDERLINE if i == result.winner else ''}{BOLD if i == leader else ''}{_format_card(c)} "
        self.history.insert(0, historystr)

    def on_round_complete(self, result: RoundResult) -> None:
        """Show round summary."""
        self.history = []
        print(f"{BOLD}=== Round complete! ==={RESET}")
        if result.shoot_the_moon:
            print(f"P{result.moon_shooter} has shot the moon!")
        print("")
        print('  '.join(f"{n:2}" for n in result.scores))
        print(', '.join(f'{CYAN if i == self.seat else GREY}P{i}{RESET}' for i in range(4)))
