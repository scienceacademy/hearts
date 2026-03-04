"""A heuristic Hearts bot — the bar students should aim to beat."""

from __future__ import annotations

from hearts.bot import Bot
from hearts.state import PassView, PlayerView
from hearts.types import Card, Suit


class RuleBot(Bot):
    """Heuristic strategy: avoid points, dump dangerous cards.

    Passing: pass Q/K/A of spades and highest hearts first,
    then highest remaining cards.

    Playing: duck (play low) when possible, dump high point
    cards when safe, void suits early.
    """

    def pass_cards(self, view: PassView) -> list[Card]:
        """Pass dangerous cards: Q/K/A♠, then highest hearts."""
        hand = sorted(view.hand, key=_pass_priority, reverse=True)
        return hand[:3]

    def play_card(self, view: PlayerView) -> Card:
        """Play using heuristic rules."""
        legal = view.legal_plays
        if len(legal) == 1:
            return legal[0]

        trick = view.trick_so_far

        # Leading
        if not trick:
            return _pick_lead(legal, view)

        lead_suit = view.lead_suit
        following = any(c.suit == lead_suit for c in legal)

        if following:
            return _pick_follow(legal, trick, lead_suit)

        # Can't follow — dump worst cards
        return _pick_dump(legal)


def _pass_priority(card: Card) -> int:
    """Higher value = more eager to pass."""
    if card.suit == Suit.SPADES and card.rank == 12:
        return 200  # Q♠ — always pass
    if card.suit == Suit.SPADES and card.rank >= 13:
        return 100 + card.rank  # K♠, A♠
    if card.suit == Suit.HEARTS:
        return 50 + card.rank  # high hearts next
    return card.rank  # otherwise by rank


def _pick_lead(
    legal: list[Card], view: PlayerView
) -> Card:
    """Choose a card to lead with."""
    # Lead low non-hearts to avoid taking points
    non_hearts = [c for c in legal if c.suit != Suit.HEARTS]
    candidates = non_hearts if non_hearts else legal

    # Prefer leading from a short suit (more likely to void it)
    suit_counts: dict[Suit, int] = {}
    for c in view.hand:
        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1

    # Among candidates pick lowest card, preferring short suits
    return min(
        candidates,
        key=lambda c: (suit_counts.get(c.suit, 0), c.rank),
    )


def _pick_follow(
    legal: list[Card],
    trick: list[tuple[int, Card]],
    lead_suit: Suit | None,
) -> Card:
    """Play when following suit."""
    # Find the highest card of lead suit played so far
    lead_cards = [
        c for _, c in trick
        if lead_suit and c.suit == lead_suit
    ]
    highest_played = max(
        (c.rank for c in lead_cards), default=0
    )

    # Try to duck under the current winner
    under = [c for c in legal if c.rank < highest_played]
    if under:
        # Play highest card that still ducks
        return max(under, key=lambda c: c.rank)

    # Can't duck — play lowest to minimize future risk
    return min(legal, key=lambda c: c.rank)


def _pick_dump(legal: list[Card]) -> Card:
    """Dump the worst card when unable to follow suit."""
    # Dump Q♠ first
    qs = Card(Suit.SPADES, 12)
    if qs in legal:
        return qs

    # Then dump highest hearts
    hearts = [c for c in legal if c.suit == Suit.HEARTS]
    if hearts:
        return max(hearts, key=lambda c: c.rank)

    # Then dump A♠ or K♠
    for rank in (14, 13):
        s = Card(Suit.SPADES, rank)
        if s in legal:
            return s

    # Otherwise dump highest card
    return max(legal, key=lambda c: c.rank)
