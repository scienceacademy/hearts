"""Rules engine for Hearts: legal move validation."""

from __future__ import annotations

from hearts.types import Card, Suit, card_points


TWO_OF_CLUBS = Card(Suit.CLUBS, 2)


def _is_point_card(card: Card) -> bool:
    return card_points(card) > 0


def get_legal_plays(
    hand: set[Card],
    trick_cards: list[tuple[int, Card]],
    lead_suit: Suit | None,
    hearts_broken: bool,
    is_first_trick: bool,
) -> list[Card]:
    """Return the list of legal cards to play.

    Args:
        hand: the player's current cards
        trick_cards: cards already played in this trick
        lead_suit: suit of the first card played this trick
        hearts_broken: whether hearts have been broken
        is_first_trick: whether this is the first trick of the round
    """
    if not hand:
        return []

    # First card of the entire round: must play 2♣
    if is_first_trick and not trick_cards:
        if TWO_OF_CLUBS in hand:
            return [TWO_OF_CLUBS]

    # Following suit: must play lead suit if possible
    if lead_suit is not None:
        follow = [c for c in hand if c.suit == lead_suit]
        if follow:
            return sorted(follow)
        # Can't follow suit — can play anything, but first trick
        # restricts point cards
        if is_first_trick:
            non_point = [c for c in hand if not _is_point_card(c)]
            if non_point:
                return sorted(non_point)
        return sorted(hand)

    # Leading a trick
    if is_first_trick:
        # Should not happen (2♣ always leads first trick),
        # but handle defensively
        return sorted(hand)

    if not hearts_broken:
        non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
        if non_hearts:
            return sorted(non_hearts)
    return sorted(hand)


def validate_play(
    card: Card,
    hand: set[Card],
    trick_cards: list[tuple[int, Card]],
    lead_suit: Suit | None,
    hearts_broken: bool,
    is_first_trick: bool,
) -> bool:
    """Return True if playing this card is legal."""
    if card not in hand:
        return False
    legal = get_legal_plays(
        hand, trick_cards, lead_suit,
        hearts_broken, is_first_trick,
    )
    return card in legal


def validate_pass(cards: list[Card], hand: set[Card]) -> bool:
    """Validate a pass selection: exactly 3 unique cards from hand."""
    if len(cards) != 3:
        return False
    if len(set(cards)) != 3:
        return False
    return all(c in hand for c in cards)
