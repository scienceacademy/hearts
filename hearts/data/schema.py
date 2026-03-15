"""Card index mapping and serialization for Hearts game data.

Cards are represented as integers 0-51 using the mapping:
    index = suit_ordinal * 13 + (rank - 2)

Suit ordinals: CLUBS=0, DIAMONDS=1, SPADES=2, HEARTS=3
Ranks: 2-14 (J=11, Q=12, K=13, A=14)

Examples: 0 = 2♣, 12 = A♣, 13 = 2♦, 37 = Q♠, 51 = A♥
"""

from __future__ import annotations

from hearts.state import PassDirection, PassView, PlayerView
from hearts.types import Card, Suit


def card_to_index(card: Card) -> int:
    """Convert a Card to its integer index (0-51)."""
    return int(card.suit) * 13 + (card.rank - 2)


def index_to_card(index: int) -> Card:
    """Convert an integer index (0-51) back to a Card."""
    suit = Suit(index // 13)
    rank = (index % 13) + 2
    return Card(suit, rank)


def serialize_player_view(view: PlayerView) -> dict:
    """Convert a PlayerView into a JSON-safe dict using card indices.

    The cards_played_this_round field is restructured from flat
    (trick_num, player_index, Card) tuples into a nested-by-trick format:
    list of tricks, each a list of [player_index, card_index] pairs.
    """
    # Group cards_played_this_round by trick number
    tricks_by_num: dict[int, list[list[int]]] = {}
    for trick_num, player_idx, card in view.cards_played_this_round:
        if trick_num not in tricks_by_num:
            tricks_by_num[trick_num] = []
        tricks_by_num[trick_num].append(
            [player_idx, card_to_index(card)]
        )
    # Convert to ordered list of tricks
    cards_played = [
        tricks_by_num[k]
        for k in sorted(tricks_by_num)
    ]

    return {
        "player_index": view.player_index,
        "trick_number": _current_trick_number(view),
        "hand": sorted(card_to_index(c) for c in view.hand),
        "legal_plays": sorted(
            card_to_index(c) for c in view.legal_plays
        ),
        "trick_so_far": [
            [pi, card_to_index(c)]
            for pi, c in view.trick_so_far
        ],
        "lead_suit": (
            view.lead_suit.name if view.lead_suit is not None else None
        ),
        "hearts_broken": view.hearts_broken,
        "is_first_trick": view.is_first_trick,
        "cards_played_this_round": cards_played,
        "tricks_won_by_player": list(view.tricks_won_by_player),
        "points_taken_by_player": list(view.points_taken_by_player),
        "pass_direction": view.pass_direction.name,
        "cards_received_in_pass": [
            card_to_index(c) for c in view.cards_received_in_pass
        ],
        "cumulative_scores": list(view.cumulative_scores),
        "round_number": _round_number_placeholder(),
    }


def _current_trick_number(view: PlayerView) -> int:
    """Infer the current trick number from cards_played_this_round."""
    if not view.cards_played_this_round:
        return 1
    last_trick_num = view.cards_played_this_round[-1][0]
    # If the current trick has cards in trick_so_far, we're in that trick
    # Otherwise we're starting the next trick
    if view.trick_so_far:
        return last_trick_num
    return last_trick_num + 1


def _round_number_placeholder() -> int:
    """Placeholder — round_number is set by the generator."""
    return 0


def deserialize_player_view(data: dict) -> PlayerView:
    """Reconstruct a PlayerView from a serialized dict.

    Flattens the nested trick structure back to
    (trick_num, player_index, Card) tuples.
    """
    cards_played_flat: list[tuple[int, int, Card]] = []
    for trick_idx, trick_plays in enumerate(
        data["cards_played_this_round"]
    ):
        trick_num = trick_idx + 1
        for player_idx, card_idx in trick_plays:
            cards_played_flat.append(
                (trick_num, player_idx, index_to_card(card_idx))
            )

    lead_suit = (
        Suit[data["lead_suit"]]
        if data["lead_suit"] is not None
        else None
    )

    return PlayerView(
        hand=set(index_to_card(i) for i in data["hand"]),
        player_index=data["player_index"],
        trick_so_far=[
            (pi, index_to_card(ci))
            for pi, ci in data["trick_so_far"]
        ],
        lead_suit=lead_suit,
        hearts_broken=data["hearts_broken"],
        is_first_trick=data["is_first_trick"],
        cards_played_this_round=cards_played_flat,
        tricks_won_by_player=list(data["tricks_won_by_player"]),
        points_taken_by_player=list(data["points_taken_by_player"]),
        pass_direction=PassDirection[data["pass_direction"]],
        cards_received_in_pass=[
            index_to_card(i) for i in data["cards_received_in_pass"]
        ],
        legal_plays=[
            index_to_card(i) for i in data["legal_plays"]
        ],
        cumulative_scores=list(data["cumulative_scores"]),
    )


def serialize_pass_view(view: PassView) -> dict:
    """Convert a PassView into a JSON-safe dict using card indices."""
    return {
        "player_index": view.player_index,
        "hand": sorted(card_to_index(c) for c in view.hand),
        "pass_direction": view.pass_direction.name,
        "cumulative_scores": list(view.cumulative_scores),
        "round_number": view.round_number,
    }


def deserialize_pass_view(data: dict) -> PassView:
    """Reconstruct a PassView from a serialized dict."""
    return PassView(
        hand=set(index_to_card(i) for i in data["hand"]),
        player_index=data["player_index"],
        pass_direction=PassDirection[data["pass_direction"]],
        cumulative_scores=list(data["cumulative_scores"]),
        round_number=data["round_number"],
    )
