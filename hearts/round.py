"""Round engine: executes a single 13-trick round of Hearts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hearts.rules import TWO_OF_CLUBS, get_legal_plays
from hearts.state import (
    PassDirection,
    PassView,
    PlayerView,
    RoundResult,
    RoundState,
    TrickResult,
    TrickState,
)
from hearts.types import Card, Suit, card_points

if TYPE_CHECKING:
    from hearts.bot import Bot


class Round:
    """Executes a single round of Hearts.

    Args:
        bots: 4 bot instances (one per player seat)
        hands: 4 sets of cards (dealt hands)
        pass_direction: which direction to pass cards
        cumulative_scores: game-level scores entering this round
        round_number: which round this is (0-indexed)
    """

    def __init__(
        self,
        bots: list[Bot],
        hands: list[set[Card]],
        pass_direction: PassDirection,
        cumulative_scores: list[int] | None = None,
        round_number: int = 0,
    ) -> None:
        self._bots = bots
        self._pass_direction = pass_direction
        self._cumulative_scores = cumulative_scores or [0, 0, 0, 0]
        self._round_number = round_number
        self._hands_dealt = [set(h) for h in hands]
        self._state = RoundState(hands=[set(h) for h in hands])
        self._cards_played_this_round: list[tuple[int, int, Card]] = []
        self._tricks_won = [0, 0, 0, 0]
        self._trick_results: list[TrickResult] = []
        self._cards_received: list[list[Card]] = [
            [] for _ in range(4)
        ]

    def play(self) -> RoundResult:
        """Execute the full round and return the result."""
        self._do_passing()
        self._find_first_leader()

        for trick_num in range(1, 14):
            self._play_trick(trick_num)

        # Check raw points to detect shoot the moon
        raw_points = [
            sum(card_points(c) for c in taken)
            for taken in self._state.cards_taken
        ]
        moon_shooter = None
        shoot = False
        for i, pts in enumerate(raw_points):
            if pts == 26:
                moon_shooter = i
                shoot = True
                break
        scores = self._state.scores

        result = RoundResult(
            scores=scores,
            tricks=self._trick_results,
            shoot_the_moon=shoot,
            moon_shooter=moon_shooter,
            pass_direction=self._pass_direction,
            hands_dealt=self._hands_dealt,
        )
        for bot in self._bots:
            bot.on_round_complete(result)
        return result

    def _do_passing(self) -> None:
        if self._pass_direction == PassDirection.NONE:
            return

        passed: list[list[Card]] = [[] for _ in range(4)]
        for i, bot in enumerate(self._bots):
            view = PassView(
                hand=set(self._state.hands[i]),
                player_index=i,
                pass_direction=self._pass_direction,
                cumulative_scores=list(self._cumulative_scores),
                round_number=self._round_number,
            )
            cards = bot.pass_cards(view)
            if len(cards) != 3 or len(set(cards)) != 3:
                raise ValueError(
                    f"Bot {i} must pass exactly 3 unique cards"
                )
            for c in cards:
                if c not in self._state.hands[i]:
                    raise ValueError(
                        f"Bot {i} passed {c} not in hand"
                    )
            passed[i] = cards

        offsets = {
            PassDirection.LEFT: 1,
            PassDirection.RIGHT: 3,
            PassDirection.ACROSS: 2,
        }
        offset = offsets[self._pass_direction]

        for i in range(4):
            for c in passed[i]:
                self._state.hands[i].remove(c)

        for i in range(4):
            target = (i + offset) % 4
            self._state.hands[target].update(passed[i])
            self._cards_received[target] = list(passed[i])

    def _find_first_leader(self) -> None:
        for i in range(4):
            if TWO_OF_CLUBS in self._state.hands[i]:
                self._state.lead_player = i
                return

    def _play_trick(self, trick_number: int) -> None:
        self._state.current_trick = TrickState()
        is_first = trick_number == 1

        for turn in range(4):
            player = (self._state.lead_player + turn) % 4
            hand = self._state.hands[player]

            legal = get_legal_plays(
                hand,
                self._state.current_trick.cards_played,
                self._state.current_trick.lead_suit,
                self._state.hearts_broken,
                is_first,
            )

            view = PlayerView(
                hand=set(hand),
                player_index=player,
                trick_so_far=list(
                    self._state.current_trick.cards_played
                ),
                lead_suit=self._state.current_trick.lead_suit,
                hearts_broken=self._state.hearts_broken,
                is_first_trick=is_first,
                cards_played_this_round=list(
                    self._cards_played_this_round
                ),
                tricks_won_by_player=list(self._tricks_won),
                points_taken_by_player=[
                    sum(card_points(c) for c in taken)
                    for taken in self._state.cards_taken
                ],
                pass_direction=self._pass_direction,
                cards_received_in_pass=list(
                    self._cards_received[player]
                ),
                legal_plays=legal,
                cumulative_scores=list(self._cumulative_scores),
            )

            card = self._bots[player].play_card(view)
            if card not in legal:
                raise ValueError(
                    f"Bot {player} played illegal card {card}"
                )

            hand.remove(card)
            if turn == 0:
                self._state.current_trick.lead_suit = card.suit
            self._state.current_trick.cards_played.append(
                (player, card)
            )
            self._cards_played_this_round.append(
                (trick_number, player, card)
            )

            if card.suit == Suit.HEARTS and not self._state.hearts_broken:
                self._state.hearts_broken = True

        trick = self._state.current_trick
        winner = trick.winner()
        trick_cards = [c for _, c in trick.cards_played]
        pts = sum(card_points(c) for c in trick_cards)

        self._state.cards_taken[winner].extend(trick_cards)
        self._tricks_won[winner] += 1
        self._state.lead_player = winner

        result = TrickResult(
            cards_played=list(trick.cards_played),
            lead_suit=trick.lead_suit,
            winner=winner,
            points=pts,
            trick_number=trick_number,
        )
        self._trick_results.append(result)
        for bot in self._bots:
            bot.on_trick_complete(result)
