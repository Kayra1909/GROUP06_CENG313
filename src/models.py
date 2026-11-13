"""Agents (policies), evaluation harness and the win-probability baseline.

In a simulation competition the "model" that gets submitted is a *policy*:
a function ``obs -> list[int]`` that picks among the legal options the
engine offers. This module contains

* ``RandomPolicy`` / ``FirstPolicy``  - reference opponents,
* ``HeuristicPolicy``                 - our rule-based baseline agent,
* evaluation helpers (win rate, Wilson CI, Bradley-Terry rating),
* ``train_value_model``               - a logistic-regression win-probability
  model trained on simulated games (groundwork for search in Assignment 2).

The policy part only uses the Python standard library so this file can be
copied unchanged into the Kaggle submission bundle (see
``utils.build_submission``). Heavy libraries are imported inside functions.
"""
from __future__ import annotations

import json
import random

# ---------------------------------------------------------------------------
# Engine enums (subset of cg/api.py)
# ---------------------------------------------------------------------------
AREA_DECK, AREA_HAND, AREA_DISCARD, AREA_ACTIVE, AREA_BENCH, AREA_PRIZE, AREA_STADIUM = 1, 2, 3, 4, 5, 6, 7
AREA_LOOKING = 12

OPT_NUMBER, OPT_YES, OPT_NO, OPT_CARD = 0, 1, 2, 3
OPT_ENERGY, OPT_PLAY, OPT_ATTACH, OPT_EVOLVE = 6, 7, 8, 9
OPT_ABILITY, OPT_DISCARD, OPT_RETREAT, OPT_ATTACK, OPT_END = 10, 11, 12, 13, 14

SEL_MAIN, SEL_CARD, SEL_ATTACHED_CARD, SEL_CARD_OR_ATTACHED = 0, 1, 2, 3
SEL_ENERGY, SEL_SKILL, SEL_ATTACK, SEL_EVOLVE, SEL_COUNT, SEL_YES_NO = 4, 5, 6, 7, 8, 9

CT_POKEMON, CT_ITEM, CT_TOOL, CT_SUPPORTER, CT_STADIUM, CT_BASIC_ENERGY, CT_SPECIAL_ENERGY = range(7)

# SelectContext groups
CTX_SETUP_ACTIVE, CTX_SETUP_BENCH, CTX_SWITCH, CTX_TO_ACTIVE = 1, 2, 3, 4
CTX_HEAL = {16, 17}                                   # remove damage counters / heal
CTX_HARM = {13, 14, 15, 25}                           # damage counters / damage / effect target
CTX_ATTACH_TARGET = {21, 22}
CTX_TAKE = {2, 5, 6, 7, 19, 24}                       # bench, to field, to hand, evolves-to, look
CTX_GIVE = {8, 9, 10, 11, 20, 26, 27, 29}             # discard, to deck, to prize, devolve ...

MAX_ACTIONS_PER_TURN = 30   # safety valve against action loops


def missing_energy(attached: list[int], cost: list[int]) -> int:
    """How many more Energy a Pokémon needs to pay ``cost``.

    Energy types follow the engine enum: 0 colorless, 10 rainbow (any type),
    11 Team Rocket (counts as Psychic or Darkness).
    """
    pool = list(attached)
    missing = 0
    for need in (c for c in cost if c != 0):
        for cand in (need, 10, 11 if need in (5, 7) else None):
            if cand is not None and cand in pool:
                pool.remove(cand)
                break
        else:
            missing += 1
    colorless = sum(1 for c in cost if c == 0)
    return missing + max(0, colorless - len(pool))


def validate_selection(choice, sel: dict, rng: random.Random) -> list[int]:
    """Make sure an answer is legal: distinct ints in range, size in [min, max].

    An illegal answer loses the whole match, so every policy goes through this.
    """
    n = len(sel.get("option") or [])
    lo, hi = int(sel.get("minCount", 0)), int(sel.get("maxCount", 0))
    hi = min(hi, n)
    out: list[int] = []
    if isinstance(choice, (list, tuple)):
        for c in choice:
            if isinstance(c, int) and 0 <= c < n and c not in out:
                out.append(c)
    out = out[:hi]
    if len(out) < lo:
        rest = [i for i in range(n) if i not in out]
        out += rest[: lo - len(out)]
    return out


class Policy:
    """Base class: handles the deck request and answer validation."""

    name = "policy"

    def __init__(self, deck: list[int], seed: int | None = None):
        self.deck = list(deck)
        self.rng = random.Random(seed)

    def __call__(self, obs) -> list[int]:
        sel = obs.get("select") if hasattr(obs, "get") else obs["select"]
        if sel is None:
            return list(self.deck)
        try:
            choice = self.choose(obs, sel)
        except Exception:  # never crash: an exception also loses the match
            choice = None
        return validate_selection(choice, sel, self.rng)

    def choose(self, obs, sel) -> list[int]:
        raise NotImplementedError


class RandomPolicy(Policy):
    """Uniformly random legal answer (same behaviour as the built-in 'random')."""

    name = "random"

    def choose(self, obs, sel):
        return self.rng.sample(range(len(sel["option"])), sel["maxCount"])


class FirstPolicy(Policy):
    """Always takes the first options (same behaviour as the built-in 'first')."""

    name = "first"

    def choose(self, obs, sel):
        return list(range(sel["maxCount"]))


class HeuristicPolicy(Policy):
    """Rule-based baseline.

    Main-phase priorities (higher score is played first; the engine returns to
    the main menu after every action, so the agent "builds" its turn greedily):

        evolve 95 > bench a Basic 90 > energy that brings an attack closer 80-90
        > item 75 > supporter 72 > ability 66 > stadium 62 > tool 58
        > retreat to a ready Pokémon 55 > other attachments 45
        > attack 20-50 (more expected damage = higher) > end turn 0

    Attacking ends the turn, so it is scored below every set-up action. Damage
    estimates come from the cleaned card data (weakness x2, resistance -30).
    Sub-decisions (search, discard, switch, targets ...) are scored with a
    simple card-value function.

    Tunable switches (``params``), chosen on training games in the notebook:
        retreat           - allow retreating to a ready Benched Pokémon
        starter           - "hp": start with the bulkiest Basic,
                            "evolution": prefer Basics that our deck can evolve
        evo_aware_energy  - when attaching Energy, aim at the attack of the
                            Pokémon's evolution (Energy stays after evolving)
    """

    name = "heuristic"
    DEFAULT_PARAMS = {"retreat": True, "starter": "hp", "evo_aware_energy": False}

    def __init__(self, card_db: dict, deck: list[int], seed: int | None = None, **params):
        super().__init__(deck, seed)
        self.cards = {int(k): v for k, v in card_db["cards"].items()}
        self.attacks = {int(k): v for k, v in card_db["attacks"].items()}
        self.params = {**self.DEFAULT_PARAMS, **params}
        # name -> card ids in our deck that evolve from it
        self.evolutions: dict[str, list[int]] = {}
        for cid in set(self.deck):
            frm = self.card(cid).get("evolves_from")
            if frm:
                self.evolutions.setdefault(frm, []).append(cid)

    # ----- small helpers -------------------------------------------------
    def card(self, cid) -> dict:
        return self.cards.get(int(cid), {}) if cid is not None else {}

    @staticmethod
    def _me(obs):
        cur = obs["current"]
        return cur, cur["yourIndex"], cur["players"][cur["yourIndex"]], cur["players"][1 - cur["yourIndex"]]

    @staticmethod
    def _active(ps):
        act = ps.get("active") or []
        return act[0] if act else None

    def _in_play(self, ps):
        return [p for p in (ps.get("active") or []) + (ps.get("bench") or []) if p]

    def damage(self, attacker: dict, attack_id: int, defender: dict | None) -> float:
        """Expected damage of ``attack_id`` from ``attacker`` to ``defender``."""
        dmg = float(self.attacks.get(int(attack_id), {}).get("damage", 0))
        if dmg <= 0 or not defender:
            return max(dmg, 0.0)
        a, d = self.card(attacker["id"]), self.card(defender["id"])
        if d.get("weakness") is not None and d.get("weakness") == a.get("energy_type"):
            dmg *= 2
        if d.get("resistance") is not None and d.get("resistance") == a.get("energy_type"):
            dmg -= 30
        return max(dmg, 0.0)

    def goal_attack(self, mon: dict, defender=None, evo_aware: bool = False):
        """The attack a Pokémon is working towards: the highest-damage one
        (optionally including the attacks of its evolutions in our deck)."""
        forms = [mon["id"]]
        if evo_aware:
            forms += self.evolutions.get(self.card(mon["id"]).get("name"), [])
        best = None
        for cid in forms:
            for aid in self.card(cid).get("attacks", []):
                dmg = self.damage({"id": cid}, aid, defender)
                if best is None or dmg > best[1]:
                    best = (aid, dmg)
        return best

    def readiness(self, mon: dict, defender=None) -> float:
        """Higher = more useful as Active Pokémon right now."""
        best_ready = 0.0
        for aid in self.card(mon["id"]).get("attacks", []):
            cost = self.attacks.get(aid, {}).get("cost", [])
            if missing_energy(mon.get("energies") or [], cost) == 0:
                best_ready = max(best_ready, self.damage(mon, aid, defender) + 1)
        return (100 if best_ready > 0 else 0) + best_ready + mon.get("hp", 0) / 10

    def card_value(self, cid, me_state) -> float:
        """How much we want a card in hand (used for search and discard)."""
        c = self.card(cid)
        ct = c.get("card_type")
        in_play = self._in_play(me_state)
        if ct == CT_POKEMON:
            if c.get("basic"):
                v = 6 + (2 if len(in_play) <= 2 else 0)
            else:
                names = {self.card(m["id"]).get("name") for m in in_play}
                v = 8 if c.get("evolves_from") in names else 4
            return v + (1 if c.get("ex") or c.get("mega_ex") else 0)
        if ct in (CT_BASIC_ENERGY, CT_SPECIAL_ENERGY):
            for m in in_play:
                goal = self.goal_attack(m)
                cost = self.attacks.get(goal[0], {}).get("cost", []) if goal else []
                if missing_energy(m.get("energies") or [], cost) > 0:
                    return 5
            return 3
        return {CT_SUPPORTER: 4.5, CT_ITEM: 4, CT_TOOL: 3, CT_STADIUM: 2}.get(ct, 1)

    def resolve(self, obs, opt):
        """Return (card-or-Pokémon dict, owner index) that an option points to."""
        cur, me, _, _ = self._me(obs)
        owner = opt.get("playerIndex")
        owner = me if owner is None else owner
        ps = cur["players"][owner]
        area, idx = opt.get("area"), opt.get("index")
        table = {
            AREA_HAND: ps.get("hand") or [], AREA_DECK: obs["select"].get("deck") or [],
            AREA_DISCARD: ps.get("discard") or [], AREA_ACTIVE: ps.get("active") or [],
            AREA_BENCH: ps.get("bench") or [], AREA_PRIZE: ps.get("prize") or [],
            AREA_LOOKING: cur.get("looking") or [], AREA_STADIUM: cur.get("stadium") or [],
        }.get(area, [])
        if isinstance(idx, int) and 0 <= idx < len(table):
            return table[idx], owner
        return None, owner

    # ----- decisions -----------------------------------------------------
    def choose(self, obs, sel):
        st = sel["type"]
        if st == SEL_MAIN:
            return self.main_phase(obs, sel)
        opts = sel["option"]
        if st == SEL_YES_NO:  # go first, mulligan, activate effects ... always yes
            yes = [i for i, o in enumerate(opts) if o["type"] == OPT_YES]
            return yes[:1] or [0]
        if st == SEL_COUNT:   # draw as many / place as many counters as allowed
            return [max(range(len(opts)), key=lambda i: opts[i].get("number", 0))]
        if st == SEL_ENERGY:
            return self.pay_energy(sel)
        if st in (SEL_CARD, SEL_ATTACHED_CARD, SEL_CARD_OR_ATTACHED):
            return self.pick_cards(obs, sel)
        if st == SEL_ATTACK:
            cur, me, mine, opp = self._me(obs)
            atk = self._active(mine) or {"id": 0}
            return [max(range(len(opts)), key=lambda i: self.damage(atk, opts[i].get("attackId", 0), self._active(opp)))]
        return list(range(sel["maxCount"]))

    def main_phase(self, obs, sel):
        cur, me, mine, opp = self._me(obs)
        my_active, opp_active = self._active(mine), self._active(opp)
        opts = sel["option"]
        can_attack = any(o["type"] == OPT_ATTACK for o in opts)
        too_long = cur.get("turnActionCount", 0) > MAX_ACTIONS_PER_TURN
        scores = []
        for o in opts:
            t = o["type"]
            if t == OPT_END:
                s = 0.0
            elif t == OPT_ATTACK:
                dmg = self.damage(my_active, o["attackId"], opp_active) if my_active else 0
                s = 20 + min(dmg, 300) / 10 if dmg > 0 else 10
            elif too_long:
                s = -100.0
            elif t == OPT_EVOLVE:
                s = 95.0
            elif t == OPT_PLAY:
                card, _ = self.resolve(obs, {"area": AREA_HAND, "index": o.get("index")})
                s = self.play_score(card)
            elif t == OPT_ATTACH:
                s = self.attach_score(obs, o)
            elif t == OPT_ABILITY:
                s = 66.0
            elif t == OPT_RETREAT:
                s = self.retreat_score(cur, mine, opp_active, can_attack)
            else:  # discarding our own card in play is rarely good
                s = -10.0
            scores.append(s)
        return [max(range(len(opts)), key=lambda i: (scores[i], -i))]

    def play_score(self, card) -> float:
        c = self.card(card["id"]) if card else {}
        ct = c.get("card_type")
        if ct == CT_POKEMON:
            return 90.0 if c.get("basic") else 50.0
        return {CT_ITEM: 75.0, CT_SUPPORTER: 72.0, CT_STADIUM: 62.0, CT_TOOL: 58.0}.get(ct, 45.0)

    def attach_score(self, obs, o) -> float:
        cur, me, mine, opp = self._me(obs)
        card, _ = self.resolve(obs, {"area": o.get("area"), "index": o.get("index")})
        target, _ = self.resolve(obs, {"area": o.get("inPlayArea"), "index": o.get("inPlayIndex"), "playerIndex": me})
        if not card or not target:
            return 40.0
        c = self.card(card["id"])
        is_active = o.get("inPlayArea") == AREA_ACTIVE
        if c.get("card_type") == CT_TOOL:
            return 58.0 + (3 if is_active else 0)
        if c.get("card_type") not in (CT_BASIC_ENERGY, CT_SPECIAL_ENERGY):
            return 50.0
        goal = self.goal_attack(target, self._active(opp), evo_aware=self.params["evo_aware_energy"])
        if not goal:
            return 44.0
        cost = self.attacks.get(goal[0], {}).get("cost", [])
        have = list(target.get("energies") or [])
        gain = missing_energy(have, cost) - missing_energy(have + [c.get("energy_type", 0)], cost)
        if gain > 0:
            return 80.0 + (8 if is_active else 0) + min(goal[1], 300) / 150
        return 45.0 + (2 if is_active else 0)

    def retreat_score(self, cur, mine, opp_active, can_attack) -> float:
        act = self._active(mine)
        if not self.params["retreat"] or cur.get("retreated") or not act or can_attack:
            return -20.0
        bench = mine.get("bench") or []
        if any(self.readiness(b, opp_active) >= 100 for b in bench):
            return 55.0
        if act.get("hp", 0) <= 0.3 * act.get("maxHp", 1) and any(b.get("hp", 0) > act.get("hp", 0) for b in bench):
            return 52.0
        return -20.0

    def pay_energy(self, sel):
        """Discard as few Energy as possible while covering the remaining cost."""
        opts = sel["option"]
        need = sel.get("remainEnergyCost", 0) or 0
        order = sorted(range(len(opts)), key=lambda i: opts[i].get("count", 1))
        out, paid = [], 0
        for i in order:
            if paid >= need and len(out) >= sel["minCount"]:
                break
            out.append(i)
            paid += opts[i].get("count", 1)
        return out

    def pick_cards(self, obs, sel):
        cur, me, mine, opp = self._me(obs)
        ctx, opts = sel["context"], sel["option"]
        opp_active = self._active(opp)
        values = []
        for o in opts:
            card, owner = self.resolve(obs, o)
            if card is None:
                values.append(0.0)
                continue
            in_play = o.get("area") in (AREA_ACTIVE, AREA_BENCH) and "hp" in card
            if ctx == CTX_SETUP_ACTIVE:
                c = self.card(card["id"])
                cheapest = min((len(self.attacks.get(a, {}).get("cost", [])) for a in c.get("attacks", [])), default=5)
                v = c.get("hp", 0) / 10 - 5 * cheapest - (10 if c.get("ex") or c.get("mega_ex") else 0)
                if self.params["starter"] == "evolution" and c.get("name") in self.evolutions:
                    v += 15
            elif in_play and owner != me:          # opponent's Pokémon: hit the weakest
                v = -card.get("hp", 0) + (40 if self.card(card["id"]).get("ex") else 0)
                if ctx in CTX_HEAL:
                    v = -v
            elif in_play:                          # our own Pokémon
                if ctx in CTX_HEAL:
                    v = card.get("maxHp", 0) - card.get("hp", 0)
                elif ctx in CTX_HARM:
                    v = -self.readiness(card, opp_active)
                elif ctx in CTX_ATTACH_TARGET:
                    goal = self.goal_attack(card, opp_active)
                    cost = self.attacks.get(goal[0], {}).get("cost", []) if goal else []
                    v = 10 * missing_energy(card.get("energies") or [], cost) + (5 if o.get("area") == AREA_ACTIVE else 0)
                else:                              # switch / to active / evolve ...
                    v = self.readiness(card, opp_active)
            else:
                v = self.card_value(card.get("id"), mine)
            values.append(v)

        hi, lo = sel["maxCount"], sel["minCount"]
        if ctx in CTX_GIVE:   # giving cards away: as few as possible, least valuable first
            order = sorted(range(len(opts)), key=lambda i: values[i])
            return order[:lo]
        order = sorted(range(len(opts)), key=lambda i: -values[i])
        k = hi if ctx in CTX_TAKE or ctx in CTX_HEAL else max(lo, min(1, hi))
        return order[:k]


def load_card_db_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Evaluation (needs kaggle-environments, pandas, numpy)
# ---------------------------------------------------------------------------

def as_kaggle_agent(policy):
    """Wrap a policy object in a plain function (what kaggle-environments expects)."""
    def agent(obs):
        return policy(obs)
    agent.__name__ = getattr(policy, "name", "agent")
    return agent


def evaluate(policy, opponents: dict, n_games: int, verbose: bool = True):
    """Play ``policy`` against every opponent (seats alternate) and return records.

    Returns (games, decisions, snapshots) DataFrames concatenated over opponents.
    """
    import pandas as pd
    from src.data import simulate_games

    parts, start = [], 0
    for opp_name, opp in opponents.items():
        if verbose:
            print(f"{policy.name} vs {opp_name}:")
        res = simulate_games(as_kaggle_agent(policy), as_kaggle_agent(opp), n_games,
                             names=(policy.name, opp_name), start_id=start, verbose=verbose)
        parts.append(res)
        start += n_games
    return tuple(pd.concat([p[i] for p in parts], ignore_index=True) for i in range(3))


def win_rate_table(games, agent: str):
    """Win rate (+ Wilson 95% CI) of ``agent`` against each opponent."""
    import pandas as pd
    from src.utils import wilson_ci

    g = games[games["agent"] == agent]
    rows = []
    for opp, d in g.groupby("opponent"):
        wins, n = int(d["won"].sum()), len(d)
        lo, hi = wilson_ci(wins, n)
        rows.append({"opponent": opp, "games": n, "wins": wins, "win_rate": wins / n,
                     "ci_low": lo, "ci_high": hi,
                     "win_rate_going_first": d.loc[d["went_first"], "won"].mean(),
                     "win_rate_going_second": d.loc[~d["went_first"], "won"].mean()})
    return pd.DataFrame(rows)


def bradley_terry(games, anchor: str | None = None, anchor_value: float = 600.0, iters: int = 500):
    """Bradley-Terry strengths from pairwise results, on an Elo-like scale.

    Kaggle's final leaderboard is a Bradley-Terry tournament, so this is the
    closest local analogue of the competition metric. Draws count as half a
    win. The scale is 400*log10(strength); ``anchor`` is pinned to
    ``anchor_value`` (600 = Kaggle's starting mu).
    """
    import numpy as np
    import pandas as pd

    agents = sorted(set(games["agent"]) | set(games["opponent"]))
    idx = {a: i for i, a in enumerate(agents)}
    k = len(agents)
    wins = np.zeros((k, k))
    for r in games.itertuples():
        wins[idx[r.agent], idx[r.opponent]] += r.won + 0.5 * r.draw
    n = wins + wins.T
    p = np.ones(k)
    for _ in range(iters):  # Hunter's MM algorithm
        w = wins.sum(axis=1) + 1e-3
        denom = np.array([sum(n[i, j] / (p[i] + p[j]) for j in range(k) if j != i) for i in range(k)])
        p = w / np.maximum(denom, 1e-12)
        p /= np.exp(np.log(p).mean())
    rating = 400 * np.log10(p)
    ref = rating[idx[anchor]] if anchor in idx else rating.mean()
    return pd.Series(rating - ref + anchor_value, index=agents, name="bt_rating").sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Win-probability model (simple supervised baseline)
# ---------------------------------------------------------------------------

VALUE_FEATURES = [
    "turn", "went_first",
    "diff_prizes_left", "diff_n_pokemon", "diff_hp_in_play", "diff_energy_in_play",
    "diff_active_hp", "diff_active_energy", "diff_hand", "diff_deck",
]


def train_value_model(snaps, features=None, test_size: float = 0.2, seed: int = 42):
    """Logistic regression: P(player to move wins | board at start of turn).

    The split is *grouped by game*: all turns of a game go to the same side,
    otherwise near-identical consecutive positions would leak between train
    and validation and inflate the score.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = features or VALUE_FEATURES
    X = snaps[features].astype(float).to_numpy()
    y = snaps["won"].to_numpy()
    groups = snaps["game_id"].to_numpy()
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X[tr], y[tr])
    p = model.predict_proba(X[va])[:, 1]
    # reference points: a constant predictor and a one-feature rule (prize difference)
    const = np.full(len(va), y[tr].mean())
    prize_rule = 1 / (1 + np.exp(snaps["diff_prizes_left"].to_numpy()[va]))  # fewer prizes left = ahead
    metrics = {
        "train_rows": len(tr), "valid_rows": len(va),
        "train_games": len(set(groups[tr])), "valid_games": len(set(groups[va])),
        "auc": roc_auc_score(y[va], p), "log_loss": log_loss(y[va], p), "accuracy": accuracy_score(y[va], p > 0.5),
        "auc_prize_rule": roc_auc_score(y[va], prize_rule), "log_loss_constant": log_loss(y[va], const),
    }
    return model, metrics, (tr, va)
