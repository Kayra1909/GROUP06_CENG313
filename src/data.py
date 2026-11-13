"""Data loading, cleaning and simulated-game generation.

Two data sources are used in this project:

1. ``EN_Card_Data_R2_full.csv`` (competition data): one row per (card, move).
   ``load_raw_cards`` -> ``clean_moves`` -> ``card_level_table`` turn it into
   a tidy, typed, one-row-per-card table.
2. The official simulator (``cabt`` environment in ``kaggle-environments``):
   we play games locally and parse every game into tabular records
   (``simulate_games``). This is our "training data" for gameplay EDA and
   for the win-probability model.

Nothing in this module writes into ``data/`` except optional caches, so the
dataset itself never ends up in git.
"""
from __future__ import annotations

import ctypes
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_file(pattern: str, root: Path | str = DATA_DIR) -> Path:
    """Return the first file under ``root`` (recursive) that matches ``pattern``.

    The Kaggle zip nests the sample submission several folders deep, so we
    search instead of hard-coding paths.
    """
    matches = sorted(Path(root).rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' under '{root}'. Download the competition "
            "data into data/ first (see README)."
        )
    return matches[0]


def read_deck(path: Path | str | None = None) -> list[int]:
    """Read a 60-card deck (one card ID per line) - default: the sample deck."""
    path = Path(path) if path else find_file("deck.csv")
    ids = [int(x) for x in Path(path).read_text().split() if x.strip()]
    if len(ids) != 60:
        raise ValueError(f"A deck must contain 60 cards, got {len(ids)}")
    return ids


# ---------------------------------------------------------------------------
# Card CSV: loading and cleaning
# ---------------------------------------------------------------------------

COLUMN_NAMES = {
    "Card ID": "card_id",
    "Card Name": "name",
    "Expansion": "expansion",
    "Collection No.": "collection_no",
    "Stage (Pokémon)/Type (Energy and Trainer)": "stage",
    "Rule": "rule",
    "Category": "category",
    "Previous stage": "previous_stage",
    "HP": "hp",
    "Type": "type",
    "Weakness": "weakness",
    "Resistance (Type)": "resistance",
    "Retreat": "retreat",
    "Move Name": "move_name",
    "Cost": "cost",
    "Damage": "damage",
    "Effect Explanation": "effect",
    "Species": "species",
}

# {G} grass, {R} fire, {W} water, {L} lightning, {P} psychic, {F} fighting,
# {D} darkness, {M} metal, {N} dragon, {C} colorless.  "●" is also colorless.
TYPE_SYMBOLS = {
    "G": "Grass", "R": "Fire", "W": "Water", "L": "Lightning", "P": "Psychic",
    "F": "Fighting", "D": "Darkness", "M": "Metal", "N": "Dragon", "C": "Colorless",
    "A": "Any", "Team Rocket": "Team Rocket",
}
ENERGY_TYPE_CODES = {  # engine EnergyType enum (cg/api.py)
    0: "Colorless", 1: "Grass", 2: "Fire", 3: "Water", 4: "Lightning", 5: "Psychic",
    6: "Fighting", 7: "Darkness", 8: "Metal", 9: "Dragon", 10: "Rainbow", 11: "Team Rocket",
}
CARD_TYPE_CODES = {  # engine CardType enum
    0: "Pokemon", 1: "Item", 2: "Tool", 3: "Supporter", 4: "Stadium",
    5: "Basic Energy", 6: "Special Energy",
}


def load_raw_cards(path: Path | str | None = None) -> pd.DataFrame:
    """Load the English card CSV *as strings*.

    ``keep_default_na=False`` matters: the file uses the literal text ``n/a``
    for "not applicable" (e.g. HP of a Trainer) and an empty cell for "none"
    (e.g. a Pokémon without weakness). pandas would silently merge both into
    NaN, so we keep them apart and decide per column in ``clean_moves``.
    """
    path = Path(path) if path else find_file("EN_Card_Data*.csv")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _symbols(text: str) -> list[str]:
    """'{W}{W}●' -> ['Water', 'Water', 'Colorless']."""
    out = [TYPE_SYMBOLS.get(s, s) for s in re.findall(r"\{([^}]+)\}", text)]
    out += ["Colorless"] * text.count("●")
    return out


def parse_cost(cost: str) -> dict:
    """Energy cost string -> counts. 'n/a' (abilities, trainers) -> NaN."""
    if cost in ("n/a", ""):
        return {"cost_total": np.nan, "cost_colorless": np.nan, "cost_typed": np.nan}
    syms = _symbols(cost)
    colorless = sum(s == "Colorless" for s in syms)
    return {"cost_total": len(syms), "cost_colorless": colorless, "cost_typed": len(syms) - colorless}


def parse_damage(damage: str) -> dict:
    """'120' -> 120, '30×' -> (30, 'x'), '50+' -> (50, '+'), '90-' -> (90, '-')."""
    if damage in ("n/a", ""):
        return {"damage_base": np.nan, "damage_mod": "none"}
    m = re.match(r"^\s*(\d+)\s*([×x+\-]?)\s*$", damage)
    if not m:
        return {"damage_base": np.nan, "damage_mod": "unparsed"}
    mod = {"×": "x", "x": "x", "+": "+", "-": "-", "": "fixed"}[m.group(2)]
    return {"damage_base": float(m.group(1)), "damage_mod": mod}


def supertype(stage: str) -> str:
    if "Pokémon" in stage and stage != "Pokémon Tool":
        return "Pokémon"
    if "Energy" in stage:
        return "Energy"
    return "Trainer"


def clean_moves(raw: pd.DataFrame) -> pd.DataFrame:
    """Typed, move-level table (one row per card x move).

    Cleaning decisions (justified in the notebook). The file uses 'n/a' and
    empty cells inconsistently, so the meaning of a blank depends on the
    card's supertype:
    * Trainer / Energy rows: HP, weakness, resistance, retreat, cost are not
      applicable -> NaN (never imputed).
    * Pokémon rows: 'n/a' weakness / resistance means the Pokémon has none
      -> 'None'; 'n/a' retreat means a free retreat -> 0 (both verified
      against the engine's own card table).
    * numeric columns (HP, retreat, collection number) cast to numbers.
    * cost / damage strings parsed into numeric features.
    * abilities (move names like '[Ability] ...' or '[Tera]') are flagged,
      they are not attacks and have no cost.
    """
    df = raw.rename(columns=COLUMN_NAMES).copy()
    df["card_id"] = df["card_id"].astype(int)
    df["supertype"] = df["stage"].map(supertype)

    pok = df["supertype"].eq("Pokémon")
    for col in ["weakness", "resistance"]:
        v = df[col].replace({"": np.nan, "n/a": np.nan})
        v = v.map(lambda s: ", ".join(_symbols(s)) if isinstance(s, str) else s)
        df[col] = v.where(~(pok & v.isna()), "None")  # a Pokémon without one has none
    df["type"] = df["type"].replace({"n/a": np.nan, "": np.nan})
    df["type"] = df["type"].map(lambda s: "/".join(sorted(set(_symbols(s)))) if isinstance(s, str) else s)

    for col in ["hp", "retreat", "collection_no"]:
        df[col] = pd.to_numeric(df[col].replace({"n/a": np.nan, "": np.nan}), errors="coerce")
    # A Pokémon whose retreat is 'n/a' has a *free* retreat (cost 0) - verified
    # against the engine's retreatCost in the notebook. This is the only imputation.
    df["retreat_imputed"] = df["supertype"].eq("Pokémon") & df["retreat"].isna()
    df.loc[df["retreat_imputed"], "retreat"] = 0

    for col in ["rule", "category", "previous_stage", "species", "move_name", "effect", "expansion"]:
        df[col] = df[col].replace({"n/a": np.nan, "": np.nan})

    df["is_ex"] = df["rule"].isin(["Pokémon ex", "Mega Pokémon ex"])
    df["is_mega_ex"] = df["rule"].eq("Mega Pokémon ex")
    df["is_ace_spec"] = df["rule"].eq("ACE SPEC")
    df["is_tera"] = df["category"].fillna("").str.startswith("Tera")
    df["is_ability"] = df["move_name"].fillna("").str.startswith("[")

    df = pd.concat([df, pd.DataFrame(df["cost"].map(parse_cost).tolist(), index=df.index)], axis=1)
    df = pd.concat([df, pd.DataFrame(df["damage"].map(parse_damage).tolist(), index=df.index)], axis=1)
    df.loc[df["is_ability"], ["cost_total", "cost_colorless", "cost_typed"]] = np.nan
    df["is_attack"] = df["move_name"].notna() & ~df["is_ability"]
    # damage per energy only makes sense for attacks with a printed number and a cost
    df["damage_per_energy"] = df["damage_base"] / df["cost_total"].replace(0, np.nan)
    return df


def card_level_table(moves: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the move-level table into one row per card."""
    static = [
        "card_id", "name", "expansion", "stage", "supertype", "rule", "category",
        "previous_stage", "hp", "type", "weakness", "resistance", "retreat",
        "is_ex", "is_mega_ex", "is_ace_spec", "is_tera",
    ]
    first = moves.groupby("card_id", as_index=False)[static[1:]].first()
    first.insert(0, "card_id", first.pop("card_id"))
    atk = moves[moves["is_attack"]]
    agg = atk.groupby("card_id").agg(
        n_attacks=("move_name", "size"),
        max_damage=("damage_base", "max"),
        min_cost=("cost_total", "min"),
        max_cost=("cost_total", "max"),
        best_damage_per_energy=("damage_per_energy", "max"),
        has_variable_damage=("damage_mod", lambda s: bool((s == "x").any() or (s == "+").any())),
    )
    abil = moves.groupby("card_id")["is_ability"].any().rename("has_ability")
    out = first.merge(agg, on="card_id", how="left").merge(abil, on="card_id", how="left")
    pok = out["supertype"].eq("Pokémon")
    out.loc[pok, "n_attacks"] = out.loc[pok, "n_attacks"].fillna(0)
    out["has_variable_damage"] = out["has_variable_damage"].astype("boolean")
    return out


# ---------------------------------------------------------------------------
# Engine tables (exact attack IDs used in observations)
# ---------------------------------------------------------------------------

def import_make():
    """Import ``kaggle_environments.make`` quietly.

    kaggle-environments tries to register every bundled game on import and
    prints a warning for games whose optional dependencies (open_spiel ...)
    are not installed. Those games are irrelevant here.
    """
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
    return make


def _engine_lib():
    import_make()
    from kaggle_environments.envs.cabt.cg.sim import get_lib  # lazy: heavy import

    lib = get_lib()
    lib.AllCard.restype = ctypes.c_char_p
    lib.AllAttack.restype = ctypes.c_char_p
    return lib


def load_engine_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Card and attack tables straight from the simulator library.

    The CSV has no attack IDs, but observations refer to attacks by
    ``attackId``; the engine tables provide that link.
    """
    lib = _engine_lib()
    cards = pd.DataFrame(json.loads(lib.AllCard().decode()))
    attacks = pd.DataFrame(json.loads(lib.AllAttack().decode()))
    return cards, attacks


def build_attack_table(moves: pd.DataFrame, engine_cards: pd.DataFrame, engine_attacks: pd.DataFrame) -> pd.DataFrame:
    """One row per (card, attackId) joining engine attacks with parsed CSV moves.

    ``est_damage`` is the damage our agent expects from an attack:
    the engine's printed damage, or - when the engine reports 0 because the
    damage is variable (e.g. "100×") - the CSV base number times 2, a rough
    guess of the multiplier we expect to hit.
    """
    rows = []
    for _, c in engine_cards.iterrows():
        for pos, aid in enumerate(c["attacks"]):
            rows.append({"card_id": c["cardId"], "attack_id": aid, "attack_pos": pos})
    link = pd.DataFrame(rows).merge(
        engine_attacks.rename(columns={"attackId": "attack_id", "name": "move_name", "damage": "engine_damage"}),
        on="attack_id", how="left",
    )
    link["energy_cost"] = link["energies"].map(len)
    csv_atk = moves.loc[moves["is_attack"], ["card_id", "move_name", "damage_base", "damage_mod"]]
    csv_atk = csv_atk.drop_duplicates(["card_id", "move_name"])
    out = link.merge(csv_atk, on=["card_id", "move_name"], how="left", indicator=True)
    out["matched_csv"] = out.pop("_merge").eq("both")
    mult = np.where(out["damage_mod"].eq("x"), 2.0, 1.0)
    fallback = out["damage_base"].fillna(0) * mult
    out["est_damage"] = np.where(out["engine_damage"] > 0, out["engine_damage"], fallback)
    return out


def build_card_db(engine_cards: pd.DataFrame, attack_table: pd.DataFrame) -> dict:
    """Compact JSON-serialisable card database used by the agent at play time."""
    cards = {}
    for _, c in engine_cards.iterrows():
        cards[str(int(c["cardId"]))] = {
            "name": c["name"],
            "card_type": int(c["cardType"]),
            "hp": int(c["hp"]),
            "energy_type": int(c["energyType"]),
            "weakness": None if pd.isna(c["weakness"]) else int(c["weakness"]),
            "resistance": None if pd.isna(c["resistance"]) else int(c["resistance"]),
            "retreat": int(c["retreatCost"]),
            "basic": bool(c["basic"]),
            "stage1": bool(c["stage1"]),
            "stage2": bool(c["stage2"]),
            "ex": bool(c["ex"]),
            "mega_ex": bool(c["megaEx"]),
            "evolves_from": c["evolvesFrom"],
            "attacks": [int(a) for a in c["attacks"]],
        }
    attacks = {
        str(int(r.attack_id)): {"cost": [int(e) for e in r.energies], "damage": float(r.est_damage)}
        for r in attack_table.itertuples()
    }
    return {"cards": cards, "attacks": attacks}


# ---------------------------------------------------------------------------
# Simulated games -> tabular records
# ---------------------------------------------------------------------------

RESULT_REASONS = {1: "Took all prizes", 2: "Opponent decked out", 3: "Opponent has no Active Pokémon", 4: "Card effect"}


def _in_play(player_state: dict) -> list[dict]:
    return [p for p in (player_state.get("active") or []) + (player_state.get("bench") or []) if p]


def _snapshot(cur: dict, me: int, game_id: int, went_first: bool) -> dict:
    """Full-information board features for player ``me`` at the start of a turn."""
    p, o = cur["players"][me], cur["players"][1 - me]

    def side(ps):
        mons = _in_play(ps)
        act = (ps.get("active") or [None])[0]
        return {
            "prizes_left": len(ps.get("prize") or []),
            "n_pokemon": len(mons),
            "hp_in_play": sum(m["hp"] for m in mons),
            "energy_in_play": sum(len(m.get("energies") or []) for m in mons),
            "active_hp": act["hp"] if act else 0,
            "active_energy": len(act.get("energies") or []) if act else 0,
            "hand": ps.get("handCount", 0),
            "deck": ps.get("deckCount", 0),
        }

    a, b = side(p), side(o)
    row = {"game_id": game_id, "player": me, "turn": cur["turn"], "went_first": went_first}
    for k in a:
        row[f"my_{k}"] = a[k]
        row[f"opp_{k}"] = b[k]
        row[f"diff_{k}"] = a[k] - b[k]
    return row


def parse_game(vis: list[dict], game_id: int, names: tuple[str, str]) -> tuple[list[dict], list[dict], list[dict]]:
    """Turn one episode's visualizer frames into (player-game rows, decision rows, turn snapshots).

    ``vis`` is ``env.steps[0][0]["visualize"]`` - the same full-information
    record that Kaggle stores in episode replays.
    """
    stats = [dict(attacks=0, damage_dealt=0, attachments=0, evolutions=0, cards_played=0,
                  switches=0, decisions=0, main_decisions=0, options_seen=0) for _ in range(2)]
    decisions, snaps = [], []
    first_player, result, reason, last_turn = None, None, None, 0
    final_cur = None
    for f in vis:
        cur = f.get("current") or {}
        if cur:
            final_cur = cur
            last_turn = max(last_turn, cur.get("turn", 0))
            if cur.get("firstPlayer", -1) in (0, 1):
                first_player = cur["firstPlayer"]
        for log in f.get("logs") or []:
            t, pi = log["type"], log.get("playerIndex")
            if t == "Attack":
                stats[pi]["attacks"] += 1
            elif t == "HpChange" and log.get("value", 0) < 0 and pi in (0, 1):
                stats[1 - pi]["damage_dealt"] += -log["value"]  # pi = owner of the damaged Pokémon
            elif t == "Attach" and log.get("cardId") is not None:
                stats[pi]["attachments"] += 1  # energy (mostly) or Pokémon Tool
            elif t == "Evolve":
                stats[pi]["evolutions"] += 1
            elif t == "Play":
                stats[pi]["cards_played"] += 1
            elif t == "Switch":
                stats[pi]["switches"] += 1
            elif t == "TurnStart" and cur and first_player is not None:
                snaps.append(_snapshot(cur, pi, game_id, pi == first_player))
            elif t == "Result":
                result, reason = log["result"], log["reason"]
        sel = f.get("select") or {}
        if cur and sel.get("type") not in (None, "None") and sel.get("option"):
            me = cur.get("yourIndex", 0)
            n_opt = len(sel["option"])
            chosen = f.get("selected")
            if chosen is None and f.get("action"):
                chosen = f["action"][me]
            chosen_type = None
            if isinstance(chosen, list) and len(chosen) == 1 and 0 <= chosen[0] < n_opt:
                chosen_type = sel["option"][chosen[0]].get("type")
            decisions.append({"game_id": game_id, "player": me, "turn": cur.get("turn"),
                              "select_type": sel["type"], "context": sel["context"],
                              "n_options": n_opt, "min_count": sel.get("minCount"),
                              "max_count": sel.get("maxCount"), "chosen_type": chosen_type})
            stats[me]["decisions"] += 1
            stats[me]["options_seen"] += n_opt
            if sel["type"] == "Main":
                stats[me]["main_decisions"] += 1

    games = []
    for i in range(2):
        prizes_left = len((final_cur or {}).get("players", [{}, {}])[i].get("prize") or []) if final_cur else np.nan
        games.append({
            "game_id": game_id, "player": i, "agent": names[i], "opponent": names[1 - i],
            "went_first": first_player == i, "won": int(result == i), "draw": int(result == 2),
            "end_reason": RESULT_REASONS.get(reason, "Unknown"), "n_turns": last_turn,
            "prizes_taken": 6 - prizes_left if prizes_left == prizes_left else np.nan,
            **stats[i],
        })
    for s in snaps:
        s["won"] = int(result == s["player"])
    return games, decisions, snaps


def simulate_games(agent0, agent1, n_games: int, names: tuple[str, str] = ("agent0", "agent1"),
                   swap_seats: bool = True, start_id: int = 0, verbose: bool = True):
    """Play ``n_games`` single games (BO1) and return three DataFrames.

    Seats are swapped every other game so that neither agent always sits in
    seat 0. Returns (games, decisions, snapshots); ``games`` has one row per
    player per game, the target column is ``won``.
    """
    make = import_make()

    all_games, all_dec, all_snap = [], [], []
    t0 = time.time()
    for g in range(n_games):
        swap = swap_seats and g % 2 == 1
        pair = (agent1, agent0) if swap else (agent0, agent1)
        nm = (names[1], names[0]) if swap else names
        env = make("cabt", configuration={"bo": 1})
        steps = env.run(list(pair))
        vis = steps[0][0].get("visualize") or []
        games, dec, snaps = parse_game(vis, start_id + g, nm)
        statuses = [s.status for s in env.state]
        for row in games:
            row["status"] = statuses[row["player"]]
        all_games += games
        all_dec += dec
        all_snap += snaps
        if verbose and (g + 1) % max(1, n_games // 5) == 0:
            print(f"  {g + 1}/{n_games} games  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(all_games), pd.DataFrame(all_dec), pd.DataFrame(all_snap)
