"""
Live fantasy-draft market data: expert consensus rankings (ECR), average
draft position (ADP), the cross-platform player-ID crosswalk, dynasty trade
values, and injury/news feeds.

WHY THIS IS SEPARATE FROM data/rankings.py: that module reads the three
FantasyPros CSVs the user periodically re-exports by hand into rankings/.
Those are a static snapshot - fine for a mid-summer sanity check, stale by
the time a real August draft happens. A draft board is only as good as how
fresh its market data is (a torn ACL on August 20th moves a player 40 spots
overnight), so everything here is fetched live at run time with a short
cache TTL, and every fetcher degrades to an empty DataFrame rather than
raising, so a dead source downgrades one column instead of taking the
draft room down mid-draft. The local CSVs stay as the offline fallback.

SOURCE NOTES
- ECR comes from DynastyProcess's nightly FantasyPros scrape (the same
  dataset nflreadpy's load_ff_rankings wraps). We fetch it directly rather
  than through nflreadpy because nflreadpy hardcodes the github.com/.../raw/
  URL form, which some corporate/proxied networks reject while allowing
  raw.githubusercontent.com - see _fetch_csv's URL list. Same bytes either
  way, so this is purely about reachability, not content.
- ADP is deliberately multi-source. ADP is the single most
  settings-sensitive input on a draft board: a 10-team half-PPR board and a
  14-team superflex board have genuinely different markets, not the same
  market rescaled. Fantasy Football Calculator publishes ADP split by both
  league size AND format (including 2QB), which is why it's the primary.
"""
import io
import math
import os
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st

# DynastyProcess publishes to GitHub; both hostnames serve identical bytes.
# raw.githubusercontent.com is listed first because it's the one that
# survives the most restrictive networks (some proxies allow it while
# 403-ing github.com's /raw/ redirect path).
_DP_HOSTS = [
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/{f}",
    "https://github.com/dynastyprocess/data/raw/master/files/{f}",
]

# Long enough that a draft-day flurry of reruns doesn't re-download a 900KB
# CSV on every widget click, short enough that news breaking mid-morning is
# picked up before an evening draft. These files are refreshed once daily at
# the source, so anything under ~12h is really just protecting against our
# own rerun rate rather than chasing genuinely newer bytes.
_MARKET_TTL = 60 * 60 * 6

_REQUEST_HEADERS = {
    # Some of these endpoints (notably Fantasy Football Calculator) return
    # 403 to requests' default python-requests/x.y UA.
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/csv,application/json,*/*",
}

# FantasyPros publishes a separate consensus board per draft format, and
# they are NOT reorderings of one another - a superflex board isn't the 1QB
# board with QBs shifted up, because the whole positional value structure
# changes when a second QB slot exists. page_type is DynastyProcess's own
# column identifying which FantasyPros page a row was scraped from.
ECR_BOARDS = {
    'Redraft 1QB': 'redraft-overall',
    'Redraft Superflex / 2QB': 'redraft-op',
    'Best Ball': 'best-overall',
    'Dynasty 1QB': 'dynasty-overall',
    'Dynasty Superflex': 'dynasty-op',
    'Dynasty Rookie': 'dynasty-rk',
}

# The per-position FantasyPros pages carry meaningfully MORE players than
# the overall board does (e.g. 226 WRs vs the overall board's 170), because
# the overall board stops once players stop being draftable in a normal
# league while the positional pages keep going. Merging them in is what
# keeps the deep end of the board (late-round dart throws, handcuffs)
# populated instead of falling off a cliff at overall rank ~500.
ECR_POSITION_PAGES = {
    'QB': 'redraft-qb', 'RB': 'redraft-rb', 'WR': 'redraft-wr',
    'TE': 'redraft-te', 'K': 'redraft-k', 'DST': 'redraft-dst',
}

# Fantasy Football Calculator's format slugs. Keyed by (scoring, is_superflex)
# because FFC models superflex/2QB as its own ADP format rather than as a
# roster-setting modifier on a PPR board - which is correct, and is exactly
# the settings-sensitivity this board is trying to capture.
FFC_FORMATS = {
    ('Standard', False): 'standard',
    ('Half-PPR', False): 'half-ppr',
    ('Full PPR', False): 'ppr',
    ('Standard', True): '2qb',
    ('Half-PPR', True): '2qb',
    ('Full PPR', True): '2qb',
}

# FFC only publishes ADP for the league sizes it has enough real drafts to
# support. Asking for teams=11 returns an empty/garbage payload rather than
# an error, so requests are snapped to the nearest supported size and the
# actual size used is reported back to the UI (see fetch_adp's 'meta').
FFC_SUPPORTED_TEAM_COUNTS = [8, 10, 12, 14]

# FantasyPros' public ADP pages, one per draft format. Their ADP is a
# consensus across the platforms people actually draft on - NFFC, Sleeper,
# ESPN, CBS, RTSports, Yahoo - which is a completely different (and far
# better) sample than one free mock site's own drafts.
FANTASYPROS_ADP_PAGES = {
    ('Standard', False): 'overall',
    ('Half-PPR', False): 'half-point-ppr-overall',
    ('Full PPR', False): 'ppr-overall',
    ('Standard', True): 'superflex',
    ('Half-PPR', True): 'superflex',
    ('Full PPR', True): 'superflex',
}

# The user's own periodic FantasyPros draft-rankings exports, which carry an
# "ECR VS. ADP" column - i.e. FantasyPros' consensus ADP, already in the
# repo. Same three files data/rankings.py reads; duplicated as a mapping
# here rather than imported so this module stays free of that dependency
# and a renamed file fails loudly in one place.
FANTASYPROS_LOCAL_FILES = {
    'Full PPR': 'rankings/fantasypros_2026_draft_rankings_ppr.csv',
    'Half-PPR': 'rankings/fantasypros_2026_draft_rankings_half_ppr.csv',
    'Standard': 'rankings/fantasypros_2026_draft_rankings_standard.csv',
}


def _fetch_bytes(urls, timeout=25, params=None, headers=None):
    """
    Try each URL in order, returning the first successful response body.

    Returns (content_bytes, error_message). A caller that gets None back
    should degrade gracefully, never raise - this whole module is
    best-effort enrichment on top of a board that must still render from
    local data when the network is unavailable (draft night on a hotel
    wifi, an office proxy blocking half the internet).
    """
    last_err = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={**_REQUEST_HEADERS, **(headers or {})}, params=params)
            if resp.status_code == 200 and resp.content:
                return resp.content, None
            last_err = f"HTTP {resp.status_code}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return None, last_err or "unreachable"


def _fetch_csv(filename):
    content, err = _fetch_bytes([h.format(f=filename) for h in _DP_HOSTS])
    if content is None:
        return pd.DataFrame(), err
    try:
        return pd.read_csv(io.BytesIO(content), low_memory=False), None
    except Exception as exc:
        return pd.DataFrame(), f"parse failed: {exc}"


@st.cache_data(ttl=_MARKET_TTL, show_spinner=False)
def load_ecr_raw():
    """
    The full FantasyPros ECR table - every board (redraft/dynasty/best-ball,
    1QB/superflex, overall/positional/IDP) stacked in one long frame, as
    published. Callers slice it via build_ecr_board rather than each
    re-fetching, so one download serves the whole app.

    Returns (df, error_message).
    """
    df, err = _fetch_csv("db_fpecr_latest.csv")
    if df.empty:
        return df, err
    df.columns = [c.strip() for c in df.columns]
    return df, None


@st.cache_data(ttl=_MARKET_TTL, show_spinner=False)
def load_player_id_map():
    """
    Cross-platform player ID crosswalk (sleeper_id / espn_id / yahoo_id /
    fantasypros_id / gsis_id / pff_id / mfl_id ...).

    This is what makes live draft sync robust. Matching a Sleeper draft pick
    to a board row by NAME is guesswork the moment two players share one
    ("Michael Thomas" has been two different active NFL players at once);
    matching on sleeper_id is exact. Name matching stays as the fallback for
    sources that only give us a name (a pasted draft log), but wherever a
    platform hands us its own ID, we use it.
    """
    df, err = _fetch_csv("db_playerids.csv")
    if df.empty:
        return df, err
    keep = [c for c in ('mfl_id', 'sleeper_id', 'espn_id', 'yahoo_id', 'fantasypros_id',
                        'gsis_id', 'pff_id', 'cbs_id', 'rotowire_id', 'sportradar_id',
                        'name', 'merge_name', 'position', 'team', 'age', 'birthdate',
                        'draft_year', 'draft_round', 'draft_pick', 'draft_ovr') if c in df.columns]
    return df[keep].copy(), None


@st.cache_data(ttl=_MARKET_TTL, show_spinner=False)
def load_dynasty_values():
    """
    DynastyProcess trade values (value_1qb / value_2qb) plus each player's
    age. Not a redraft input - a 29-year-old RB and a 23-year-old RB with
    identical projections are identical picks in a one-year league - but
    it's the difference between a redraft board and a keeper/dynasty board,
    and the age column is genuinely useful context in either.
    """
    df, err = _fetch_csv("values-players.csv")
    if df.empty:
        return df, err
    return df, None


def parse_ecr_upload(uploaded_file):
    """
    A FantasyPros rankings export straight into the board's ECR columns.

    WHY THIS EXISTS: every consensus rank on this board comes from
    DynastyProcess's nightly mirror of FantasyPros, and that mirror can
    simply stop - it did, sitting on a 2026-07-31 scrape for a week with no
    fresher copy to fetch. There is no way to fix that from this side, and
    "your rankings are stale and there's nothing you can do" is a bad answer
    two days before a draft. Exporting the CSV yourself takes ten seconds and
    is always current.

    Accepts their export as-is: RK, TIERS, "PLAYER NAME", TEAM, POS, "BYE
    WEEK", "ECR VS. ADP". Also accepts the wider export that carries BEST /
    WORST / AVG / STD DEV, which is strictly better - that spread is what
    Upside/Bust/Risk are built from, and without it those columns fall back
    to a modelled estimate.

    Returns (df, error) shaped exactly like build_ecr_board's output, so it
    drops in wherever the live board does.
    """
    if uploaded_file is None:
        return pd.DataFrame(), None
    try:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file)
    except Exception as exc:
        return pd.DataFrame(), f"couldn't read that file: {exc}"

    raw.columns = [str(c).strip().upper() for c in raw.columns]
    lower = {c: c for c in raw.columns}

    def _col(*names):
        for name in names:
            if name in lower:
                return raw[name]
        return None

    names = _col('PLAYER NAME', 'PLAYER', 'NAME')
    rank = _col('RK', 'RANK', 'ECR', 'AVG.', 'OVERALL')
    if names is None or rank is None:
        return pd.DataFrame(), ("that CSV has no player-name and rank columns — export the "
                                "Draft Rankings view from FantasyPros (the one with RK, "
                                "PLAYER NAME, POS).")

    pos = _col('POS', 'POSITION')
    out = pd.DataFrame({
        'Player': names.astype(str).str.strip().str.strip('"'),
        # Their POS carries a positional rank suffix ("WR1"); the board's Pos
        # is a bare code everywhere else.
        'Pos': (pos.astype(str).str.upper().str.replace(r'\d+$', '', regex=True).str.strip()
                if pos is not None else ''),
        'Team': (_col('TEAM', 'TM').astype(str).str.strip()
                 if _col('TEAM', 'TM') is not None else ''),
        'ECR': pd.to_numeric(rank, errors='coerce'),
        'Bye': pd.to_numeric(_col('BYE WEEK', 'BYE'), errors='coerce')
               if _col('BYE WEEK', 'BYE') is not None else np.nan,
    })
    for column, candidates in (('ECR SD', ('STD DEV', 'SD', 'STDEV')),
                               ('ECR Best', ('BEST',)),
                               ('ECR Worst', ('WORST',))):
        series = _col(*candidates)
        out[column] = pd.to_numeric(series, errors='coerce') if series is not None else np.nan

    out = out.dropna(subset=['ECR'])
    out = out[out['Player'].ne('') & out['Player'].str.lower().ne('nan')]
    if out.empty:
        return pd.DataFrame(), "no ranked players found in that file."
    out = out.sort_values('ECR').drop_duplicates(subset=['Player'], keep='first')
    out['Pos Rank'] = out.groupby('Pos')['ECR'].rank(method='first').astype(int)
    return out.reset_index(drop=True), None


def build_ecr_board(ecr_raw, board='Redraft 1QB', include_positional_depth=True):
    """
    One clean board out of the stacked ECR table: consensus rank, the spread
    of expert opinion behind it, position, team and bye.

    The 'sd' / 'best' / 'worst' columns are the reason this is worth doing
    over just taking a rank column. A consensus rank of 30 with experts
    ranging 28-32 and a consensus rank of 30 with experts ranging 12-70 are
    completely different draft picks, and every good paid tool surfaces that
    distinction as some flavor of "upside/bust" or "risk". Here it becomes
    Upside/Bust/Risk downstream in data.draft_board - carried through rather
    than collapsed into a single number, so the board can show WHY a player
    is volatile, not just that he is.
    """
    if ecr_raw is None or ecr_raw.empty or 'page_type' not in ecr_raw.columns:
        return pd.DataFrame()

    page = ECR_BOARDS.get(board, 'redraft-overall')
    df = ecr_raw[ecr_raw['page_type'] == page].copy()
    if df.empty:
        return pd.DataFrame()

    def _tidy(frame):
        out = pd.DataFrame({
            'Player': frame['player'].astype(str).str.strip(),
            'Pos': frame['pos'].astype(str).str.upper().str.strip(),
            'Team': frame.get('team', pd.Series(index=frame.index, dtype=str)).astype(str).str.strip(),
            'ECR': pd.to_numeric(frame['ecr'], errors='coerce'),
            'ECR SD': pd.to_numeric(frame.get('sd'), errors='coerce'),
            'ECR Best': pd.to_numeric(frame.get('best'), errors='coerce'),
            'ECR Worst': pd.to_numeric(frame.get('worst'), errors='coerce'),
            'Bye': pd.to_numeric(frame.get('bye'), errors='coerce'),
            'fantasypros_id': frame.get('id'),
        })
        return out.dropna(subset=['ECR']).query("Player != '' and Player != 'nan'")

    board_df = _tidy(df)

    if include_positional_depth and board.startswith('Redraft'):
        # Players the overall board omits still get a row, ranked behind
        # everyone the overall board DID include, so they're draftable in
        # the late rounds instead of simply absent. Their ECR is synthesized
        # from positional rank rather than invented: see the offset below.
        pos_frames = []
        for pos, pos_page in ECR_POSITION_PAGES.items():
            pos_rows = ecr_raw[ecr_raw['page_type'] == pos_page]
            if pos_rows.empty:
                continue
            tidy = _tidy(pos_rows)
            tidy['Pos Rank'] = tidy['ECR'].rank(method='first').astype(int)
            pos_frames.append(tidy)
        if pos_frames:
            pos_all = pd.concat(pos_frames, ignore_index=True)
            have = set(board_df['Player'].str.lower())
            extra = pos_all[~pos_all['Player'].str.lower().isin(have)].copy()
            if not extra.empty:
                # Ordering among the extras is by positional rank scaled by
                # how deep that position already ran on the overall board -
                # a WR who is WR180 is a later pick than a TE who is TE40,
                # and sorting the extras by raw positional rank alone would
                # get that exactly backwards.
                pos_depth = board_df.groupby('Pos')['ECR'].max().to_dict()
                base = board_df['ECR'].max()
                extra['ECR'] = base + extra['Pos Rank'] + extra['Pos'].map(
                    lambda p: pos_depth.get(p, base) / 100.0).fillna(0)
                board_df = pd.concat([board_df, extra.drop(columns=['Pos Rank'])], ignore_index=True)

    board_df = board_df.sort_values('ECR').drop_duplicates(subset=['Player'], keep='first')
    board_df['ECR'] = board_df['ECR'].round(2)
    board_df['Pos Rank'] = board_df.groupby('Pos')['ECR'].rank(method='first').astype(int)
    return board_df.reset_index(drop=True)


@st.cache_data(ttl=_MARKET_TTL, show_spinner=False)
def fetch_ffc_adp(scoring, num_teams, is_superflex, year):
    """
    Fantasy Football Calculator ADP, split by league size and scoring
    format - real drafts, not a projection of what a market should look
    like.

    League size is a real input here, not decoration. In an 8-team league
    the elite QBs slide (there's enough talent that nobody reaches) while in
    a 14-teamer they go a full round earlier; a board that shows one ADP for
    all league sizes is quietly wrong for everyone not in a 12-teamer.

    Returns (df, meta) where meta records what was ACTUALLY requested after
    snapping to a supported league size, so the UI can be honest about
    showing 12-team ADP to someone in an 11-team league.
    """
    fmt = FFC_FORMATS.get((scoring, bool(is_superflex)), 'ppr')
    teams = min(FFC_SUPPORTED_TEAM_COUNTS, key=lambda t: abs(t - int(num_teams)))
    meta = {'source': 'Fantasy Football Calculator', 'format': fmt, 'teams': teams,
            'requested_teams': int(num_teams), 'year': year, 'error': None}

    content, err = _fetch_bytes(
        [f"https://fantasyfootballcalculator.com/api/v1/adp/{fmt}"],
        params={'teams': teams, 'year': year, 'position': 'all'},
    )
    if content is None:
        meta['error'] = err
        return pd.DataFrame(), meta

    try:
        import json
        payload = json.loads(content)
        rows = payload.get('players', []) if isinstance(payload, dict) else []
        if not rows:
            meta['error'] = 'no players in response'
            return pd.DataFrame(), meta
        df = pd.DataFrame(rows)
        out = pd.DataFrame({
            'Player': df.get('name', pd.Series(dtype=str)).astype(str).str.strip(),
            'Pos': df.get('position', pd.Series(dtype=str)).astype(str).str.upper().str.strip(),
            'Team': df.get('team', pd.Series(dtype=str)).astype(str).str.strip(),
            'ADP': pd.to_numeric(df.get('adp'), errors='coerce'),
            # FFC's own stdev of observed draft slots. This is the real
            # thing - actual variance in where he went across hundreds of
            # drafts - and it's what makes the availability model in
            # data.draft_board honest instead of a guess.
            'ADP SD': pd.to_numeric(df.get('stdev'), errors='coerce'),
            'ADP High': pd.to_numeric(df.get('high'), errors='coerce'),
            'ADP Low': pd.to_numeric(df.get('low'), errors='coerce'),
            'ADP N': pd.to_numeric(df.get('times_drafted'), errors='coerce'),
        }).dropna(subset=['ADP'])
        out = out[out['Player'].ne('') & out['Player'].str.lower().ne('nan')]
        return out.sort_values('ADP').reset_index(drop=True), meta
    except Exception as exc:
        meta['error'] = f"parse failed: {exc}"
        return pd.DataFrame(), meta


def _clean_fantasypros_adp_name(value):
    """
    "Ja'Marr Chase CIN (6)" -> "Ja'Marr Chase".

    FantasyPros renders the team code and bye week inside the same cell as
    the name on their ADP pages. Stripped with an explicit trailing-token
    walk rather than a regex on team codes, because the team column is
    absent for free agents and the bye is absent in the preseason, so the
    cell has four different shapes across one table.
    """
    text = str(value or '').replace('\xa0', ' ').strip()
    text = re.sub(r'\s*\(\s*\d+\s*\)\s*$', '', text)          # trailing "(6)" bye
    text = re.sub(r'\s+(FA|[A-Z]{2,3})$', '', text)            # trailing team code
    # Their tables sometimes repeat the name inside the cell for the mobile
    # layout ("Ja'Marr ChaseJa'Marr Chase"); collapse an exact doubling.
    half = len(text) // 2
    if len(text) > 6 and text[:half] == text[half:]:
        text = text[:half]
    return text.strip(' .,-')


@st.cache_data(ttl=_MARKET_TTL, show_spinner=False)
def fetch_fantasypros_adp(scoring, is_superflex, year=None):
    """
    FantasyPros' consensus ADP, scraped from their public ADP page.

    WHY THIS IS THE DEFAULT NOW AND FFC ISN'T: FantasyPros averages the ADP
    published by the sites people actually draft on - NFFC (high-stakes, real
    money), Sleeper, ESPN, CBS, RTSports. Fantasy Football Calculator
    publishes ADP measured from free mock drafts run on its own site. Both
    are "real humans", but only one of them is a sample of drafts anybody
    cared about winning, and the gap shows up exactly where it hurts: FFC
    slides tight ends and quarterbacks a full round or more past where they
    go in real leagues, and every downstream column - value-vs-market,
    availability, VONA - inherits that error while still looking authoritative.

    League size is NOT a parameter here, unlike FFC: FantasyPros publishes
    one consensus per format rather than one per league size. A generic ADP
    measured from real drafts beats a size-specific one measured from mocks,
    but the UI says which it is rather than pretending the distinction
    doesn't exist.

    Returns (df, meta), empty rather than raising on any failure.
    """
    page = FANTASYPROS_ADP_PAGES.get((scoring, bool(is_superflex)), 'ppr-overall')
    meta = {'source': 'FantasyPros consensus ADP', 'format': page, 'teams': None,
            'year': year, 'error': None}

    content, err = _fetch_bytes([f"https://www.fantasypros.com/nfl/adp/{page}.php"],
                                headers={'Accept': 'text/html,application/xhtml+xml,*/*'})
    if content is None:
        meta['error'] = err
        return pd.DataFrame(), meta

    try:
        tables = pd.read_html(io.BytesIO(content))
    except Exception as exc:
        meta['error'] = f"parse failed: {exc}"
        return pd.DataFrame(), meta

    # Their page carries several tables (nav widgets, an ad unit); the ADP
    # table is the one with a name-ish column, a POS column and an AVG.
    for table in sorted(tables, key=len, reverse=True):
        table = table.copy()
        table.columns = [str(c).strip() for c in table.columns]
        lower = {c.lower(): c for c in table.columns}
        name_col = next((lower[c] for c in lower if c in ('player team (bye)', 'player', 'player name')), None)
        avg_col = next((lower[c] for c in lower if c in ('avg', 'avg.', 'adp')), None)
        if not name_col or not avg_col:
            continue
        out = pd.DataFrame({
            'Player': table[name_col].map(_clean_fantasypros_adp_name),
            'ADP': pd.to_numeric(table[avg_col], errors='coerce'),
        })
        pos_col = lower.get('pos')
        if pos_col:
            out['Pos'] = (table[pos_col].astype(str).str.upper()
                          .str.replace(r'\d+$', '', regex=True).str.strip())
        out = out.dropna(subset=['ADP'])
        out = out[out['Player'].ne('') & out['Player'].str.lower().ne('nan')]
        # A page that renders but whose table shape has drifted produces a
        # short or nonsensical frame. Better to fail over to the local
        # export than to silently seed the whole board off ten rows.
        if len(out) >= 100 and float(out['ADP'].min()) < 5:
            return out.sort_values('ADP').reset_index(drop=True), meta

    meta['error'] = 'no ADP table found on the page'
    return pd.DataFrame(), meta


@st.cache_data(show_spinner=False)
def load_fantasypros_local_adp(scoring):
    """
    FantasyPros' consensus ADP, recovered from the draft-rankings CSVs
    already in rankings/.

    Those exports carry an "ECR VS. ADP" column, which is exactly
    (ADP - expert consensus rank) for every player - so ADP is just
    RK + that difference. It is the same consensus number the live page
    serves, published a few days earlier, and it needs no network at all.

    That last part is why this exists: draft night is the worst possible
    time to discover a site is unreachable, and the difference between a
    board with real ADP and one without is the entire value/availability
    half of it. This is the floor under that.
    """
    path = FANTASYPROS_LOCAL_FILES.get(scoring)
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    if 'RK' not in df.columns or 'ECR VS. ADP' not in df.columns:
        return pd.DataFrame()

    rank = pd.to_numeric(df['RK'], errors='coerce')
    # The column arrives as a signed string ("+12", "-3", "0"); a stray
    # non-numeric marker means "no ADP for this player", not zero.
    delta = pd.to_numeric(df['ECR VS. ADP'].astype(str).str.replace('+', '', regex=False),
                          errors='coerce')
    out = pd.DataFrame({
        'Player': df.get('PLAYER NAME', pd.Series(dtype=str)).astype(str).str.strip(),
        'Pos': df.get('POS', pd.Series(dtype=str)).astype(str).str.upper()
                 .str.replace(r'\d+$', '', regex=True).str.strip(),
        'ADP': rank + delta,
    }).dropna(subset=['ADP'])
    out = out[out['Player'].ne('') & out['Player'].str.lower().ne('nan')]
    # The difference is published rounded to whole picks, so several players
    # can land on the same ADP. Ties are broken by consensus rank, which is
    # what the market is closest to agreeing on within a tie anyway, and
    # leaving exact ties in place would make the board's ordering arbitrary.
    out['ADP'] = out['ADP'] + rank.reindex(out.index).rank(method='first') / 10000.0
    return out.sort_values('ADP').reset_index(drop=True)


def parse_adp_upload(uploaded_file):
    """
    Any CSV with a player-name column and an ADP-ish column. The escape
    hatch that matters: if none of the live sources are reachable (or the
    user has ADP from a source we don't integrate - their own league's
    history, a paid site's export), a two-column paste keeps the entire
    value/availability half of this board working.
    """
    if uploaded_file is None:
        return pd.DataFrame()
    # An already-parsed frame passes straight through. The UI parses an
    # upload the moment it arrives and holds the frame rather than the file
    # handle, because Streamlit's uploader widget stops existing as soon as
    # the settings panel is closed - and an ADP source that silently reverted
    # when you collapsed the settings would be a bad surprise mid-draft.
    if isinstance(uploaded_file, pd.DataFrame):
        return uploaded_file if 'ADP' in uploaded_file.columns else pd.DataFrame()
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    name_col = next((lower[c] for c in lower if c in ('player', 'player name', 'name', 'full name')), None)
    if not name_col:
        return pd.DataFrame()
    adp_col = next((lower[c] for c in lower if c in ('adp', 'avg', 'average', 'avg pick', 'average pick', 'rank', 'rk', 'overall')), None)
    out = pd.DataFrame({'Player': df[name_col].astype(str).str.strip()})
    out['ADP'] = pd.to_numeric(df[adp_col], errors='coerce') if adp_col else range(1, len(df) + 1)
    sd_col = next((lower[c] for c in lower if c in ('stdev', 'std dev', 'sd', 'adp sd')), None)
    if sd_col:
        out['ADP SD'] = pd.to_numeric(df[sd_col], errors='coerce')
    pos_col = next((lower[c] for c in lower if c in ('pos', 'position')), None)
    if pos_col:
        out['Pos'] = df[pos_col].astype(str).str.upper().str.replace(r'\d+$', '', regex=True)
    out = out.dropna(subset=['ADP'])
    out = out[out['Player'].ne('') & out['Player'].str.lower().ne('nan')]
    return out.sort_values('ADP').reset_index(drop=True)


# What each ADP source actually measures. Shown verbatim in the UI, because
# "ADP" from two sources can differ by two full rounds on the same player and
# the reason is always the sample, never the arithmetic.
ADP_SOURCE_NOTES = {
    'FFA import': "Fantasy Football Advice's own draft corpus, human picks, cross-platform "
                  "composite. Closest to where players actually go in real drafts.",
    'Uploaded CSV': "Your own file.",
    'FantasyPros consensus ADP': "Averaged across NFFC, Sleeper, ESPN, CBS and RTSports - the "
                                 "platforms real leagues draft on, high-stakes money drafts "
                                 "included. One consensus per format rather than per league size.",
    'FantasyPros export (local)': "The same FantasyPros consensus, recovered from the ranking "
                                  "CSVs in rankings/ (RK plus their ECR-vs-ADP column). Needs no "
                                  "network - as fresh as your last export.",
    'Fantasy Football Calculator': "Mock drafts run on fantasyfootballcalculator.com. A free "
                                   "public mock site skews casual - tight ends and quarterbacks "
                                   "consistently slide later there than in real or sharp drafts. "
                                   "Kept only as a manual override.",
    'ECR estimate': "Not ADP. Consensus expert rank standing in as a draft-order estimate so "
                    "the availability model still works.",
}

# What the ADP-source dropdown offers, in preference order. Fantasy Football
# Calculator is still selectable, but it is no longer part of Auto - see
# fetch_adp.
ADP_SOURCE_CHOICES = [
    'Auto', 'FFA import', 'FantasyPros consensus ADP', 'FantasyPros export (local)',
    'Uploaded CSV', 'Fantasy Football Calculator',
]


def fetch_adp(scoring, num_teams, is_superflex, year, uploaded=None, ffa_df=None,
              source='Auto'):
    """
    Resolve ADP from the best market source available.

    PREFERENCE ORDER, AND WHY IT CHANGED AGAIN: Fantasy Football Calculator
    is now out of the automatic chain entirely. Its ADP is computed from mock
    drafts run on its own free public site, and while those are real humans,
    they are people mock-drafting for free on a calculator site - the
    resulting board drifts hard from where players actually go, sliding tight
    ends and quarterbacks a full round or more. Everything downstream
    (value-vs-market, survival probability, VONA, the opponent model)
    inherits that error while still looking authoritative, which is worse
    than a visibly missing column. It stays selectable by hand, never
    automatic.

    The chain is now: your own upload, then an FFA import (a cross-platform
    composite of real drafts), then FantasyPros' consensus ADP live, then the
    same FantasyPros consensus recovered offline from the ranking exports
    already sitting in rankings/. That last step is what makes the board
    work on a dead network, which is the case that actually happens on draft
    night.

    Sleeper is deliberately absent: it was built on their 'search_rank'
    field, a popularity ordering rather than a draft position, and it put the
    consensus QB1 second overall. A number that looks like ADP and is
    labelled ADP but was never measured from drafts is worse than none.

    An absent ADP is left absent rather than silently substituting ECR into
    the ADP column. ECR (what analysts think) and ADP (what drafters do)
    disagreeing IS the signal this board sells.
    """
    def _meta(name, **extra):
        out = {'source': name, 'error': None, 'teams': num_teams,
               'note': ADP_SOURCE_NOTES.get(name, '')}
        out.update(extra)
        return out

    if source in ('Auto', 'Uploaded CSV') and uploaded is not None:
        up = parse_adp_upload(uploaded)
        if not up.empty:
            return up, _meta('Uploaded CSV')

    if source in ('Auto', 'FFA import') and ffa_df is not None and not ffa_df.empty:
        ffa_adp = ffa_adp_frame(ffa_df)
        if not ffa_adp.empty:
            return ffa_adp, _meta('FFA import')

    fallback_error = None
    if source in ('Auto', 'FantasyPros consensus ADP'):
        df, meta = fetch_fantasypros_adp(scoring, is_superflex, year)
        if not df.empty:
            meta = dict(meta)
            meta['note'] = ADP_SOURCE_NOTES['FantasyPros consensus ADP']
            return df, meta
        fallback_error = (meta or {}).get('error')

    if source in ('Auto', 'FantasyPros consensus ADP', 'FantasyPros export (local)'):
        local = load_fantasypros_local_adp(scoring)
        if not local.empty:
            meta = _meta('FantasyPros export (local)')
            if fallback_error:
                # Not an error state - the board is fully populated - but the
                # user should know they're reading last week's export rather
                # than today's consensus.
                meta['note'] += f"  (live FantasyPros unreachable: {str(fallback_error)[:60]})"
            return local, meta

    if source == 'Fantasy Football Calculator':
        df, meta = fetch_ffc_adp(scoring, num_teams, is_superflex, year)
        meta = dict(meta or {})
        meta['note'] = ADP_SOURCE_NOTES['Fantasy Football Calculator']
        return df, meta

    return pd.DataFrame(), {'source': source,
                            'error': fallback_error or 'no ADP source available'}


def ffa_adp_frame(ffa_df):
    """
    Turn an FFA import into an ADP frame the board can consume.

    Prefers their `adpComposite` over the bare `adp`: the composite is drawn
    across the major host platforms rather than one, which is what makes it a
    market number rather than one site's habits. Falls back to `adp` where the
    composite is missing.
    """
    if ffa_df is None or ffa_df.empty:
        return pd.DataFrame()
    composite = pd.to_numeric(ffa_df.get('FFA ADP Composite'), errors='coerce')
    plain = pd.to_numeric(ffa_df.get('FFA ADP'), errors='coerce')
    adp = composite.fillna(plain) if composite is not None else plain
    if adp is None or adp.notna().sum() < 10:
        return pd.DataFrame()
    out = pd.DataFrame({
        'Player': ffa_df['Player'],
        'Pos': ffa_df.get('Pos'),
        'ADP': adp,
    }).dropna(subset=['ADP'])
    # No published spread in their export, so the availability model falls
    # back to its rank-scaled estimate (see effective_adp_sd).
    return out.sort_values('ADP').reset_index(drop=True)


def ecr_age_days(ecr_raw):
    """How stale the consensus scrape is, in days, or None if unknown."""
    if ecr_raw is None or ecr_raw.empty or 'scrape_date' not in ecr_raw.columns:
        return None
    try:
        scraped = pd.to_datetime(ecr_raw['scrape_date'].dropna().max())
        return int((pd.Timestamp.utcnow().tz_localize(None).normalize() - scraped.normalize()).days)
    except Exception:
        return None


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_injury_report(year):
    """
    Current injury designations from nflverse.

    Deliberately TTL'd far shorter than the market data: a Wednesday
    practice report or a Saturday IR move is exactly the sort of thing that
    should invalidate a cached board within the hour, not within the day.

    Falls back a season at a time when the requested year has no feed yet.
    That is the normal case during draft season, not an edge case: you draft
    the 2026 season in August 2026, and nflverse has no 2026 injury rows
    until games start - so asking for the season you're drafting raises
    "Season must be between 2009 and 2025" every single time. Last season's
    designations are stale for lineup decisions but still tell you who ended
    the year hurt, which is exactly the draft-relevant question.
    """
    attempted = []
    for candidate in (int(year), int(year) - 1, int(year) - 2):
        try:
            import nflreadpy
            df = nflreadpy.load_injuries([candidate]).to_pandas()
        except Exception as exc:
            attempted.append(f"{candidate}: {type(exc).__name__}")
            continue
        if df is not None and not df.empty:
            year = candidate
            break
    else:
        return pd.DataFrame(), "; ".join(attempted) or "no injury data"
    name_col = next((c for c in ('full_name', 'player_name', 'gsis_id') if c in df.columns), None)
    if not name_col:
        return pd.DataFrame(), "unexpected schema"
    # Latest report per player - the injuries feed is one row per player per
    # week, and a September board wants the current designation, not a
    # player's whole injury history stacked into the board.
    if 'week' in df.columns:
        df = df.sort_values('week').groupby(name_col, as_index=False).last()
    out = pd.DataFrame({
        'Player': df[name_col].astype(str).str.strip(),
        'Injury Status': df.get('report_status', df.get('game_status', pd.Series(index=df.index, dtype=str))),
        'Injury Detail': df.get('report_primary_injury', pd.Series(index=df.index, dtype=str)),
        'Practice Status': df.get('practice_status', pd.Series(index=df.index, dtype=str)),
    })
    out = out[out['Injury Status'].notna() & out['Injury Status'].astype(str).str.strip().ne('')]
    out.attrs['season'] = year
    return out.reset_index(drop=True), None


@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_player_news(limit=60):
    """
    Recent NFL headlines from ESPN's public site API.

    Scoped intentionally narrow: this is a headline feed keyed to player
    names so a board row can carry a "there is news on this guy" flag and
    the draft room can show what it is. It is NOT a replacement for a real
    beat-reporter feed, and it's the first thing to fail on a locked-down
    network, so every consumer treats an empty return as "no news column"
    rather than an error.
    """
    # Several candidates because ESPN's public JSON endpoints move around and
    # some networks 403 the site.api host specifically. Tried in order; the
    # first that answers wins.
    content, err = _fetch_bytes([
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news",
        "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/news",
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/news",
    ], params={'limit': limit}, headers={
        # ESPN's edge rejects requests without a browser-ish Accept and
        # Referer on some paths.
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.espn.com/nfl/',
        'Origin': 'https://www.espn.com',
    })
    if content is None:
        return pd.DataFrame(), err
    try:
        import json
        payload = json.loads(content)
        articles = payload.get('articles', []) if isinstance(payload, dict) else []
        rows = []
        for art in articles:
            headline = art.get('headline') or art.get('title') or ''
            desc = art.get('description') or ''
            published = art.get('published') or ''
            link = ''
            links = art.get('links') or {}
            if isinstance(links, dict):
                link = (links.get('web') or {}).get('href', '')
            names = []
            for cat in (art.get('categories') or []):
                if isinstance(cat, dict) and cat.get('type') == 'athlete':
                    nm = (cat.get('athlete') or {}).get('displayName') or cat.get('description')
                    if nm:
                        names.append(str(nm).strip())
            rows.append({'Headline': headline, 'Description': desc, 'Published': published,
                         'Link': link, 'Players': names})
        if not rows:
            return pd.DataFrame(), "no articles"
        return pd.DataFrame(rows), None
    except Exception as exc:
        return pd.DataFrame(), f"parse failed: {exc}"


def news_by_player(news_df):
    """Flatten the headline feed into player -> list of (headline, link)."""
    if news_df is None or news_df.empty or 'Players' not in news_df.columns:
        return {}
    out = {}
    for _, row in news_df.iterrows():
        for name in (row['Players'] or []):
            out.setdefault(name, []).append((row['Headline'], row.get('Link', '')))
    return out


def normal_sf(x):
    """
    P(Z > x) for a standard normal, via math.erf.

    Written out rather than pulling scipy in for one function - scipy is a
    heavy dependency for a survival function, and requirements.txt is
    deliberately short. Used by the pick-availability model in
    data.draft_board, which needs this per player per pick.
    """
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def vectorized_normal_sf(x):
    """Array version of normal_sf - the board evaluates it over every row at once."""
    arr = np.asarray(x, dtype=float)
    with np.errstate(invalid='ignore'):
        return 0.5 * np.vectorize(math.erfc)(arr / math.sqrt(2.0))
