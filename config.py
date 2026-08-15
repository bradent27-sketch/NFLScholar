"""
Shared constants: team metadata, stat-formatting rules, and the design-system
THEME token dict. THEME is synthesized from design.md/DESIGN.md (Sleeper -
primary reference), "Dunks and Threes Design MD.md" (dense stat-table
conventions, monospace numerals), and "Spotify Design MD.md" (spacing rhythm,
restrained card elevation). Sleeper is the named target aesthetic; the other
two fill in table-density detail Sleeper's own spec doesn't cover.
"""

TEAM_CONFIG = {
    'ARI': {'id': '1',  'name': 'Arizona Cardinals', 'color': '#97233F'},
    'ATL': {'id': '2',  'name': 'Atlanta Falcons', 'color': '#A71930'},
    'BAL': {'id': '33', 'name': 'Baltimore Ravens', 'color': '#241773'},
    'BUF': {'id': '2',  'name': 'Buffalo Bills', 'color': '#00338D'},
    'CAR': {'id': '29', 'name': 'Carolina Panthers', 'color': '#0085CA'},
    'CHI': {'id': '3',  'name': 'Chicago Bears', 'color': '#0B2265'},
    'CIN': {'id': '4',  'name': 'Cincinnati Bengals', 'color': '#FB4F14'},
    'CLE': {'id': '5',  'name': 'Cleveland Browns', 'color': '#311D00'},
    'DAL': {'id': '6',  'name': 'Dallas Cowboys', 'color': '#869397'},
    'DEN': {'id': '7',  'name': 'Denver Broncos', 'color': '#FB4F14'},
    'DET': {'id': '8',  'name': 'Detroit Lions', 'color': '#0076B6'},
    'GB':  {'id': '9',  'name': 'Green Bay Packers', 'color': '#203731'},
    'HOU': {'id': '34', 'name': 'Houston Texans', 'color': '#03202F'},
    'IND': {'id': '11', 'name': 'Indianapolis Colts', 'color': '#002C5F'},
    'JAX': {'id': '30', 'name': 'Jacksonville Jaguars', 'color': '#006778'},
    'KC':  {'id': '12', 'name': 'Kansas City Chiefs', 'color': '#E31837'},
    'LV':  {'id': '13', 'name': 'Las Vegas Raiders', 'color': '#A5ACAF'},
    'LAC': {'id': '24', 'name': 'Los Angeles Chargers', 'color': '#0080C6'},
    'LA':  {'id': '14', 'name': 'Los Angeles Rams', 'color': '#003594'},
    'MIA': {'id': '15', 'name': 'Miami Dolphins', 'color': '#008E97'},
    'MIN': {'id': '16', 'name': 'Minnesota Vikings', 'color': '#4F2683'},
    'NE':  {'id': '17', 'name': 'New England Patriots', 'color': '#002244'},
    'NO':  {'id': '18', 'name': 'New Orleans Saints', 'color': '#D3BC8D'},
    'NYG': {'id': '19', 'name': 'New York Giants', 'color': '#0B2265'},
    'NYJ': {'id': '20', 'name': 'New York Jets', 'color': '#125740'},
    'PHI': {'id': '21', 'name': 'Philadelphia Eagles', 'color': '#004C54'},
    'PIT': {'id': '23', 'name': 'Pittsburgh Steelers', 'color': '#FFB612'},
    'SF':  {'id': '25', 'name': 'San Francisco 49ers', 'color': '#AA0000'},
    'SEA': {'id': '26', 'name': 'Seattle Seahawks', 'color': '#002244'},
    'TB':  {'id': '27', 'name': 'Tampa Bay Buccaneers', 'color': '#D50A0A'},
    'TEN': {'id': '10', 'name': 'Tennessee Titans', 'color': '#4B92DB'},
    'WAS': {'id': '28', 'name': 'Washington Commanders', 'color': '#5A1414'}
}
MASTER_TEAMS_LIST = sorted(list(TEAM_CONFIG.keys()))

# PFF's coverage-scheme exports (defense_coverage_scheme_2025.csv,
# receiving_scheme_2025.csv) use a handful of team codes that don't match
# TEAM_CONFIG's nflverse-style abbreviations - mapped here so the Defensive
# Coverage Correlator (data/coverage_radar.py) can look up the right
# TEAM_CONFIG entry (full name, color) and SumerSports row for a PFF team
# code. Any code not listed here is already identical between the two.
PFF_TEAM_CODE_TO_ABBR = {
    'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU',
}


def pff_team_to_abbr(pff_code):
    return PFF_TEAM_CODE_TO_ABBR.get(pff_code, pff_code)


ABBR_TO_PFF_TEAM_CODE = {abbr: pff_code for pff_code, abbr in PFF_TEAM_CODE_TO_ABBR.items()}


def abbr_to_pff_team(abbr):
    """Reverse of pff_team_to_abbr - needed to pre-select the Defensive
    Coverage Correlator's "Opposing defense (PFF team code)" control from a
    standard nflverse abbreviation (e.g. the Matchup Bridge button on a
    player's bio card, which only knows the standard abbreviation)."""
    return ABBR_TO_PFF_TEAM_CODE.get(abbr, abbr)


# Shared position-group lists used anywhere the app branches on a player's
# position (Player Search's positional matrix + game log column selection).
# Broadened beyond the "clean" position codes (DE/DT/CB/FS/SS/ILB/OLB/NT) to
# also include the coarser generic codes (DL/DB/SAF/LB) that really do show
# up in stats_player_week_*.csv's own 'position' column for some players even
# after the roster-merge fix in data/loaders.load_year_data (see the
# roster_position_group rename there) - some players are only ever tagged
# generically by the source data itself, not just via the merge bug that fix
# addresses, so defensive stats would still silently disappear for exactly
# those players without this broadening.
OLINE_POSITIONS = ['T', 'G', 'C', 'OT', 'OG', 'OL', 'LT', 'RT', 'LG', 'RG']
# Skill-position offense + kickers - used to keep Risers/Waiver Wire and
# Rookie Watch fantasy-relevant (a standard non-IDP league doesn't roster
# defensive players or O-linemen off the waiver wire).
OFFENSE_SKILL_POSITIONS = ['QB', 'RB', 'FB', 'WR', 'TE', 'K']

# How far back every "Season Data Framework" / "Season" dropdown across the
# app reaches - a single source of truth so extending (or pulling back)
# history is a one-line change instead of hunting down 7 separate hardcoded
# lists. 2019 is the current standard depth: nflreadpy's live sources
# (weekly stats, schedules, snap counts, NGS, PFR) all cover this range on
# their own, and pff_imports/ is organized as one subfolder per year back to
# 2019 (see data.loaders.load_pff_year) to match it.
AVAILABLE_SEASONS_WITH_UPCOMING = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019]
AVAILABLE_SEASONS = [2025, 2024, 2023, 2022, 2021, 2020, 2019]

# Every NFL player-prop market key The Odds API documents (as of this
# writing). Not every book posts every market for every game - especially
# far out from kickoff - so this is requested as one big list and whatever
# comes back, comes back; live-tested against a real key and confirmed The
# Odds API silently omits unavailable markets rather than erroring the
# whole request.
ODDS_API_PLAYER_PROP_MARKETS = [
    'player_pass_yds', 'player_pass_tds', 'player_pass_completions', 'player_pass_attempts',
    'player_pass_interceptions', 'player_pass_longest_completion',
    'player_rush_yds', 'player_rush_attempts', 'player_rush_longest',
    'player_reception_yds', 'player_receptions', 'player_reception_longest',
    'player_anytime_td', 'player_1st_td', 'player_last_td',
    'player_kicking_points', 'player_field_goals',
    'player_tackles_assists', 'player_sacks', 'player_solo_tackles',
]
DEFENSIVE_POSITIONS = [
    'DE', 'DT', 'EDGE', 'DL', 'DI', 'ED', 'NT',
    'LB', 'ILB', 'OLB', 'MLB',
    'CB', 'S', 'FS', 'SS', 'DB', 'SAF',
]

# Position-identity accent colors - a fixed color per position GROUP (same
# grouping ui.player_snapshot.position_branch_group already uses: QB / RB /
# WR / TE / OLINE_POSITIONS / DEFENSIVE_POSITIONS / K / [P, LS]), for
# scannability on real HTML surfaces (bio card, Player Compare legend) -
# deliberately NOT applied inside st.dataframe tables, since the grid
# (glide-data-grid) renders cells on an HTML canvas and only reads
# background-color/color/font-weight out of a Styler, not a chip/pill shape
# (see ui.styling.style_depth_chart_table's docstring on the same limit).
# Chosen to stay clear of every color already carrying a DIFFERENT meaning
# elsewhere in THEME - primary cyan (CTA/interactive), secondary blue,
# tertiary orange (alerts), positive green / negative red (stat deltas),
# and the grade-color scale's own red-to-lavender range - so a position tag
# never reads as a grade, a delta, or a call to action.
# Position identity colors. Hues follow the convention the mainstream draft
# tools have converged on (orange QB, green RB, blue WR, purple TE, cyan
# D/ST, pink K), because a drafter reading this board has that mapping
# already memorised from every other tool they use - an app-specific palette
# would be a small tax paid on every glance, under a pick clock.
#
# Each position carries a text color and a darker background, so a position
# can render as a filled chip in a table cell and still hold contrast
# against the navy surface.
POSITION_COLORS = {
    'QB': '#f59e0b',
    'RB': '#22c55e',
    'WR': '#60a5fa',
    'TE': '#c084fc',
    'K': '#f472b6',
    'DST': '#22d3ee',
    'DEF': '#22d3ee',
    'P': '#cbd5e1',
    'LS': '#cbd5e1',
}
POSITION_CHIP_BG = {
    'QB': '#5a3410',
    'RB': '#0f3d22',
    'WR': '#12315c',
    'TE': '#3b1f55',
    'K': '#4d1330',
    'DST': '#0b3a44',
    'DEF': '#0b3a44',
}


def get_position_chip_bg(pos):
    """Darker background pairing for a position chip; None if unmapped."""
    return POSITION_CHIP_BG.get(str(pos).upper())
for _p in OLINE_POSITIONS:
    POSITION_COLORS.setdefault(_p, '#a8a29e')
for _p in DEFENSIVE_POSITIONS:
    POSITION_COLORS.setdefault(_p, '#818cf8')


def get_position_color(pos):
    """Falls back to the theme's neutral secondary-text color for anything
    unmapped (FB, N/A, a future/unknown position code) - never a jarring
    default, and never silently reuses a color that means something else."""
    return POSITION_COLORS.get(str(pos).upper(), THEME['colors']['on_surface_variant'])


# Exact tab label strings, shared between app.py's st.tabs(...) call and any
# tab that needs to programmatically switch the active tab (e.g. a "View
# Player" link from Risers jumping to Player Search) - st.tabs' key-based
# state (Streamlit >=1.59) is driven by matching one of these label strings
# exactly, so every cross-tab jump imports this instead of retyping the
# string, which would silently break navigation on any future rename.
TAB_PLAYER_SEARCH = "PLAYER SEARCH"
TAB_DEPTH_CHARTS = "NFL DEPTH CHARTS"
TAB_DEFENSIVE_YIELD = "DEFENSIVE YIELD SCHEMES"
TAB_RISERS = "RISERS / WAIVER WIRE"
TAB_ROOKIE_WATCH = "ROOKIE WATCH"
TAB_RANKINGS = "WEEKLY RANKINGS"
TAB_LIVE_ODDS = "LIVE ODDS"
TAB_PLAYER_COMPARE = "PLAYER COMPARE"
TAB_MATCHUP_ANALYZER = "MATCHUP ANALYZER"
TAB_DRAFT_HQ = "DRAFT HQ"
TAB_GAME_SLATE = "GAME SLATE"
# Draft HQ sits LAST, deliberately. It's the most capable tab in the app,
# but it answers a question that's only live for a week or two a year -
# every other tab is used all season. Ordering by how good a tab is rather
# than how often it's opened would push the nine in-season tabs one click
# further away for eleven months to save a click during one draft.
#
# It also absorbed the old VORP Draft Sheet, which is why there are nine
# labels here and not ten: that tab computed replacement-level VORP off
# last season's per-game pace, which is a strict subset of what Draft HQ
# does (real projections, market blend, injury and aging adjustments).
# Game Slate sits FIRST, by the same logic that puts Draft HQ last. It is a
# launchpad rather than a report - it answers "who is playing" and then hands
# the game to another tab with both teams filled in - so it is the natural
# first thing opened in a prep session, and being first is what makes it
# useful rather than a detour.
#
# Matchup Analyzer is inserted directly after Player Compare rather than
# anywhere earlier, deliberately: every existing tab keeps the position it
# has always had, and Draft HQ stays last for the reason above. It sits next
# to Compare because the two answer adjacent questions - Compare is
# player-vs-player, Matchup Analyzer is player-vs-defense - so the pair reads
# as one "who do I actually start" neighbourhood.
TAB_LABELS = [
    TAB_GAME_SLATE, TAB_PLAYER_SEARCH, TAB_DEPTH_CHARTS, TAB_DEFENSIVE_YIELD,
    TAB_RISERS, TAB_ROOKIE_WATCH, TAB_RANKINGS, TAB_LIVE_ODDS, TAB_PLAYER_COMPARE,
    TAB_MATCHUP_ANALYZER, TAB_DRAFT_HQ,
]

# Decimal places per stat, approved list: whole numbers for counting stats
# (yards, TDs, attempts, tackles, INTs, forced fumbles), 1 decimal for
# rate/average stats (Y/C, YPC, snap %, fantasy points) and sacks
# specifically since those land on half-increments (1 or 2.5, never partial
# tenths) - not "0 decimals" like the other whole-number defensive stats.
STAT_DECIMALS = {
    'y/c': 1, 'ypc': 1, 'sacks': 1, 'weekly_snap_pct': 1, 'snap_pct_avg': 1, 'fantasy_points': 1,
}

# ==========================================
# DESIGN SYSTEM TOKENS
# ==========================================
THEME = {
    'colors': {
        'surface': '#050921',
        'surface_dim': '#030614',
        'surface_bright': '#0a0f2a',
        'surface_container_lowest': '#020409',
        'surface_container_low': '#0a0f2a',
        'surface_container': '#131b38',
        'surface_container_high': '#1a2447',
        'surface_container_highest': '#242d52',
        'on_surface': '#ffffff',
        # Lightened vs. Sleeper's literal #d8d8d8 spec - this app's tables are
        # far denser than Sleeper's own UI, so secondary text needs more
        # contrast against the navy surface at small sizes (Dunks & Threes'
        # "test contrast, especially on small text" principle).
        'on_surface_variant': '#b9c0e0',
        'outline': '#565d8c',
        'outline_variant': '#2c3260',
        'primary': '#00fff9',
        'on_primary': '#050921',
        'primary_container': '#00ceb8',
        'on_primary_container': '#050921',
        'secondary': '#3860be',
        'on_secondary': '#ffffff',
        'tertiary': '#ffae58',
        'on_tertiary': '#050921',
        'error': '#ff5468',
        'on_error': '#ffffff',
        # Positive/negative stat-delta convention borrowed from Dunks & Threes
        # (VORP, Pct Jump, ECR vs ADP columns) - distinct from the primary
        # cyan accent, which stays reserved for interactive/CTA elements only.
        'positive': '#1ed760',
        'negative': '#ef4444',
    },
    'fonts': {
        # Two families, not three - Inter carries both display (heavy
        # weights) and body (regular) per the Savant-overhaul brief's own
        # explicit "Inter or Roboto" instruction; JetBrains Mono stays for
        # every numeric table cell (a real technical/coding typeface, not a
        # generic monospace - this app's tables are almost entirely
        # numeric, and proportional digits misalign at this density).
        # Self-hosted (see ui.styling.inject_theme's _font_face_css) - a
        # live audit this session found the previous Google Fonts
        # @import url(...) never actually fired a network request in this
        # environment (confirmed via document.fonts.check() + the network
        # log showing zero requests to fonts.googleapis.com), so every
        # custom typeface had been silently falling back to a generic
        # sans-serif/monospace this entire project. Embedding the font
        # files directly removes that external dependency entirely.
        'display': "'Inter', sans-serif",
        'body': "'Inter', sans-serif",
        'mono': "'JetBrains Mono', monospace",
    },
    'radius': {
        'sm': '2px', 'default': '4px', 'md': '10px', 'lg': '16px', 'xl': '20px', 'full': '9999px',
    },
    'spacing': {
        'xs': '4px', 'sm': '12px', 'md': '24px', 'lg': '40px', 'xl': '64px', 'gutter': '24px',
    },
}
