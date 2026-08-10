"""
Where every manual import in this app comes from, in one place.

WHY A REGISTRY RATHER THAN TEXT AT EACH UPLOADER. There are eleven places in
this app that ask you to go and fetch something, and the instructions for
them had drifted into three different styles: some named a URL in prose,
some described a page without naming it, and some said nothing at all and
assumed you remembered. That is fine the week you build it and useless four
months later at 11pm before a draft.

One entry per source, each with a clickable title, so the answer to "where
do I get this again" is a link rather than a memory.

A URL is optional and deliberately so. Two of these are subscription
products whose export lives behind a login that has no stable public page
worth linking, and a fabricated link is worse than none - it sends you
somewhere wrong with confidence.
"""
import collections

ImportSource = collections.namedtuple('ImportSource', 'title url how')


IMPORT_SOURCES = {
    # --- rankings and ADP ---------------------------------------------------
    'fantasypros_draft': ImportSource(
        'FantasyPros Draft Rankings',
        'https://www.fantasypros.com/nfl/rankings/consensus-cheatsheets.php',
        'Pick your scoring format, then **Download CSV**. Include BEST / WORST / '
        'STD DEV if the export offers them — Upside, Bust and Risk are built from '
        'that spread.',
    ),
    'fantasypros_weekly': ImportSource(
        'FantasyPros Weekly Rankings',
        'https://www.fantasypros.com/nfl/rankings/ppr-flex.php',
        'This week, your scoring format, then **Download CSV**.',
    ),
    'fantasypros_adp': ImportSource(
        'FantasyPros ADP',
        'https://www.fantasypros.com/nfl/adp/overall.php',
        'Any CSV with a player-name column and an ADP or rank column works — this '
        'is just the one that is easiest to export.',
    ),
    'fantasypros_cheatsheet': ImportSource(
        'FantasyPros Draft Wizard',
        'https://draftwizard.fantasypros.com/football/draft-assistant/',
        'Open the draft assistant and save the `getCheatSheet` response. Add the '
        "draft room's player array from the same capture and every player "
        'resolves, instead of about 80%.',
    ),
    'ffa': ImportSource(
        'Fantasy Football Advice',
        None,
        'Export the players JSON from your subscription and upload it. Their '
        'projected STAT LINE gets re-scored under your league settings, so the '
        'half-PPR total in the file never reaches your board.',
    ),

    # --- season-long book lines --------------------------------------------
    'underdog': ImportSource(
        'Underdog — over/under lines',
        'https://api.underdogfantasy.com/beta/v5/over_under_lines',
        'Save the JSON. One payload carries both the season-long and the weekly '
        'board, so this same file serves Live Odds too.',
    ),
    'prizepicks_season': ImportSource(
        'PrizePicks — season (NFLSZN)',
        'https://api.prizepicks.com/leagues',
        'Find the **NFLSZN** league id in that list, then open '
        '`api.prizepicks.com/projections?league_id=<id>&per_page=1000` and save it. '
        'League 9 is the WEEKLY board and belongs in Live Odds, not here.',
    ),
    'draftkings_season': ImportSource(
        'DraftKings — player futures',
        'https://sportsbook-nash.draftkings.com/api/sportscontent/dkusoh/v1/leagues/88808/categories/1759/subcategories/20168',
        'One call per stat — swap the last number: passing yards 17147, passing '
        'TDs 17148, rushing yards 17223, rushing TDs 17224, receiving yards 17314, '
        'receiving TDs 17315, receptions 20168, sacks 17316. Drop them all in at '
        'once, or one at a time; they accumulate. '
        '`python scripts/probe_season_odds.py --draftkings --save-dir .` fetches the lot.',
    ),
    'fanduel': ImportSource(
        'FanDuel — NFL page',
        'https://sbapi.oh.sportsbook.fanduel.com/api/content-managed-page?page=CUSTOM&customPageId=nfl&_ak=FhMFpcPWXMeyZxOx',
        'Save the JSON. One page carries every season-long player prop; the state '
        "in the subdomain doesn't matter for this content.",
    ),
    'pinnacle': ImportSource(
        'Pinnacle — NFL matchups',
        'https://guest.api.arcadia.pinnacle.com/0.1/leagues/889/matchups',
        'Save the JSON. Sharpest lines of the five, but this feed carries no '
        'prices, so nothing from it can be devigged.',
    ),

    # --- weekly book lines --------------------------------------------------
    'prizepicks_weekly': ImportSource(
        'PrizePicks — weekly (league 9)',
        'https://api.prizepicks.com/projections?league_id=9&per_page=1000',
        'Save the JSON. **League 9 is the weekly board** — the season one (NFLSZN) '
        'is a different league and goes in Draft HQ. This is the one book that '
        'refuses this app outright, so a saved file is the only way in.',
    ),
    'draftkings_weekly': ImportSource(
        'DraftKings — weekly boards',
        'https://sportsbook-nash.draftkings.com/api/sportscontent/dkusoh/v1/leagues/88808',
        'Weekly subcategory ids change and are only posted in season. '
        '`python scripts/probe_season_odds.py --weekly` prints the live ones; save '
        'each and drop them in together.',
    ),

    # --- keys and ids -------------------------------------------------------
    'odds_api': ImportSource(
        'The Odds API',
        'https://the-odds-api.com/#get-access',
        'Free tier is 500 requests a month. Billing follows markets RETURNED, not '
        'requested, so asking for everything is cheap when little is posted.',
    ),
    'sleeper': ImportSource(
        'Sleeper',
        'https://sleeper.com/',
        'Open your draft in a browser — the id is the last part of the URL.',
    ),
}


def markdown(key, prefix='') -> str:
    """
    One source as a markdown line: linked title, then what to do.

    Returned rather than rendered so this module stays importable without
    Streamlit, which keeps it testable and keeps the registry usable from
    scripts and docs.
    """
    source = IMPORT_SOURCES[key]
    title = f"[{source.title}]({source.url})" if source.url else f"**{source.title}**"
    return f"{prefix}{title} — {source.how}"


def markdown_list(*keys) -> str:
    """Several sources as a bulleted list."""
    return '\n\n'.join(markdown(key, prefix='• ') for key in keys)
