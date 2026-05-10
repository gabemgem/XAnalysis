"""
create_descriptive_stats.py

Generates descriptive statistics tables and figures from tweets and community notes data.

Data sources:
  - full_tweets.csv   (pre-computed by create_distribution.ipynb)
  - MySQL database    (notes and ratings; can be skipped with --skip-db)

Usage:
    python create_descriptive_stats.py
    python create_descriptive_stats.py --tweets-csv full_tweets.csv --output-dir output/descriptive_stats
    python create_descriptive_stats.py --skip-db          # skip DB-dependent figures
    python create_descriptive_stats.py --cache-db         # cache notes/ratings to pickle after loading
    python create_descriptive_stats.py --load-cache       # load notes/ratings from cached pickle
"""

import argparse
import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

# ─── Config ───────────────────────────────────────────────────────────────────

DB_CONFIG = dict(
    host='localhost',
    user='gmaayan',
    password='j.UyNY.3NsP6vrdfy3LNq.2f',
    database='tweets',
    charset='utf8mb4',
    collation='utf8mb4_unicode_ci',
)

CATEGORY_COLS = [
    'misleadingOther',
    'misleadingFactualError',
    'misleadingManipulatedMedia',
    'misleadingOutdatedInformation',
    'misleadingMissingImportantContext',
    'misleadingUnverifiedClaimAsFact',
    'misleadingSatire',
    'notMisleadingOther',
    'notMisleadingFactuallyCorrect',
    'notMisleadingOutdatedButNotWhenWritten',
    'notMisleadingClearlySatire',
    'notMisleadingPersonalOpinion',
]

CATEGORY_LABELS = {
    'misleadingOther':                      'Misleading (Other)',
    'misleadingFactualError':               'Factual Error',
    'misleadingManipulatedMedia':           'Manipulated Media',
    'misleadingOutdatedInformation':        'Outdated Info',
    'misleadingMissingImportantContext':    'Missing Context',
    'misleadingUnverifiedClaimAsFact':      'Unverified Claim',
    'misleadingSatire':                     'Satire (Misleading)',
    'notMisleadingOther':                   'Not Misleading (Other)',
    'notMisleadingFactuallyCorrect':        'Factually Correct',
    'notMisleadingOutdatedButNotWhenWritten': 'Outdated When Written',
    'notMisleadingClearlySatire':           'Clearly Satire',
    'notMisleadingPersonalOpinion':         'Personal Opinion',
}

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_tweets(csv_path):
    tweets = pd.read_csv(csv_path)
    if 'impressions_per_month' in tweets.columns:
        tweets['thousands_impressions_per_month'] = tweets['impressions_per_month'] / 1000
    print(f"Loaded {len(tweets):,} tweets from {csv_path}")
    return tweets


def load_notes_and_ratings_from_db(tweet_ids):
    if not MYSQL_AVAILABLE:
        raise RuntimeError("mysql-connector-python not installed.")

    print("Connecting to database...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Notes
    print(f"Querying notes for {len(tweet_ids):,} tweet IDs...")
    placeholders = ', '.join(['%s'] * len(tweet_ids))
    query = f"""SELECT
        noteId, tweetId, classification,
        misleadingOther, misleadingFactualError, misleadingManipulatedMedia,
        misleadingOutdatedInformation, misleadingMissingImportantContext,
        misleadingUnverifiedClaimAsFact, misleadingSatire,
        notMisleadingOther, notMisleadingFactuallyCorrect,
        notMisleadingOutdatedButNotWhenWritten, notMisleadingClearlySatire,
        notMisleadingPersonalOpinion
    FROM notes WHERE tweetId IN ({placeholders})"""
    cursor.execute(query, tweet_ids)
    rows = cursor.fetchall()
    notes = pd.DataFrame(rows, columns=[
        'noteId', 'tweetId', 'classification',
        'misleadingOther', 'misleadingFactualError', 'misleadingManipulatedMedia',
        'misleadingOutdatedInformation', 'misleadingMissingImportantContext',
        'misleadingUnverifiedClaimAsFact', 'misleadingSatire',
        'notMisleadingOther', 'notMisleadingFactuallyCorrect',
        'notMisleadingOutdatedButNotWhenWritten', 'notMisleadingClearlySatire',
        'notMisleadingPersonalOpinion',
    ])
    notes = notes.dropna(subset=['classification'])
    print(f"  Retrieved {len(notes):,} notes.")

    note_ids = [str(nid) for nid in notes['noteId'].unique()]

    # Ratings (only helpfulnessLevel needed)
    print(f"Querying ratings for {len(note_ids):,} note IDs (this may take a minute)...")
    r_placeholders = ', '.join(['%s'] * len(note_ids))
    query = f"SELECT noteId, helpfulnessLevel FROM note_ratings WHERE noteId IN ({r_placeholders})"
    cursor.execute(query, note_ids)
    rows = cursor.fetchall()
    ratings = pd.DataFrame(rows, columns=['noteId', 'helpfulnessLevel'])
    print(f"  Retrieved {len(ratings):,} ratings.")

    cursor.close()
    conn.close()
    return notes, ratings


def load_notes_and_ratings_from_cache(cache_path):
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    notes, ratings = data['notes'], data['ratings']
    print(f"Loaded {len(notes):,} notes and {len(ratings):,} ratings from cache.")
    return notes, ratings


def save_cache(notes, ratings, cache_path):
    with open(cache_path, 'wb') as f:
        pickle.dump({'notes': notes, 'ratings': ratings}, f)
    print(f"Cached notes/ratings to {cache_path}")


def compute_note_rating_counts(notes, ratings):
    """Add agree/disagree columns to notes dataframe (in-place copy)."""
    agree = (ratings[ratings['helpfulnessLevel'] == 'HELPFUL']
             .groupby('noteId').size().rename('agree'))
    disagree = (ratings[ratings['helpfulnessLevel'] == 'NOT_HELPFUL']
                .groupby('noteId').size().rename('disagree'))
    df = notes.copy()
    df['agree'] = df['noteId'].map(agree).fillna(0).astype(int)
    df['disagree'] = df['noteId'].map(disagree).fillna(0).astype(int)
    return df


# ─── Descriptive tables ───────────────────────────────────────────────────────

def table_tweet_stats(tweets, tables_dir):
    cols = [c for c in [
        'impression_count', 'impressions_per_month', 'thousands_impressions_per_month',
        'action_count', 'action_count_per_1000_impressions',
        'agreement_score', 'ext_per_month', 'v_score', 'e_score',
        'interaction_score',
    ] if c in tweets.columns]

    stats = tweets[cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    stats.columns = ['count', 'mean', 'std', 'min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max']
    path = os.path.join(tables_dir, 'tweet_summary_stats.csv')
    stats.to_csv(path)
    print(f"\nTweet summary statistics ({len(tweets):,} posts):")
    print(stats.to_string())
    print(f"  -> {path}")


def table_tweet_correlations(tweets, tables_dir):
    cols = [c for c in [
        'thousands_impressions_per_month', 'action_count_per_1000_impressions',
        'agreement_score', 'ext_per_month', 'v_score', 'e_score',
    ] if c in tweets.columns]
    corr = tweets[cols].corr().round(4)
    path = os.path.join(tables_dir, 'tweet_correlations.csv')
    corr.to_csv(path)
    print(f"\nTweet metric correlations:")
    print(corr.to_string())
    print(f"  -> {path}")


def table_note_classification(notes, tables_dir):
    clf = (notes['classification'].value_counts()
           .rename_axis('classification').reset_index(name='count'))
    clf['pct'] = (clf['count'] / clf['count'].sum() * 100).round(2)
    path = os.path.join(tables_dir, 'note_classification.csv')
    clf.to_csv(path, index=False)
    print(f"\nNote classification breakdown ({len(notes):,} notes):")
    print(clf.to_string(index=False))
    print(f"  -> {path}")


def table_note_categories(notes, tables_dir):
    rows = []
    for cat in CATEGORY_COLS:
        if cat not in notes.columns:
            continue
        count = int((notes[cat] == 1).sum())
        rows.append({'category': cat, 'label': CATEGORY_LABELS[cat], 'count': count,
                     'pct': round(count / len(notes) * 100, 2)})
    df = pd.DataFrame(rows).sort_values('count', ascending=False)
    path = os.path.join(tables_dir, 'note_category_frequency.csv')
    df.to_csv(path, index=False)
    print(f"\nNote category frequency:")
    print(df.to_string(index=False))
    print(f"  -> {path}")


def table_rating_distribution(ratings, tables_dir):
    dist = (ratings['helpfulnessLevel'].value_counts()
            .rename_axis('helpfulnessLevel').reset_index(name='count'))
    dist['pct'] = (dist['count'] / dist['count'].sum() * 100).round(2)
    path = os.path.join(tables_dir, 'rating_distribution.csv')
    dist.to_csv(path, index=False)
    print(f"\nRating helpfulness distribution ({len(ratings):,} total):")
    print(dist.to_string(index=False))
    print(f"  -> {path}")


def table_category_rating_totals(notes_r, tables_dir):
    rows = []
    for cat in CATEGORY_COLS:
        if cat not in notes_r.columns:
            continue
        subset = notes_r[notes_r[cat] == 1]
        rows.append({
            'category': cat,
            'label': CATEGORY_LABELS[cat],
            'n_notes': len(subset),
            'total_agree': int(subset['agree'].sum()),
            'total_disagree': int(subset['disagree'].sum()),
        })
    df = pd.DataFrame(rows).sort_values('n_notes', ascending=False)
    df['agree_rate'] = (df['total_agree'] / (df['total_agree'] + df['total_disagree'])).round(4)
    path = os.path.join(tables_dir, 'category_rating_totals.csv')
    df.to_csv(path, index=False)
    print(f"\nRating totals by note category:")
    print(df.to_string(index=False))
    print(f"  -> {path}")
    return df


# ─── Figures ──────────────────────────────────────────────────────────────────

def _save(fig, figures_dir, name):
    path = os.path.join(figures_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_note_scatter(notes_r, figures_dir):
    """Scatter of agree vs disagree rating counts per note, colored by classification."""
    clf_colors = {
        'MISINFORMED_OR_POTENTIALLY_MISLEADING': ('#d62728', 'Misleading'),
        'NOT_MISLEADING':                        ('#2ca02c', 'Not Misleading'),
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    for clf, (color, label) in clf_colors.items():
        grp = notes_r[notes_r['classification'] == clf]
        ax.scatter(grp['disagree'], grp['agree'],
                   s=5, alpha=0.25, color=color, label=f'{label} (n={len(grp):,})',
                   rasterized=True)

    lim = max(notes_r['agree'].max(), notes_r['disagree'].max()) * 1.05
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.plot([0, lim], [0, lim], '--', color='gray', linewidth=1.2, label='1:1 line')

    ax.set_xlabel('Ratings Disagreeing with Note (NOT_HELPFUL)', fontsize=11)
    ax.set_ylabel('Ratings Agreeing with Note (HELPFUL)', fontsize=11)
    ax.set_title('Community Note Rating Agreement\nby Note Classification', fontsize=12)
    ax.legend(markerscale=3, fontsize=10)
    ax.grid(True, alpha=0.3)

    _save(fig, figures_dir, 'note_agree_vs_disagree.png')


def fig_category_bar(notes_r, figures_dir):
    """Paired bar chart: total agree/disagree ratings per note category, log scale."""
    rows = []
    for cat in CATEGORY_COLS:
        if cat not in notes_r.columns:
            continue
        subset = notes_r[notes_r[cat] == 1]
        rows.append({
            'label': CATEGORY_LABELS[cat],
            'agree': int(subset['agree'].sum()),
            'disagree': int(subset['disagree'].sum()),
        })
    df = pd.DataFrame(rows)
    df['total'] = df['agree'] + df['disagree']
    df = df.sort_values('total', ascending=False)

    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, df['agree'],    width, label='Agree (HELPFUL)',     color='#2ca02c', alpha=0.85)
    ax.bar(x + width / 2, df['disagree'], width, label='Disagree (NOT_HELPFUL)', color='#d62728', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df['label'], rotation=35, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylabel('Total Ratings (log scale)', fontsize=11)
    ax.set_title('Total Agree vs Disagree Ratings by Note Category', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    _save(fig, figures_dir, 'category_rating_bar.png')


def fig_impressions_vs_agreement(tweets, figures_dir):
    """Scatter: thousands of impressions/month (log x) vs agreement score per 1K impressions."""
    df = tweets[['thousands_impressions_per_month', 'agreement_score']].dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['thousands_impressions_per_month'], df['agreement_score'],
               s=10, alpha=0.4, color='steelblue', rasterized=True)
    ax.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Thousands of Impressions per Month (log scale)', fontsize=11)
    ax.set_ylabel('Agreement Score per 1K Impressions', fontsize=11)
    ax.set_title('Post Reach vs Community Note Agreement Score', fontsize=12)
    ax.grid(True, alpha=0.3)

    _save(fig, figures_dir, 'impressions_vs_agreement.png')


def fig_impressions_vs_actions(tweets, figures_dir):
    """Scatter: thousands of impressions/month (log x) vs actions per 1K impressions."""
    df = tweets[['thousands_impressions_per_month', 'action_count_per_1000_impressions']].dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['thousands_impressions_per_month'], df['action_count_per_1000_impressions'],
               s=10, alpha=0.4, color='darkorange', rasterized=True)
    ax.set_xscale('log')
    ax.set_xlabel('Thousands of Impressions per Month (log scale)', fontsize=11)
    ax.set_ylabel('Actions per 1K Impressions', fontsize=11)
    ax.set_title('Post Reach vs Engagement Rate', fontsize=12)
    ax.grid(True, alpha=0.3)

    _save(fig, figures_dir, 'impressions_vs_actions.png')


def fig_agreement_vs_actions_by_reach(tweets, figures_dir):
    """Scatter: agreement score vs actions per 1K impressions, color = reach (log scale)."""
    cols = ['thousands_impressions_per_month', 'agreement_score', 'action_count_per_1000_impressions']
    df = tweets[cols].dropna()

    # Clip extreme agreement scores so color gradient is visible
    p99_reach = df['thousands_impressions_per_month'].quantile(0.99)
    reach_norm = mcolors.LogNorm(
        vmin=df['thousands_impressions_per_month'].clip(lower=0.001).min(),
        vmax=df['thousands_impressions_per_month'].clip(upper=p99_reach).max(),
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        df['agreement_score'],
        df['action_count_per_1000_impressions'],
        c=df['thousands_impressions_per_month'].clip(lower=0.001),
        s=12, alpha=0.5, cmap='plasma', norm=reach_norm,
        rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Thousands of Impressions per Month (log scale)', fontsize=10)

    ax.axvline(0, color='gray', linewidth=0.9, linestyle='--', alpha=0.6)
    ax.set_xlabel('Agreement Score per 1K Impressions', fontsize=11)
    ax.set_ylabel('Actions per 1K Impressions', fontsize=11)
    ax.set_title('Note Agreement vs Engagement,\ncolored by Post Reach', fontsize=12)
    ax.grid(True, alpha=0.3)

    _save(fig, figures_dir, 'agreement_vs_actions_by_reach.png')


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tweets-csv',  default='full_tweets.csv',
                   help='Path to full_tweets.csv (default: full_tweets.csv)')
    p.add_argument('--output-dir',  default='output/descriptive_stats',
                   help='Root output directory (default: output/descriptive_stats)')
    p.add_argument('--skip-db',     action='store_true',
                   help='Skip figures and tables that require database access')
    p.add_argument('--cache-db',    action='store_true',
                   help='Save notes/ratings to a pickle cache after loading from DB')
    p.add_argument('--load-cache',  action='store_true',
                   help='Load notes/ratings from pickle cache instead of DB')
    return p.parse_args()


def main():
    args = parse_args()

    figures_dir = os.path.join(args.output_dir, 'figures')
    tables_dir  = os.path.join(args.output_dir, 'tables')
    cache_path  = os.path.join(args.output_dir, 'notes_ratings_cache.pkl')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir,  exist_ok=True)

    # ── Tweets ────────────────────────────────────────────────────────────────
    tweets = load_tweets(args.tweets_csv)

    print("\n=== Descriptive Tables: Tweets ===")
    table_tweet_stats(tweets, tables_dir)
    table_tweet_correlations(tweets, tables_dir)

    print("\n=== Figures: Posts ===")
    fig_impressions_vs_agreement(tweets, figures_dir)
    fig_impressions_vs_actions(tweets, figures_dir)
    fig_agreement_vs_actions_by_reach(tweets, figures_dir)

    # ── Notes + Ratings ───────────────────────────────────────────────────────
    if args.skip_db:
        print("\nSkipping DB-dependent tables and figures (--skip-db).")
        return

    notes, ratings = None, None
    try:
        if args.load_cache:
            notes, ratings = load_notes_and_ratings_from_cache(cache_path)
        else:
            tweet_ids = tweets['id'].dropna().astype(int).tolist()
            notes, ratings = load_notes_and_ratings_from_db(tweet_ids)
            if args.cache_db:
                save_cache(notes, ratings, cache_path)
    except Exception as exc:
        print(f"\nWarning: could not load notes/ratings ({exc})")
        print("Skipping DB-dependent figures. Use --skip-db to suppress this message.")
        return

    print("\n=== Descriptive Tables: Notes & Ratings ===")
    table_note_classification(notes, tables_dir)
    table_note_categories(notes, tables_dir)
    table_rating_distribution(ratings, tables_dir)

    notes_r = compute_note_rating_counts(notes, ratings)
    table_category_rating_totals(notes_r, tables_dir)

    print("\n=== Figures: Notes & Ratings ===")
    fig_note_scatter(notes_r, figures_dir)
    fig_category_bar(notes_r, figures_dir)

    print("\nAll outputs written to:", args.output_dir)


if __name__ == '__main__':
    main()
