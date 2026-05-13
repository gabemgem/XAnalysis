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
    if 'agreement_score' in tweets.columns:
        before = len(tweets)
        tweets = tweets[tweets['agreement_score'] >= -400]
        print(f"Dropped {before - len(tweets):,} outlier rows with agreement_score < -400")
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
    query = f"SELECT noteId, helpfulnessLevel, agree, disagree FROM note_ratings WHERE noteId IN ({r_placeholders})"
    cursor.execute(query, note_ids)
    rows = cursor.fetchall()
    ratings = pd.DataFrame(rows, columns=['noteId', 'helpfulnessLevel', 'agree', 'disagree'])
    print(f"  Retrieved {len(ratings):,} ratings.")

    cursor.close()
    conn.close()
    
    # Print number of ratings with no agreement signal (both agree and disagree are 0)
    no_signal = ratings[(ratings['agree'] == 0) & (ratings['disagree'] == 0)]
    if len(no_signal) > 0:
        warnings.warn(f"{len(no_signal):,} ratings have no agreement signal (agree=0 and disagree=0).")
    
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
    """Add helpful/somewhat_helpful/not_helpful columns to notes dataframe (in-place copy)."""
    agree = (ratings[ratings['helpfulnessLevel'] == 'HELPFUL']
             .groupby('noteId').size().rename('agree'))
    somewhat = (ratings[ratings['helpfulnessLevel'] == 'SOMEWHAT_HELPFUL']
                .groupby('noteId').size().rename('somewhat_helpful'))
    disagree = (ratings[ratings['helpfulnessLevel'] == 'NOT_HELPFUL']
                .groupby('noteId').size().rename('disagree'))
    df = notes.copy()
    df['agree']           = df['noteId'].map(agree).fillna(0).astype(int)
    df['somewhat_helpful'] = df['noteId'].map(somewhat).fillna(0).astype(int)
    df['disagree']        = df['noteId'].map(disagree).fillna(0).astype(int)
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

    lim = float(np.percentile(
        np.concatenate([notes_r['agree'].values, notes_r['disagree'].values]), 99.9
    )) * 1.05
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
    """Grouped bar chart: helpful / somewhat helpful / not helpful ratings per note category, log scale."""
    rows = []
    for cat in CATEGORY_COLS:
        if cat not in notes_r.columns:
            continue
        subset = notes_r[notes_r[cat] == 1]
        rows.append({
            'label':           CATEGORY_LABELS[cat],
            'helpful':         int(subset['agree'].sum()),
            'somewhat':        int(subset['somewhat_helpful'].sum()),
            'not_helpful':     int(subset['disagree'].sum()),
        })
    df = pd.DataFrame(rows)
    df['total'] = df['helpful'] + df['somewhat'] + df['not_helpful']
    df = df.sort_values('total', ascending=False)

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, df['helpful'],     width, label='Helpful',          color='#2ca02c', alpha=0.85)
    ax.bar(x,         df['somewhat'],    width, label='Somewhat Helpful', color='#ff7f0e', alpha=0.85)
    ax.bar(x + width, df['not_helpful'], width, label='Not Helpful',      color='#d62728', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df['label'], rotation=35, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylabel('Total Ratings (log scale)', fontsize=11)
    ax.set_title('Helpfulness Ratings by Note Category', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    _save(fig, figures_dir, 'category_rating_bar.png')


def fig_multi_note_posts(notes_r, figures_dir):
    """Four-panel figure: notes-per-post distribution, direction mix, combination heatmap, rating sources."""

    MIS = 'MISINFORMED_OR_POTENTIALLY_MISLEADING'
    NM  = 'NOT_MISLEADING'

    is_mis = notes_r['classification'] == MIS
    is_nm  = notes_r['classification'] == NM

    work = pd.DataFrame({
        'tweetId':      notes_r['tweetId'].values,
        'noteId':       notes_r['noteId'].values,
        'n_mis':        is_mis.astype(int).values,
        'n_nm':         is_nm.astype(int).values,
        'agree_mis':    np.where(is_mis, notes_r['agree'],    0),
        'disagree_mis': np.where(is_mis, notes_r['disagree'], 0),
        'agree_nm':     np.where(is_nm,  notes_r['agree'],    0),
        'disagree_nm':  np.where(is_nm,  notes_r['disagree'], 0),
    })

    post = work.groupby('tweetId').agg(
        n_notes          = ('noteId',       'count'),
        n_misleading     = ('n_mis',        'sum'),
        n_not_misleading = ('n_nm',         'sum'),
        agree_mis        = ('agree_mis',    'sum'),
        disagree_mis     = ('disagree_mis', 'sum'),
        agree_nm         = ('agree_nm',     'sum'),
        disagree_nm      = ('disagree_nm',  'sum'),
    ).reset_index()

    n_posts = len(post)
    has_mis = post['n_misleading'] > 0
    has_nm  = post['n_not_misleading'] > 0
    only_mis = int((has_mis & ~has_nm).sum())
    only_nm  = int((~has_mis & has_nm).sum())
    mixed    = int((has_mis & has_nm).sum())

    # Data-driven caps: 95th percentile for bar chart, 90th for heatmap axes
    cap_bars = int(min(post['n_notes'].quantile(0.95), 40))
    cap_heat = int(min(
        max(post['n_misleading'].quantile(0.90), post['n_not_misleading'].quantile(0.90), 5),
        20,
    ))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.flat

    # Panel 1: Distribution of notes per post (individual bars up to cap_bars)
    clipped = post['n_notes'].clip(upper=cap_bars)
    vals, cnts = np.unique(clipped, return_counts=True)
    xlabels1 = [str(int(v)) if v < cap_bars else f'{cap_bars}+' for v in vals]
    ax1.bar(xlabels1, cnts, color='steelblue', alpha=0.85, edgecolor='white')
    # Label only bars that are tall enough to avoid clutter
    threshold = max(cnts) * 0.03
    for i, cnt in enumerate(cnts):
        if cnt >= threshold:
            ax1.text(i, cnt + max(cnts) * 0.005, f'{cnt:,}',
                     ha='center', va='bottom', fontsize=7)
    ax1.set_xlabel('Number of Notes per Post', fontsize=11)
    ax1.set_ylabel('Number of Posts', fontsize=11)
    ax1.set_title(f'Distribution of Notes per Post  (n={n_posts:,} posts,  showing up to {cap_bars}+)', fontsize=10)
    ax1.tick_params(axis='x', labelsize=8, rotation=45 if cap_bars > 10 else 0)
    ax1.grid(True, axis='y', alpha=0.3)

    # Panel 2: Direction mix
    dir_labels = ['Only\nMisleading', 'Only\nNot-Misleading', 'Mixed\n(Both)']
    dir_counts = [only_mis, only_nm, mixed]
    dir_colors = ['#d62728', '#2ca02c', '#ff7f0e']
    bars2 = ax2.bar(dir_labels, dir_counts, color=dir_colors, alpha=0.85, edgecolor='white')
    for bar, cnt in zip(bars2, dir_counts):
        pct = cnt / n_posts * 100
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(dir_counts) * 0.01,
                 f'{cnt:,} ({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('Number of Posts', fontsize=11)
    ax2.set_title('Note Direction Mix per Post', fontsize=11)
    ax2.grid(True, axis='y', alpha=0.3)

    # Panel 3: Heatmap — data-driven cap, log colorscale to handle wide range
    post_h = post.assign(
        mis_cap=post['n_misleading'].clip(upper=cap_heat),
        nm_cap=post['n_not_misleading'].clip(upper=cap_heat),
    )
    heat = (post_h.groupby(['nm_cap', 'mis_cap']).size()
            .unstack(fill_value=0)
            .reindex(index=range(cap_heat + 1), columns=range(cap_heat + 1), fill_value=0))
    heat_arr = heat.to_numpy(dtype=int)

    vmax = float(heat_arr.max()) or 1.0
    vmin_pos = float(heat_arr[heat_arr > 0].min()) if (heat_arr > 0).any() else 1.0
    norm = mcolors.LogNorm(vmin=max(vmin_pos, 0.5), vmax=vmax)
    im = ax3.imshow(heat_arr, aspect='auto', cmap='YlOrRd', origin='lower', norm=norm)

    tick_labels = [str(i) if i < cap_heat else f'{cap_heat}+' for i in range(cap_heat + 1)]
    ax3.set_xticks(range(cap_heat + 1))
    ax3.set_yticks(range(cap_heat + 1))
    ax3.set_xticklabels(tick_labels, fontsize=8)
    ax3.set_yticklabels(tick_labels, fontsize=8)

    # Annotate cells; skip zeros; shrink font for large grids
    cell_fs = max(5, 9 - cap_heat // 3)
    for i in range(heat_arr.shape[0]):
        for j in range(heat_arr.shape[1]):
            v = heat_arr[i, j]
            if v == 0:
                continue
            ax3.text(j, i, f'{v:,}', ha='center', va='center', fontsize=cell_fs,
                     color='white' if v > vmax * 0.4 else 'black')

    fig.colorbar(im, ax=ax3, shrink=0.85, label='Number of Posts (log scale)')
    ax3.set_xlabel('# Misleading Notes', fontsize=11)
    ax3.set_ylabel('# Not-Misleading Notes', fontsize=11)
    ax3.set_title(f'Posts by Note-Count Combination  (axes show up to {cap_heat}+)', fontsize=10)

    # Panel 4: Rating source breakdown
    cat_labels = [
        'Helpful on\nMisleading note',
        'Not-Helpful on\nMisleading note',
        'Helpful on\nNot-Misleading note',
        'Not-Helpful on\nNot-Misleading note',
    ]
    cat_values = [
        int(post['agree_mis'].sum()),
        int(post['disagree_mis'].sum()),
        int(post['agree_nm'].sum()),
        int(post['disagree_nm'].sum()),
    ]
    cat_colors = ['#c00000', '#74c476', '#2ca02c', '#ff6b6b']
    total_ratings = sum(cat_values)

    xpos = np.arange(len(cat_labels))
    bars4 = ax4.bar(xpos, cat_values, color=cat_colors, alpha=0.9, edgecolor='white')
    for bar, cnt in zip(bars4, cat_values):
        pct = cnt / total_ratings * 100 if total_ratings > 0 else 0
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(cat_values) * 0.01,
                 f'{cnt:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax4.set_xticks(xpos)
    ax4.set_xticklabels(cat_labels, fontsize=9)
    ax4.set_ylabel('Total Ratings (across all posts)', fontsize=11)
    ax4.set_title('Rating Sources: Agreement Signal Breakdown', fontsize=11)
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.text(0.5, -0.16,
             'Dark red / light green = rater agrees with note  |  '
             'Dark green / light red = rater contradicts note direction',
             transform=ax4.transAxes, ha='center', fontsize=8, style='italic', color='gray')

    fig.suptitle('Posts with Multiple and Conflicting Community Notes', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, figures_dir, 'multi_note_posts.png')


def fig_escore_new_rating_mapping(notes, ratings, tweets, figures_dir):
    """Side-by-side e-score vs v-score comparing original (-1/0/1) and new (0/0.5/1) rating maps."""

    MIS = 'MISINFORMED_OR_POTENTIALLY_MISLEADING'
    NM  = 'NOT_MISLEADING'

    new_ratings_map = {
        'HELPFUL':          1.0,
        'SOMEWHAT_HELPFUL': 0.5,
        'NOT_HELPFUL':      0.0,
    }
    notes_map = {MIS: -1.0, NM: 1.0}

    # Map each rating to its new numeric score, drop unmapped rows
    r = ratings[['noteId', 'helpfulnessLevel']].copy()
    r['rating_score'] = r['helpfulnessLevel'].map(new_ratings_map)
    r = r.dropna(subset=['rating_score'])

    # Sum ratings per note, add 1 for the original noter
    note_sums = (r.groupby('noteId')['rating_score'].sum()
                 .reset_index()
                 .rename(columns={'rating_score': 'rating_sum'}))
    note_sums['rating_sum'] += 1.0

    # Attach note_score and merge rating sums
    n = notes[['noteId', 'tweetId', 'classification']].copy()
    n['note_score'] = n['classification'].map(notes_map)
    n = n.dropna(subset=['note_score'])
    n = n.merge(note_sums, on='noteId', how='left')
    n['new_ext_score'] = n['note_score'] * n['rating_sum']

    # Sum externality scores per tweet
    ext_by_tweet = (n.groupby('tweetId')['new_ext_score'].sum()
                    .reset_index()
                    .rename(columns={'tweetId': 'id'}))

    need = [c for c in ['id', 'impression_count', 'impressions_per_month', 'v_score', 'agreement_score']
            if c in tweets.columns]
    tw = tweets[need].copy()
    tw['id']           = tw['id'].astype(str)
    ext_by_tweet['id'] = ext_by_tweet['id'].astype(str)
    df = tw.merge(ext_by_tweet, on='id', how='inner').dropna()
    df = df[df['impression_count'] > 0]

    # Follow notebook pattern exactly
    df['new_agreement_score'] = df['new_ext_score'] / (df['impression_count'] / 1000)
    avg_new = float(df['new_agreement_score'].mean())
    df['new_externality'] = df['new_agreement_score'] / np.abs(avg_new)
    df['new_e_score']     = df['new_externality'] * df['impressions_per_month'] / 1000

    # 95th-percentile axis limits
    orig = tweets[['e_score', 'v_score']].dropna() if 'e_score' in tweets.columns else df[['new_e_score', 'v_score']].rename(columns={'new_e_score': 'e_score'})
    e_lim_orig = float(np.percentile(orig['e_score'].abs(), 95)) * 1.05
    e_lim_new  = float(np.percentile(df['new_e_score'].abs(), 95)) * 1.05
    v_lim = float(np.percentile(
        np.concatenate([orig['v_score'].values, df['v_score'].values]), 95
    )) * 1.05

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    ax_l.scatter(orig['e_score'], orig['v_score'],
                 s=8, alpha=0.4, color='steelblue', rasterized=True)
    ax_l.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax_l.set_xlim(-e_lim_orig, e_lim_orig)
    ax_l.set_ylim(0, v_lim)
    ax_l.set_xlabel('e-score (externality per month)', fontsize=11)
    ax_l.set_ylabel('v-score (value per month)', fontsize=11)
    ax_l.set_title('Original Map  (NOT_HELPFUL=−1, SOMEWHAT_HELPFUL=0, HELPFUL=1)', fontsize=11)
    ax_l.grid(True, alpha=0.3)

    ax_r.scatter(df['new_e_score'], df['v_score'],
                 s=8, alpha=0.4, color='darkorange', rasterized=True)
    ax_r.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax_r.set_xlim(-e_lim_new, e_lim_new)
    ax_r.set_xlabel('new e-score (externality per month)', fontsize=11)
    ax_r.set_title('New Map  (NOT_HELPFUL=0, SOMEWHAT_HELPFUL=0.5, HELPFUL=1)', fontsize=11)
    ax_r.grid(True, alpha=0.3)

    avg_orig = float(tweets['agreement_score'].mean()) if 'agreement_score' in tweets.columns else float('nan')
    fig.suptitle(
        f'e-score vs v-score: Shifting NOT_HELPFUL from −1 to 0, SOMEWHAT_HELPFUL from 0 to 0.5\n'
        f'avg agreement score — original: {avg_orig:.4f},  new: {avg_new:.4f}',
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, figures_dir, 'escore_new_rating_mapping.png')


def fig_escore_per_month_raw(notes, ratings, tweets, figures_dir):
    """Side-by-side scatter: original e/v scores vs new per-month-raw scores.

    New e-score: Σ(note_direction × (Σ new_rating_scores + 1)) / months_active
    New v-score: action_count / months_active
    Rating map:  NOT_HELPFUL=0, SOMEWHAT_HELPFUL=0.5, HELPFUL=1
    """

    MIS = 'MISINFORMED_OR_POTENTIALLY_MISLEADING'
    NM  = 'NOT_MISLEADING'

    new_ratings_map = {
        'HELPFUL':          1.0,
        'SOMEWHAT_HELPFUL': 0.5,
        'NOT_HELPFUL':      0.0,
    }

    # Sum new rating scores per note, add 1 for original noter
    r = ratings[['noteId', 'helpfulnessLevel']].copy()
    r['rating_score'] = r['helpfulnessLevel'].map(new_ratings_map)
    r = r.dropna(subset=['rating_score'])
    note_sums = (r.groupby('noteId')['rating_score'].sum()
                 .reset_index()
                 .rename(columns={'rating_score': 'rating_sum'}))
    note_sums['rating_sum'] += 1.0

    # note externality = note_direction × rating_sum
    n = notes[['noteId', 'tweetId', 'classification']].copy()
    n['note_score'] = n['classification'].map({MIS: -1.0, NM: 1.0})
    n = n.dropna(subset=['note_score'])
    n = n.merge(note_sums, on='noteId', how='left')
    n['ext_score'] = n['note_score'] * n['rating_sum']

    ext_by_tweet = (n.groupby('tweetId')['ext_score'].sum()
                    .reset_index()
                    .rename(columns={'tweetId': 'id'}))

    need = [c for c in ['id', 'action_count', 'impression_count', 'impressions_per_month',
                        'months_old', 'e_score', 'v_score']
            if c in tweets.columns]
    tw = tweets[need].copy()
    tw['id']           = tw['id'].astype(str)
    ext_by_tweet['id'] = ext_by_tweet['id'].astype(str)
    df = tw.merge(ext_by_tweet, on='id', how='inner').dropna()

    # months_active: prefer months_old if present, else derive from impression columns
    if 'months_old' in df.columns:
        df['months_active'] = df['months_old']
    else:
        df['months_active'] = df['impression_count'] / df['impressions_per_month']
    df = df[df['months_active'] > 0]

    df['new_e_score'] = df['ext_score']    / df['months_active']
    df['new_v_score'] = df['action_count'] / df['months_active']

    # 95th-percentile axis limits (independent axes since v-score units differ)
    orig = tweets[['e_score', 'v_score']].dropna() if 'e_score' in tweets.columns else None
    e_lim_orig = float(np.percentile(orig['e_score'].abs(), 95)) * 1.05 if orig is not None else 1.0
    v_lim_orig = float(np.percentile(orig['v_score'],        95)) * 1.05 if orig is not None else 1.0
    e_lim_new  = float(np.percentile(df['new_e_score'].abs(), 95)) * 1.05
    v_lim_new  = float(np.percentile(df['new_v_score'],       95)) * 1.05

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))

    if orig is not None:
        ax_l.scatter(orig['e_score'], orig['v_score'],
                     s=8, alpha=0.4, color='steelblue', rasterized=True)
        ax_l.set_xlim(-e_lim_orig, e_lim_orig)
        ax_l.set_ylim(0, v_lim_orig)
    ax_l.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax_l.set_xlabel('e-score (externality per month)', fontsize=11)
    ax_l.set_ylabel('v-score (value per month)', fontsize=11)
    ax_l.set_title('Original e-score & v-score\n(complex reach-weighted normalization)', fontsize=11)
    ax_l.grid(True, alpha=0.3)

    ax_r.scatter(df['new_e_score'], df['new_v_score'],
                 s=8, alpha=0.4, color='purple', rasterized=True)
    ax_r.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax_r.set_xlim(-e_lim_new, e_lim_new)
    ax_r.set_ylim(0, v_lim_new)
    ax_r.set_xlabel('new e-score (note externality sum / months active)', fontsize=11)
    ax_r.set_ylabel('new v-score (actions / months active)', fontsize=11)
    ax_r.set_title('New e-score & v-score\n(NOT_HELPFUL=0, SOMEWHAT_HELPFUL=0.5, HELPFUL=1)', fontsize=11)
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(
        'e-score vs v-score: Per-Month Raw Scoring\n'
        'e = Σ note_direction × (Σ rating_score + 1) / months_active   |   v = actions / months_active',
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, figures_dir, 'escore_per_month_raw.png')


def fig_escore_misleading_only(notes, ratings, tweets, figures_dir):
    """Side-by-side e-score vs v-score: original vs new (0/0.5/1) map, using only misleading notes."""

    MIS = 'MISINFORMED_OR_POTENTIALLY_MISLEADING'

    new_ratings_map = {
        'HELPFUL':          1.0,
        'SOMEWHAT_HELPFUL': 0.5,
        'NOT_HELPFUL':      0.0,
    }

    r = ratings[['noteId', 'helpfulnessLevel']].copy()
    r['rating_score'] = r['helpfulnessLevel'].map(new_ratings_map)
    r = r.dropna(subset=['rating_score'])

    note_sums = (r.groupby('noteId')['rating_score'].sum()
                 .reset_index()
                 .rename(columns={'rating_score': 'rating_sum'}))
    note_sums['rating_sum'] += 1.0

    # Keep only misleading notes; note_score is always -1
    n = notes[notes['classification'] == MIS][['noteId', 'tweetId']].copy()
    n['note_score'] = -1.0
    n = n.merge(note_sums, on='noteId', how='left')
    n['new_ext_score'] = n['note_score'] * n['rating_sum']

    ext_by_tweet = (n.groupby('tweetId')['new_ext_score'].sum()
                    .reset_index()
                    .rename(columns={'tweetId': 'id'}))

    need = [c for c in ['id', 'impression_count', 'impressions_per_month', 'v_score']
            if c in tweets.columns]
    tw = tweets[need].copy()
    tw['id']           = tw['id'].astype(str)
    ext_by_tweet['id'] = ext_by_tweet['id'].astype(str)
    df = tw.merge(ext_by_tweet, on='id', how='inner').dropna()
    df = df[df['impression_count'] > 0]

    df['new_agreement_score'] = df['new_ext_score'] / (df['impression_count'] / 1000)
    avg_new = float(df['new_agreement_score'].mean())
    df['new_externality'] = df['new_agreement_score'] / np.abs(avg_new)
    df['new_e_score']     = df['new_externality'] * df['impressions_per_month'] / 1000

    orig = tweets[['e_score', 'v_score']].dropna() if 'e_score' in tweets.columns else df[['new_e_score', 'v_score']].rename(columns={'new_e_score': 'e_score'})
    e_lim_orig = float(np.percentile(orig['e_score'].abs(), 95)) * 1.05
    e_lim_new  = float(np.percentile(df['new_e_score'].abs(), 95)) * 1.05
    v_lim = float(np.percentile(
        np.concatenate([orig['v_score'].values, df['v_score'].values]), 95
    )) * 1.05

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    ax_l.scatter(orig['e_score'], orig['v_score'],
                 s=8, alpha=0.4, color='steelblue', rasterized=True)
    ax_l.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax_l.set_xlim(-e_lim_orig, e_lim_orig)
    ax_l.set_ylim(0, v_lim)
    ax_l.set_xlabel('e-score (externality per month)', fontsize=11)
    ax_l.set_ylabel('v-score (value per month)', fontsize=11)
    ax_l.set_title('Original Map  (all notes, NOT_HELPFUL=−1, SOMEWHAT_HELPFUL=0, HELPFUL=1)', fontsize=11)
    ax_l.grid(True, alpha=0.3)

    ax_r.scatter(df['new_e_score'], df['v_score'],
                 s=8, alpha=0.4, color='crimson', rasterized=True)
    ax_r.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax_r.set_xlim(-e_lim_new, e_lim_new)
    ax_r.set_xlabel('new e-score (externality per month)', fontsize=11)
    ax_r.set_title(f'New Map  (misleading notes only, n={len(df):,} posts,\nNOT_HELPFUL=0, SOMEWHAT_HELPFUL=0.5, HELPFUL=1)', fontsize=11)
    ax_r.grid(True, alpha=0.3)

    avg_orig = float(tweets['agreement_score'].mean()) if 'agreement_score' in tweets.columns else float('nan')
    fig.suptitle(
        f'e-score vs v-score: New Rating Map Applied to Misleading Notes Only\n'
        f'avg agreement score — original (all notes): {avg_orig:.4f},  new (misleading only): {avg_new:.4f}',
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, figures_dir, 'escore_misleading_only.png')

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


def fig_impressions_vs_actions_per_month(tweets, figures_dir):
    """Scatter: thousands of impressions/month (log x) vs actions per month (log y)."""
    cols = ['thousands_impressions_per_month', 'action_count_per_1000_impressions']
    df = tweets[cols].dropna()
    df = df.assign(actions_per_month=df['action_count_per_1000_impressions'] * df['thousands_impressions_per_month'])
    df = df[df['actions_per_month'] > 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['thousands_impressions_per_month'], df['actions_per_month'],
               s=10, alpha=0.4, color='steelblue', rasterized=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Thousands of Impressions per Month (log scale)', fontsize=11)
    ax.set_ylabel('Actions per Month (log scale)', fontsize=11)
    ax.set_title('Post Reach vs Monthly Actions', fontsize=12)
    ax.grid(True, alpha=0.3)

    _save(fig, figures_dir, 'impressions_vs_actions_per_month.png')


def fig_neg_impressions_vs_actions_per_month(tweets, figures_dir):
    """Scatter: negative thousands of impressions/month (linear x) vs actions per month (linear y)."""
    cols = ['thousands_impressions_per_month', 'action_count_per_1000_impressions']
    df = tweets[cols].dropna()
    df = df.assign(actions_per_month=df['action_count_per_1000_impressions'] * df['thousands_impressions_per_month'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(-df['thousands_impressions_per_month'], df['actions_per_month'],
               s=10, alpha=0.4, color='steelblue', rasterized=True)
    ax.set_xlabel('Negative Thousands of Impressions per Month', fontsize=11)
    ax.set_ylabel('Actions per Month', fontsize=11)
    ax.set_title('Post Reach vs Monthly Actions (Linear Scale)', fontsize=12)
    ax.grid(True, alpha=0.3)

    _save(fig, figures_dir, 'neg_impressions_vs_actions_per_month.png')


def fig_reach_scores_multiplot(tweets, figures_dir):
    """Two-panel scatter: left = reach×agreement vs actions/month; right = reach×agreement vs actions/month×v_score."""
    cols = ['thousands_impressions_per_month', 'action_count_per_1000_impressions',
            'agreement_score', 'v_score', 'e_score']
    df = tweets[[c for c in cols if c in tweets.columns]].dropna()
    df = df.assign(
        actions_per_month=df['action_count_per_1000_impressions'] * df['thousands_impressions_per_month'],
        x=df['thousands_impressions_per_month'] * df['agreement_score'],
    )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: reach×agreement vs actions_per_month
    ax_left.scatter(df['x'], df['actions_per_month'],
                    s=10, alpha=0.4, color='steelblue', rasterized=True)
    ax_left.set_xlabel('Thousands of Impressions per Month × Agreement Score', fontsize=11)
    ax_left.set_ylabel('Actions per Month', fontsize=11)
    ax_left.set_title('Weighted Reach vs Monthly Actions', fontsize=12)
    ax_left.grid(True, alpha=0.3)

    # Right: e_score vs v_score
    if 'e_score' in df.columns and 'v_score' in df.columns:
        ax_right.scatter(df['e_score'], df['v_score'],
                         s=10, alpha=0.4, color='darkorange', rasterized=True)
        ax_right.set_xlabel('Externality Score (e)', fontsize=11)
        ax_right.set_ylabel('Engagement Score (v)', fontsize=11)
        ax_right.set_title('Externality Score vs Engagement Score', fontsize=12)
        ax_right.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, figures_dir, 'reach_scores_multiplot.png')


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
    ax.set_xlim(-20, 5)

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
    fig_impressions_vs_actions_per_month(tweets, figures_dir)
    fig_neg_impressions_vs_actions_per_month(tweets, figures_dir)
    fig_reach_scores_multiplot(tweets, figures_dir)
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
    fig_multi_note_posts(notes_r, figures_dir)
    fig_escore_new_rating_mapping(notes, ratings, tweets, figures_dir)
    fig_escore_per_month_raw(notes, ratings, tweets, figures_dir)
    fig_escore_misleading_only(notes, ratings, tweets, figures_dir)

    print("\nAll outputs written to:", args.output_dir)


if __name__ == '__main__':
    main()
