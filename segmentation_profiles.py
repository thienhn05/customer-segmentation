import numpy as np
import pandas as pd


def assign_data_driven_segment_name(cluster_row, population_medians, n_clusters):
    """
    Assign a meaningful, non-duplicated business segment name based on
    actual cluster statistics relative to the overall customer population.
    """
    rec = cluster_row['Recency_median']
    freq = cluster_row['Frequency_median']
    mon = cluster_row['Monetary_median']
    aov = cluster_row['AvgOrderValue_median']

    med_rec = population_medians['Recency']
    med_freq = population_medians['Frequency']
    med_mon = population_medians['Monetary']
    med_aov = population_medians['AvgOrderValue']

    if n_clusters == 2:
        # Binary segmentation: High Engagement vs Low Engagement
        if mon >= med_mon or (rec <= med_rec and freq >= med_freq):
            return 'High-Value Active Customers'
        else:
            return 'Low-Engagement / Lapsed Spenders'

    elif n_clusters == 3:
        if mon >= med_mon * 2 and freq >= med_freq * 2:
            return 'Champions & High-Value Loyal'
        elif rec <= med_rec and mon >= med_mon * 0.8:
            return 'Active Regular Spenders'
        else:
            return 'At-Risk / Lapsed Inactive Customers'

    elif n_clusters == 4:
        if mon >= med_mon * 2 and freq >= med_freq * 2 and rec <= med_rec:
            return 'High-Value Loyal Champions'
        elif rec <= med_rec * 0.8 and freq <= med_freq:
            return 'Promising Recent Buyers'
        elif rec >= med_rec * 1.5 and mon <= med_mon:
            return 'Hibernating / Low-Value Inactive'
        elif mon >= med_mon or aov >= med_aov:
            return 'At-Risk / Moderate Spenders'
        else:
            return 'Occasional Regular Buyers'

    else:
        # General rule for arbitrary K
        if rec <= med_rec and freq >= med_freq and mon >= med_mon:
            return 'High-Value Loyal Customers'
        elif rec >= med_rec and freq <= med_freq and mon <= med_mon:
            return 'At-Risk Inactive Customers'
        elif rec <= med_rec and mon >= med_mon:
            return 'Recent High-Spenders'
        elif rec <= med_rec:
            return 'Recent Occasional Buyers'
        elif aov >= med_aov:
            return 'High-Ticket Bulk Buyers'
        else:
            return 'Moderate Engagement Spenders'


def create_final_segment_profiles(customer_features, labels):
    """
    Build a comprehensive segment profile DataFrame with:
    - Cluster ID
    - Meaningful Segment Name
    - Customer Count & Percentage
    - Mean and Median Recency, Frequency, Monetary, AvgOrderValue
    - Characteristics
    - Actionable Business Strategy
    """
    df = customer_features.copy()
    df['Cluster'] = labels
    n_clusters = len(np.unique(labels))
    total_customers = len(df)

    pop_medians = {
        'Recency': float(df['Recency'].median()),
        'Frequency': float(df['Frequency'].median()),
        'Monetary': float(df['Monetary'].median()),
        'AvgOrderValue': float(df['AvgOrderValue'].median()),
    }

    # Aggregate by cluster
    agg_dict = {
        'CustomerID': 'count',
        'Recency': ['mean', 'median'],
        'Frequency': ['mean', 'median'],
        'Monetary': ['mean', 'median'],
        'AvgOrderValue': ['mean', 'median'],
    }
    grouped = df.groupby('Cluster').agg(agg_dict)

    # Flatten multi-level column names
    flat_cols = ['Customers', 'Recency_mean', 'Recency_median',
                 'Frequency_mean', 'Frequency_median',
                 'Monetary_mean', 'Monetary_median',
                 'AvgOrderValue_mean', 'AvgOrderValue_median']
    grouped.columns = flat_cols
    grouped = grouped.reset_index()

    profiles = []
    used_names = set()

    for _, row in grouped.iterrows():
        cluster_id = int(row['Cluster'])
        count = int(row['Customers'])
        pct = (count / total_customers) * 100

        seg_name = assign_data_driven_segment_name(row, pop_medians, n_clusters)
        # Ensure distinct names if fallback
        if seg_name in used_names:
            seg_name = f"{seg_name} (Tier {cluster_id + 1})"
        used_names.add(seg_name)

        # Describe characteristics
        char_parts = []
        if row['Recency_median'] <= pop_medians['Recency']:
            char_parts.append(f"Highly recent purchases (median {row['Recency_median']:.0f} days)")
        else:
            char_parts.append(f"Long absence / inactive (median {row['Recency_median']:.0f} days)")

        if row['Frequency_median'] >= pop_medians['Frequency']:
            char_parts.append(f"frequent orders (median {row['Frequency_median']:.0f} orders)")
        else:
            char_parts.append(f"infrequent orders (median {row['Frequency_median']:.0f} order)")

        if row['Monetary_median'] >= pop_medians['Monetary']:
            char_parts.append(f"above-average spend (median £{row['Monetary_median']:,.2f})")
        else:
            char_parts.append(f"low monetary volume (median £{row['Monetary_median']:,.2f})")

        characteristics = "; ".join(char_parts)

        # Business recommendation
        if 'High-Value' in seg_name or 'Champion' in seg_name:
            strategy = (
                "VIP loyalty programs, early access to new product releases, "
                "dedicated account management, and premium bundle cross-selling to maximize lifetime value."
            )
        elif 'At-Risk' in seg_name or 'Lapsed' in seg_name or 'Hibernating' in seg_name or 'Low-Engagement' in seg_name:
            strategy = (
                "Automated win-back email workflows, time-sensitive re-engagement discounts, "
                "surveys to identify customer churn drivers, and low-friction catalog recommendations."
            )
        elif 'Promising' in seg_name or 'Recent' in seg_name:
            strategy = (
                "Onboarding welcome sequences, product recommendation engines based on first purchase, "
                "and loyalty points incentives on second orders to build habitual purchasing."
            )
        else:
            strategy = (
                "Value-focused cross-selling campaigns, minimum spend free-shipping thresholds, "
                "and category expansion promotions to increase order frequency."
            )

        profiles.append({
            'ClusterID': cluster_id,
            'SegmentName': seg_name,
            'CustomerCount': count,
            'Percentage': round(pct, 2),
            'Recency_Mean': round(row['Recency_mean'], 2),
            'Recency_Median': round(row['Recency_median'], 2),
            'Frequency_Mean': round(row['Frequency_mean'], 2),
            'Frequency_Median': round(row['Frequency_median'], 2),
            'Monetary_Mean': round(row['Monetary_mean'], 2),
            'Monetary_Median': round(row['Monetary_median'], 2),
            'AvgOrderValue_Mean': round(row['AvgOrderValue_mean'], 2),
            'AvgOrderValue_Median': round(row['AvgOrderValue_median'], 2),
            'Characteristics': characteristics,
            'RecommendedAction': strategy,
        })

    return pd.DataFrame(profiles).sort_values('ClusterID').reset_index(drop=True)
