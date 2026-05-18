"""
Model comparison and evaluation table for tiered architecture.
"""

import pandas as pd
from typing import Dict, List


def create_model_comparison_table() -> pd.DataFrame:
    """
    Create a comprehensive model comparison table for tiered architecture.
    
    Returns:
        DataFrame with model comparison metrics
    """
    comparison_data = {
        'Criteria': [
            'Accuracy (PR-AUC)',
            'Response Latency',
            'Hardware Requirements',
            'Deployment & Operations Cost',
            'Explainability'
        ],
        'Rule-based (Tier 1)': [
            'Low (Easily bypassed)',
            '< 5ms (Extremely Fast)',
            'CPU only (Minimal)',
            'Near $0',
            '100% (Fully explainable)'
        ],
        'Tree-based (Tier 2 - LightGBM/XGBoost)': [
            'High to Very High',
            '< 50ms (Fast)',
            'CPU only (Standard)',
            'Low to Medium',
            'Good (SHAP / Feature Importance)'
        ],
        'Deep Learning / GNN (Tier 3)': [
            'Very High (Best for complex cases)',
            '> 200ms (Slow)',
            'GPU required (High)',
            'Very High',
            'Black-box (Difficult to explain)'
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("MODEL EVALUATION COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


def get_tier_recommendations(hardware_constraints: Dict = None) -> Dict:
    """
    Get tier recommendations based on hardware constraints.
    
    Args:
        hardware_constraints: Dictionary with hardware specs (ram_gb, has_gpu, etc.)
    
    Returns:
        Dictionary with tier recommendations
    """
    if hardware_constraints is None:
        hardware_constraints = {
            'ram_gb': 16,
            'has_gpu': False,
            'cpu_cores': 8
        }
    
    recommendations = {
        'tier_1': {
            'recommended': True,
            'reason': 'Always recommended - zero cost, filters 90% of transactions',
            'latency': '< 5ms',
            'cost': '$0'
        },
        'tier_2': {
            'recommended': True,
            'reason': 'Recommended for CPU-only setups - handles remaining 10% efficiently',
            'latency': '< 50ms',
            'cost': 'Low'
        }
    }
    
    if hardware_constraints.get('has_gpu', False) and hardware_constraints.get('ram_gb', 0) >= 32:
        recommendations['tier_3'] = {
            'recommended': True,
            'reason': 'Optional - for complex fraud rings and batch analysis',
            'latency': '> 200ms',
            'cost': 'High'
        }
    else:
        recommendations['tier_3'] = {
            'recommended': False,
            'reason': f'Not recommended - requires GPU and 32GB+ RAM (current: {hardware_constraints["ram_gb"]}GB RAM, GPU: {hardware_constraints["has_gpu"]})',
            'latency': 'N/A',
            'cost': 'N/A'
        }
    
    print("\n" + "="*80)
    print("TIER RECOMMENDATIONS BASED ON HARDWARE")
    print("="*80)
    print(f"Hardware: {hardware_constraints['ram_gb']}GB RAM, GPU: {'Yes' if hardware_constraints['has_gpu'] else 'No'}")
    print()
    
    for tier, info in recommendations.items():
        status = "RECOMMENDED" if info['recommended'] else "NOT RECOMMENDED"
        print(f"{tier.upper()}: {status}")
        print(f"  Reason: {info['reason']}")
        print(f"  Latency: {info['latency']}")
        print(f"  Cost: {info['cost']}")
        print()
    
    print("="*80)
    
    return recommendations


def calculate_cost_savings(
    total_transactions: int = 1000000,
    tier_1_filter_rate: float = 0.90,
    tier_2_filter_rate: float = 0.90,
    tier_3_filter_rate: float = 0.50
) -> Dict:
    """
    Calculate cost savings from tiered architecture.
    
    Args:
        total_transactions: Total number of transactions
        tier_1_filter_rate: Percentage filtered by Tier 1
        tier_2_filter_rate: Percentage of remaining filtered by Tier 2
        tier_3_filter_rate: Percentage of remaining filtered by Tier 3
    
    Returns:
        Dictionary with cost analysis
    """
    # Calculate transactions at each tier
    tier_1_processed = total_transactions
    tier_1_filtered = tier_1_processed * tier_1_filter_rate
    
    tier_2_processed = tier_1_processed - tier_1_filtered
    tier_2_filtered = tier_2_processed * tier_2_filter_rate
    
    tier_3_processed = tier_2_processed - tier_2_filtered
    tier_3_filtered = tier_3_processed * tier_3_filter_rate
    
    # Cost assumptions (relative units)
    cost_tier_1 = 0.001  # $0.001 per transaction
    cost_tier_2 = 0.01   # $0.01 per transaction
    cost_tier_3 = 0.1    # $0.1 per transaction
    
    # Calculate costs
    cost_with_tiers = (
        tier_1_processed * cost_tier_1 +
        tier_2_processed * cost_tier_2 +
        tier_3_processed * cost_tier_3
    )
    
    cost_without_tiers = total_transactions * cost_tier_2  # Assume all go through Tier 2
    
    savings = cost_without_tiers - cost_with_tiers
    savings_percentage = (savings / cost_without_tiers) * 100
    
    analysis = {
        'total_transactions': total_transactions,
        'tier_1_processed': tier_1_processed,
        'tier_1_filtered': tier_1_filtered,
        'tier_2_processed': tier_2_processed,
        'tier_2_filtered': tier_2_filtered,
        'tier_3_processed': tier_3_processed,
        'tier_3_filtered': tier_3_filtered,
        'cost_with_tiers': cost_with_tiers,
        'cost_without_tiers': cost_without_tiers,
        'savings': savings,
        'savings_percentage': savings_percentage
    }
    
    print("\n" + "="*80)
    print("COST SAVINGS ANALYSIS")
    print("="*80)
    print(f"Total Transactions: {total_transactions:,}")
    print()
    print(f"Tier 1 (Rules):")
    print(f"  Processed: {tier_1_processed:,}")
    print(f"  Filtered: {tier_1_filtered:,} ({tier_1_filter_rate:.1%})")
    print(f"  Passed to Tier 2: {tier_2_processed:,}")
    print()
    print(f"Tier 2 (LightGBM):")
    print(f"  Processed: {tier_2_processed:,}")
    print(f"  Filtered: {tier_2_filtered:,} ({tier_2_filter_rate:.1%})")
    print(f"  Passed to Tier 3: {tier_3_processed:,}")
    print()
    print(f"Tier 3 (Deep Learning):")
    print(f"  Processed: {tier_3_processed:,}")
    print(f"  Filtered: {tier_3_filtered:,} ({tier_3_filter_rate:.1%})")
    print()
    print(f"Cost Analysis:")
    print(f"  Cost with Tiers: ${cost_with_tiers:,.2f}")
    print(f"  Cost without Tiers (all Tier 2): ${cost_without_tiers:,.2f}")
    print(f"  Savings: ${savings:,.2f} ({savings_percentage:.1f}%)")
    print("="*80)
    
    return analysis
