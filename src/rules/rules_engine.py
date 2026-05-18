"""
Tier 1: Rules Engine for Fraud Detection
==========================================
Fast, rule-based filtering to eliminate obvious legitimate transactions.
Latency: <5ms, Cost: ~$0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RuleResult:
    """Result of rule evaluation."""
    is_flagged: bool
    rule_name: str
    reason: str
    confidence: float


class RulesEngine:
    """
    Rules Engine for fast fraud detection.
    
    This tier filters out ~90% of legitimate transactions using simple,
    deterministic rules before expensive ML models are applied.
    """
    
    def __init__(self):
        self.rules = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default fraud detection rules."""
        self.rules = [
            self._high_amount_rule,
            self._cross_border_rule,
            self._velocity_rule,
            self._missing_identity_rule,
            self._suspicious_device_rule
        ]
    
    def _high_amount_rule(self, row: pd.Series) -> RuleResult:
        """
        Rule: Flag transactions with unusually high amounts.
        
        Threshold: > $1000 (adjustable based on business needs)
        """
        threshold = 1000.0
        amount = row.get('TransactionAmt', 0)
        
        if amount > threshold:
            return RuleResult(
                is_flagged=True,
                rule_name='high_amount',
                reason=f'Transaction amount ${amount:.2f} exceeds threshold ${threshold}',
                confidence=0.3
            )
        
        return RuleResult(
            is_flagged=False,
            rule_name='high_amount',
            reason='Amount within normal range',
            confidence=0.0
        )
    
    def _cross_border_rule(self, row: pd.Series) -> RuleResult:
        """
        Rule: Flag cross-border transactions with high amounts.
        
        Logic: If card issuing country differs from transaction country
        and amount > $500, flag as suspicious.
        """
        amount_threshold = 500.0
        
        card_country = row.get('card6', None)  # card type/region
        addr_country = row.get('addr1', None)  # billing address
        
        # This is a simplified rule - in production, use actual country codes
        if card_country and addr_country and card_country != addr_country:
            amount = row.get('TransactionAmt', 0)
            if amount > amount_threshold:
                return RuleResult(
                    is_flagged=True,
                    rule_name='cross_border',
                    reason=f'Cross-border transaction with amount ${amount:.2f}',
                    confidence=0.5
                )
        
        return RuleResult(
            is_flagged=False,
            rule_name='cross_border',
            reason='Domestic transaction or low amount',
            confidence=0.0
        )
    
    def _velocity_rule(self, row: pd.Series) -> RuleResult:
        """
        Rule: Flag high-velocity transactions.
        
        Logic: If a card has too many transactions in a short time window,
        flag as potential fraud.
        
        Note: This requires pre-computed velocity features.
        """
        # Check for velocity features (if they exist)
        velocity_col = 'card1_TransactionAmt_count_1h'
        
        if velocity_col in row.index:
            count = row.get(velocity_col, 0)
            if count > 10:  # More than 10 transactions in 1 hour
                return RuleResult(
                    is_flagged=True,
                    rule_name='high_velocity',
                    reason=f'High transaction velocity: {count} transactions in 1 hour',
                    confidence=0.6
                )
        
        return RuleResult(
            is_flagged=False,
            rule_name='high_velocity',
            reason='Normal transaction velocity',
            confidence=0.0
        )
    
    def _missing_identity_rule(self, row: pd.Series) -> RuleResult:
        """
        Rule: Flag transactions with missing identity information.
        
        Logic: Missing device info or email domain is suspicious.
        """
        device_info = row.get('DeviceInfo', None)
        email_domain = row.get('P_emaildomain', None)
        
        missing_count = sum([pd.isna(device_info), pd.isna(email_domain)])
        
        if missing_count >= 2:
            return RuleResult(
                is_flagged=True,
                rule_name='missing_identity',
                reason=f'Missing {missing_count} identity fields',
                confidence=0.4
            )
        
        return RuleResult(
            is_flagged=False,
            rule_name='missing_identity',
            reason='Identity information present',
            confidence=0.0
        )
    
    def _suspicious_device_rule(self, row: pd.Series) -> RuleResult:
        """
        Rule: Flag transactions from suspicious devices.
        
        Logic: Known fraudulent device patterns.
        """
        device_info = row.get('DeviceInfo', None)
        
        if pd.isna(device_info):
            return RuleResult(
                is_flagged=False,
                rule_name='suspicious_device',
                reason='No device info',
                confidence=0.0
            )
        
        # Known suspicious device patterns (simplified)
        suspicious_patterns = ['unknown', 'root', 'debug', 'test']
        
        device_str = str(device_info).lower()
        if any(pattern in device_str for pattern in suspicious_patterns):
            return RuleResult(
                is_flagged=True,
                rule_name='suspicious_device',
                reason=f'Suspicious device pattern: {device_info}',
                confidence=0.5
            )
        
        return RuleResult(
            is_flagged=False,
            rule_name='suspicious_device',
            reason='Device appears normal',
            confidence=0.0
        )
    
    def evaluate_transaction(self, row: pd.Series) -> Tuple[bool, List[RuleResult]]:
        """
        Evaluate a single transaction against all rules.
        
        Args:
            row: Transaction data as pandas Series
        
        Returns:
            Tuple of (is_flagged, list of rule results)
        """
        results = []
        
        for rule in self.rules:
            try:
                result = rule(row)
                results.append(result)
            except Exception as e:
                # If a rule fails, log it and continue
                print(f"Rule failed: {e}")
                continue
        
        # Transaction is flagged if ANY rule flags it
        is_flagged = any(r.is_flagged for r in results)
        
        return is_flagged, results
    
    def evaluate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate a batch of transactions.
        
        Args:
            data: DataFrame of transactions
        
        Returns:
            DataFrame with rule evaluation results
        """
        results = []
        
        for idx, row in data.iterrows():
            is_flagged, rule_results = self.evaluate_transaction(row)
            
            # Get the highest confidence rule that flagged
            flagged_rules = [r for r in rule_results if r.is_flagged]
            max_confidence = max([r.confidence for r in flagged_rules]) if flagged_rules else 0.0
            primary_reason = flagged_rules[0].reason if flagged_rules else "No flags"
            
            results.append({
                'TransactionID': row.get('TransactionID', idx),
                'tier1_flagged': is_flagged,
                'tier1_confidence': max_confidence,
                'tier1_reason': primary_rule
            })
        
        return pd.DataFrame(results)
    
    def add_custom_rule(self, rule_func):
        """
        Add a custom rule to the engine.
        
        Args:
            rule_func: Function that takes a pandas Series and returns RuleResult
        """
        self.rules.append(rule_func)
    
    def get_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Get statistics about rule performance.
        
        Args:
            data: DataFrame with rule evaluation results
        
        Returns:
            Dictionary of statistics
        """
        total = len(data)
        flagged = data['tier1_flagged'].sum()
        
        return {
            'total_transactions': total,
            'flagged_transactions': flagged,
            'flag_rate': flagged / total if total > 0 else 0,
            'pass_rate': 1 - (flagged / total) if total > 0 else 0
        }


def create_rules_engine() -> RulesEngine:
    """
    Factory function to create a rules engine instance.
    
    Returns:
        Configured RulesEngine instance
    """
    return RulesEngine()
