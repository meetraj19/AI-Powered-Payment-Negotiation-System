import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import asyncio
from dataclasses import dataclass
from enum import Enum

class FinancialSituation(Enum):
    CRITICAL = "critical"
    STRESSED = "stressed"
    MODERATE = "moderate"
    STABLE = "stable"

@dataclass
class FinancialProfile:
    monthly_income: float
    monthly_obligations: float
    liquid_assets: float
    credit_score: int
    debt_to_income: float
    employment_status: str
    hardship_factors: List[str]

class DebtorAgent:
    def __init__(self, agent_id: str, debt_amount: float, financial_profile: FinancialProfile):
        self.agent_id = agent_id
        self.debt_amount = debt_amount
        self.financial_profile = financial_profile
        self.situation = self._assess_financial_situation()
        
        # Negotiation parameters
        self.max_affordable = self._calculate_max_affordable()
        self.target_settlement = debt_amount * 0.4  # Target 40% settlement
        self.negotiation_history = []
        self.stress_level = 0.5  # 0-1 scale
        self.urgency_factor = 0.3  # Need for quick resolution
        
        # Learning parameters
        self.q_table = {}
        self.epsilon = 0.15
        self.learning_rate = 0.12
        self.gamma = 0.9
        
        # Strategy parameters
        self.concession_pattern = "graduated"  # graduated, linear, aggressive
        self.emotional_appeal_used = False
        self.hardship_documented = False
        
    def _assess_financial_situation(self) -> FinancialSituation:
        """Assess current financial situation"""
        disposable_income = self.financial_profile.monthly_income - self.financial_profile.monthly_obligations
        income_ratio = disposable_income / self.financial_profile.monthly_income if self.financial_profile.monthly_income > 0 else 0
        
        if income_ratio < 0.1 or self.financial_profile.credit_score < 500:
            return FinancialSituation.CRITICAL
        elif income_ratio < 0.2 or self.financial_profile.credit_score < 600:
            return FinancialSituation.STRESSED
        elif income_ratio < 0.3 or self.financial_profile.credit_score < 700:
            return FinancialSituation.MODERATE
        else:
            return FinancialSituation.STABLE
    
    def _calculate_max_affordable(self) -> float:
        """Calculate maximum affordable payment"""
        disposable_income = self.financial_profile.monthly_income - self.financial_profile.monthly_obligations
        
        # Factor in liquid assets
        available_lump_sum = self.financial_profile.liquid_assets * 0.7  # Keep 30% emergency fund
        
        # Monthly payment capacity (36 months max)
        monthly_capacity = disposable_income * 0.5  # Use 50% of disposable income
        installment_capacity = monthly_capacity * 36
        
        return min(available_lump_sum + installment_capacity, self.debt_amount)
    
    async def get_profile(self) -> Dict:
        """Return debtor profile for creditor analysis"""
        return {
            "credit_score": self.financial_profile.credit_score,
            "monthly_income": self.financial_profile.monthly_income,
            "monthly_obligations": self.financial_profile.monthly_obligations,
            "debt_to_income": self.financial_profile.debt_to_income,
            "employment_status": self.financial_profile.employment_status,
            "default_history": self._get_default_history(),
            "financial_situation": self.situation.value
        }
    
    def _get_default_history(self) -> Dict:
        """Generate default history based on credit score"""
        if self.financial_profile.credit_score >= 700:
            return {"rate": 0.02, "recent_defaults": 0}
        elif self.financial_profile.credit_score >= 600:
            return {"rate": 0.05, "recent_defaults": 1}
        elif self.financial_profile.credit_score >= 500:
            return {"rate": 0.15, "recent_defaults": 2}
        else:
            return {"rate": 0.30, "recent_defaults": 3}
    
    async def receive_offer(self, offer: 'Offer') -> Optional['Offer']:
        """Receive and evaluate creditor's offer"""
        self.negotiation_history.append({
            "type": "offer_received",
            "offer": offer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Evaluate offer
        evaluation = self._evaluate_offer(offer)
        
        # Decision making with Q-learning
        state = self._get_state(offer)
        action = self._select_action(state, evaluation)
        
        if action == "accept":
            self._update_stress_level(-0.3)  # Relief from resolution
            return None  # None indicates acceptance
            
        elif action == "counter":
            counter_offer = self._generate_counter_offer(offer, evaluation)
            self._update_stress_level(0.1)  # Slight stress increase
            return counter_offer
            
        else:  # reject
            self._update_stress_level(0.2)  # Stress from prolonged negotiation
            return self._generate_aggressive_counter(offer)
    
    def _evaluate_offer(self, offer: 'Offer') -> Dict:
        """Comprehensive offer evaluation"""
        # Financial feasibility
        total_payment = offer.amount * (1 + offer.interest_rate * offer.timeline_days / 365)
        affordability_score = 1 - (total_payment / self.max_affordable) if self.max_affordable > 0 else 0
        
        # Timeline evaluation
        timeline_score = self._evaluate_timeline(offer.timeline_days)
        
        # Payment structure evaluation
        structure_score = self._evaluate_payment_structure(offer.payment_terms)
        
        # Overall score
        weights = {"affordability": 0.5, "timeline": 0.3, "structure": 0.2}
        overall_score = (
            affordability_score * weights["affordability"] +
            timeline_score * weights["timeline"] +
            structure_score * weights["structure"]
        )
        
        return {
            "overall_score": overall_score,
            "affordability_score": affordability_score,
            "timeline_score": timeline_score,
            "structure_score": structure_score,
            "acceptable": overall_score > 0.6,
            "negotiable": 0.3 < overall_score <= 0.6,
            "unacceptable": overall_score <= 0.3
        }
    
    def _evaluate_timeline(self, days: int) -> float:
        """Evaluate payment timeline"""
        if self.situation == FinancialSituation.CRITICAL:
            # Prefer longer timelines
            return min(days / 365, 1.0)
        elif self.situation == FinancialSituation.STABLE:
            # Prefer shorter timelines to resolve quickly
            return 1.0 - min(days / 180, 1.0)
        else:
            # Moderate preference for 90-180 days
            if 90 <= days <= 180:
                return 1.0
            elif days < 90:
                return days / 90
            else:
                return 180 / days
    
    def _evaluate_payment_structure(self, payment_terms: Dict) -> float:
        """Evaluate payment structure suitability"""
        if payment_terms["type"] == "lump_sum":
            # Check if lump sum is affordable
            if payment_terms["amounts"][0] <= self.financial_profile.liquid_assets * 0.7:
                return 0.9
            else:
                return 0.3
        else:  # installment
            monthly_payment = payment_terms["amounts"][0]
            disposable_income = self.financial_profile.monthly_income - self.financial_profile.monthly_obligations
            
            if monthly_payment <= disposable_income * 0.3:
                return 1.0
            elif monthly_payment <= disposable_income * 0.5:
                return 0.7
            else:
                return 0.2
    
    def _get_state(self, offer: 'Offer') -> str:
        """Convert situation to state for Q-learning"""
        settlement_ratio = int((offer.amount / self.debt_amount) * 10)
        timeline_bucket = min(offer.timeline_days // 30, 12)
        stress_bucket = int(self.stress_level * 5)
        round_number = len(self.negotiation_history)
        
        return f"{settlement_ratio}_{timeline_bucket}_{stress_bucket}_{round_number}"
    
    def _select_action(self, state: str, evaluation: Dict) -> str:
        """Select action using epsilon-greedy Q-learning"""
        if np.random.random() < self.epsilon:
            # Exploration
            if evaluation["acceptable"]:
                return np.random.choice(["accept", "counter"], p=[0.7, 0.3])
            elif evaluation["negotiable"]:
                return np.random.choice(["accept", "counter", "reject"], p=[0.2, 0.6, 0.2])
            else:
                return np.random.choice(["counter", "reject"], p=[0.7, 0.3])
        else:
            # Exploitation
            if state in self.q_table:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                # Default strategy based on evaluation
                if evaluation["acceptable"]:
                    return "accept"
                elif evaluation["negotiable"]:
                    return "counter"
                else:
                    return "reject"
    
    def _generate_counter_offer(self, received_offer: 'Offer', evaluation: Dict) -> 'Offer':
        """Generate strategic counter-offer"""
        rounds = len([h for h in self.negotiation_history if h["type"] == "counter_offer_sent"])
        
        # Base counter calculation
        if evaluation["affordability_score"] < 0.5:
            # Can't afford current offer
            target_amount = self.max_affordable * 0.8
        else:
            # Strategic negotiation
            if self.concession_pattern == "graduated":
                concession = 0.1 * (1 + rounds * 0.05)
            elif self.concession_pattern == "linear":
                concession = 0.05 * rounds
            else:  # aggressive
                concession = 0.02 * rounds
            
            current_gap = received_offer.amount - self.target_settlement
            target_amount = self.target_settlement + (current_gap * concession)
        
        # Adjust timeline based on situation
        if self.situation == FinancialSituation.CRITICAL:
            preferred_timeline = min(received_offer.timeline_days + 60, 365)
        else:
            preferred_timeline = received_offer.timeline_days
        
        # Include hardship documentation if applicable
        hardship_docs = None
        if not self.hardship_documented and len(self.financial_profile.hardship_factors) > 0:
            hardship_docs = {
                "factors": self.financial_profile.hardship_factors,
                "documentation": "available upon request"
            }
            self.hardship_documented = True
        
        counter_offer = Offer(
            amount=target_amount,
            payment_terms=self._propose_payment_structure(target_amount, preferred_timeline),
            interest_rate=min(received_offer.interest_rate * 0.5, 0.03),
            timeline_days=preferred_timeline,
            collateral=hardship_docs
        )
        
        self.negotiation_history.append({
            "type": "counter_offer_sent",
            "offer": counter_offer,
            "timestamp": datetime.now().isoformat()
        })
        
        return counter_offer
    
    def _generate_aggressive_counter(self, received_offer: 'Offer') -> 'Offer':
        """Generate aggressive counter for unacceptable offers"""
        # Start from target settlement
        counter_offer = Offer(
            amount=self.target_settlement,
            payment_terms=self._propose_payment_structure(self.target_settlement, 365),
            interest_rate=0.0,
            timeline_days=365,
            collateral={"hardship_claim": True, "bankruptcy_consideration": True}
        )
        
        self.negotiation_history.append({
            "type": "aggressive_counter",
            "offer": counter_offer,
            "timestamp": datetime.now().isoformat()
        })
        
        return counter_offer
    
    def _propose_payment_structure(self, amount: float, timeline_days: int) -> Dict:
        """Propose payment structure based on financial situation"""
        if self.financial_profile.liquid_assets >= amount * 0.5:
            # Can make substantial down payment
            down_payment = amount * 0.3
            remaining = amount - down_payment
            monthly_payments = remaining / (timeline_days / 30)
            
            return {
                "type": "hybrid",
                "down_payment": down_payment,
                "installments": int(timeline_days / 30),
                "amounts": [monthly_payments] * int(timeline_days / 30),
                "due_dates": [(i + 1) * 30 for i in range(int(timeline_days / 30))]
            }
        else:
            # Pure installment plan
            num_installments = int(timeline_days / 30)
            installment_amount = amount / num_installments
            
            return {
                "type": "installment",
                "installments": num_installments,
                "amounts": [installment_amount] * num_installments,
                "due_dates": [(i + 1) * 30 for i in range(num_installments)]
            }
    
    def _update_stress_level(self, delta: float):
        """Update debtor's stress level"""
        self.stress_level = max(0.0, min(1.0, self.stress_level + delta))
        
        # High stress affects negotiation strategy
        if self.stress_level > 0.8:
            self.concession_pattern = "aggressive"
            self.urgency_factor = min(0.8, self.urgency_factor + 0.1)
        elif self.stress_level < 0.3:
            self.concession_pattern = "graduated"
            self.urgency_factor = max(0.2, self.urgency_factor - 0.1)
    
    def get_negotiation_summary(self) -> Dict:
        """Get summary of negotiation progress"""
        offers_received = [h for h in self.negotiation_history if h["type"] == "offer_received"]
        counters_sent = [h for h in self.negotiation_history if "counter" in h["type"]]
        
        return {
            "agent_id": self.agent_id,
            "rounds_negotiated": len(offers_received),
            "counters_made": len(counters_sent),
            "current_stress_level": self.stress_level,
            "financial_situation": self.situation.value,
            "max_affordable": self.max_affordable,
            "target_settlement": self.target_settlement,
            "hardship_documented": self.hardship_documented
        }