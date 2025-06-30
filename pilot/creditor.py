import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import asyncio
from dataclasses import dataclass
from enum import Enum

class NegotiationStrategy(Enum):
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    FLEXIBLE = "flexible"
    ADAPTIVE = "adaptive"

@dataclass
class Offer:
    amount: float
    payment_terms: Dict
    interest_rate: float
    timeline_days: int
    collateral: Optional[Dict] = None
    penalties: Optional[Dict] = None

class CreditorAgent:
    def __init__(self, agent_id: str, initial_claim: float):
        self.agent_id = agent_id
        self.initial_claim = initial_claim
        self.current_offer = None
        self.negotiation_history = []
        self.strategy = NegotiationStrategy.ADAPTIVE
        self.min_acceptable = initial_claim * 0.6  # 60% recovery minimum
        self.patience_level = 10  # rounds before escalation
        self.risk_tolerance = 0.3
        self.learning_rate = 0.1
        
        # Q-learning parameters
        self.q_table = {}
        self.epsilon = 0.1  # exploration rate
        self.gamma = 0.95  # discount factor
        
        # Performance metrics
        self.successful_negotiations = 0
        self.total_negotiations = 0
        self.average_recovery_rate = 0.0
        
    def analyze_debtor_profile(self, debtor_info: Dict) -> Dict:
        """Analyze debtor's financial profile and payment history"""
        risk_score = self._calculate_risk_score(debtor_info)
        payment_capacity = self._estimate_payment_capacity(debtor_info)
        
        return {
            "risk_score": risk_score,
            "payment_capacity": payment_capacity,
            "recommended_strategy": self._select_strategy(risk_score, payment_capacity),
            "initial_offer_multiplier": 1.0 - (risk_score * 0.3)
        }
    
    def _calculate_risk_score(self, debtor_info: Dict) -> float:
        """Calculate risk score based on debtor information"""
        base_score = 0.5
        
        # Credit history impact
        if "credit_score" in debtor_info:
            credit_factor = (debtor_info["credit_score"] - 300) / 550
            base_score -= credit_factor * 0.3
        
        # Payment history impact
        if "default_history" in debtor_info:
            default_rate = debtor_info["default_history"].get("rate", 0)
            base_score += default_rate * 0.4
        
        # Current financial situation
        if "debt_to_income" in debtor_info:
            dti = debtor_info["debt_to_income"]
            base_score += min(dti * 0.3, 0.3)
        
        return max(0.0, min(1.0, base_score))
    
    def _estimate_payment_capacity(self, debtor_info: Dict) -> float:
        """Estimate debtor's capacity to pay"""
        monthly_income = debtor_info.get("monthly_income", 0)
        existing_obligations = debtor_info.get("monthly_obligations", 0)
        
        available_income = monthly_income - existing_obligations
        capacity = available_income * 36  # 3-year payment capacity
        
        return min(capacity / self.initial_claim, 1.0)
    
    def _select_strategy(self, risk_score: float, payment_capacity: float) -> NegotiationStrategy:
        """Select negotiation strategy based on analysis"""
        if risk_score > 0.7:
            return NegotiationStrategy.AGGRESSIVE
        elif payment_capacity > 0.8:
            return NegotiationStrategy.MODERATE
        elif risk_score < 0.3 and payment_capacity > 0.5:
            return NegotiationStrategy.FLEXIBLE
        else:
            return NegotiationStrategy.ADAPTIVE
    
    def generate_initial_offer(self, debtor_profile: Dict) -> Offer:
        """Generate initial settlement offer"""
        analysis = self.analyze_debtor_profile(debtor_profile)
        
        base_amount = self.initial_claim * analysis["initial_offer_multiplier"]
        
        if self.strategy == NegotiationStrategy.AGGRESSIVE:
            offer_amount = base_amount * 0.95
            timeline = 30
            interest_rate = 0.12
        elif self.strategy == NegotiationStrategy.MODERATE:
            offer_amount = base_amount * 0.85
            timeline = 90
            interest_rate = 0.08
        elif self.strategy == NegotiationStrategy.FLEXIBLE:
            offer_amount = base_amount * 0.75
            timeline = 180
            interest_rate = 0.05
        else:  # ADAPTIVE
            offer_amount = base_amount * 0.80
            timeline = 120
            interest_rate = 0.06
        
        offer = Offer(
            amount=offer_amount,
            payment_terms=self._generate_payment_terms(offer_amount, timeline),
            interest_rate=interest_rate,
            timeline_days=timeline,
            penalties=self._generate_penalty_structure()
        )
        
        self.current_offer = offer
        self.negotiation_history.append({
            "type": "initial_offer",
            "offer": offer,
            "timestamp": datetime.now().isoformat()
        })
        
        return offer
    
    def evaluate_counter_offer(self, counter_offer: Offer) -> Tuple[bool, Optional[Offer]]:
        """Evaluate counter-offer from debtor"""
        self.negotiation_history.append({
            "type": "counter_offer_received",
            "offer": counter_offer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Calculate offer quality
        recovery_rate = counter_offer.amount / self.initial_claim
        time_value_adjustment = self._calculate_time_value(counter_offer.timeline_days)
        adjusted_recovery = recovery_rate * time_value_adjustment
        
        # Q-learning state
        state = self._get_state(counter_offer)
        
        # Decision making
        if adjusted_recovery >= self.min_acceptable / self.initial_claim:
            # Accept offer
            self._update_q_value(state, "accept", adjusted_recovery)
            self.successful_negotiations += 1
            self.total_negotiations += 1
            self._update_metrics(recovery_rate)
            return True, None
        
        elif len(self.negotiation_history) >= self.patience_level:
            # Final offer
            final_offer = self._generate_final_offer()
            self._update_q_value(state, "final_offer", 0.7)
            return False, final_offer
        
        else:
            # Generate counter-offer
            new_offer = self._generate_counter_offer(counter_offer)
            self._update_q_value(state, "counter", 0.5)
            return False, new_offer
    
    def _calculate_time_value(self, days: int) -> float:
        """Calculate time value adjustment factor"""
        annual_discount_rate = 0.10
        years = days / 365.0
        return 1 / ((1 + annual_discount_rate) ** years)
    
    def _generate_payment_terms(self, amount: float, timeline_days: int) -> Dict:
        """Generate structured payment terms"""
        if timeline_days <= 30:
            return {
                "type": "lump_sum",
                "installments": 1,
                "amounts": [amount],
                "due_dates": [timeline_days]
            }
        else:
            num_installments = min(timeline_days // 30, 12)
            installment_amount = amount / num_installments
            return {
                "type": "installment",
                "installments": num_installments,
                "amounts": [installment_amount] * num_installments,
                "due_dates": [(i + 1) * 30 for i in range(num_installments)]
            }
    
    def _generate_penalty_structure(self) -> Dict:
        """Generate penalty structure for defaults"""
        return {
            "late_payment_fee": 0.05,  # 5% of installment
            "default_interest": 0.15,   # 15% annual
            "grace_period_days": 7,
            "acceleration_clause": True  # Full amount due on default
        }
    
    def _generate_counter_offer(self, previous_offer: Offer) -> Offer:
        """Generate strategic counter-offer"""
        rounds = len([h for h in self.negotiation_history if h["type"] == "counter_offer_sent"])
        
        # Gradual concession strategy
        concession_rate = 0.05 * (1 + rounds * 0.1)
        
        if self.current_offer:
            new_amount = self.current_offer.amount * (1 - concession_rate)
            new_amount = max(new_amount, self.min_acceptable)
        else:
            new_amount = self.initial_claim * 0.9
        
        # Adjust other terms
        new_timeline = min(previous_offer.timeline_days + 30, 365)
        new_interest = max(previous_offer.interest_rate - 0.01, 0.03)
        
        offer = Offer(
            amount=new_amount,
            payment_terms=self._generate_payment_terms(new_amount, new_timeline),
            interest_rate=new_interest,
            timeline_days=new_timeline,
            penalties=self._generate_penalty_structure()
        )
        
        self.current_offer = offer
        self.negotiation_history.append({
            "type": "counter_offer_sent",
            "offer": offer,
            "timestamp": datetime.now().isoformat()
        })
        
        return offer
    
    def _generate_final_offer(self) -> Offer:
        """Generate final take-it-or-leave-it offer"""
        offer = Offer(
            amount=self.min_acceptable,
            payment_terms=self._generate_payment_terms(self.min_acceptable, 180),
            interest_rate=0.04,
            timeline_days=180,
            penalties=self._generate_penalty_structure()
        )
        
        self.negotiation_history.append({
            "type": "final_offer",
            "offer": offer,
            "timestamp": datetime.now().isoformat()
        })
        
        return offer
    
    def _get_state(self, offer: Offer) -> str:
        """Convert offer to state for Q-learning"""
        recovery_bucket = int((offer.amount / self.initial_claim) * 10)
        timeline_bucket = min(offer.timeline_days // 30, 12)
        round_number = len(self.negotiation_history)
        
        return f"{recovery_bucket}_{timeline_bucket}_{round_number}"
    
    def _update_q_value(self, state: str, action: str, reward: float):
        """Update Q-table with learning"""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # Q-learning update
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.learning_rate * (reward - old_value)
    
    def _update_metrics(self, recovery_rate: float):
        """Update agent performance metrics"""
        n = self.successful_negotiations
        self.average_recovery_rate = (self.average_recovery_rate * (n - 1) + recovery_rate) / n
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        return {
            "agent_id": self.agent_id,
            "total_negotiations": self.total_negotiations,
            "successful_negotiations": self.successful_negotiations,
            "success_rate": self.successful_negotiations / max(self.total_negotiations, 1),
            "average_recovery_rate": self.average_recovery_rate,
            "current_strategy": self.strategy.value,
            "q_table_size": len(self.q_table)
        }
    
    async def negotiate(self, debtor_agent, mediator=None):
        """Main negotiation loop"""
        debtor_profile = await debtor_agent.get_profile()
        
        # Generate initial offer
        offer = self.generate_initial_offer(debtor_profile)
        
        rounds = 0
        max_rounds = 20
        
        while rounds < max_rounds:
            # Send offer to debtor
            counter_offer = await debtor_agent.receive_offer(offer)
            
            if counter_offer is None:
                # Debtor accepted
                return {
                    "status": "accepted",
                    "final_offer": offer,
                    "rounds": rounds
                }
            
            # Evaluate counter-offer
            accept, new_offer = self.evaluate_counter_offer(counter_offer)
            
            if accept:
                return {
                    "status": "accepted",
                    "final_offer": counter_offer,
                    "rounds": rounds
                }
            
            if new_offer:
                offer = new_offer
            else:
                break
            
            rounds += 1
        
        return {
            "status": "failed",
            "final_offer": None,
            "rounds": rounds
        }