import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from scipy.optimize import linprog

class MediationStrategy(Enum):
    FACILITATIVE = "facilitative"  # Help parties communicate
    EVALUATIVE = "evaluative"      # Provide assessment
    TRANSFORMATIVE = "transformative"  # Focus on empowerment
    DIRECTIVE = "directive"        # Strongly guide to solution

@dataclass
class MediationSession:
    session_id: str
    creditor_id: str
    debtor_id: str
    start_time: datetime
    status: str
    rounds: int = 0
    impasse_count: int = 0

class MediatorAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.active_sessions = {}
        self.completed_sessions = []
        self.strategy = MediationStrategy.FACILITATIVE
        
        # Mediation expertise
        self.domain_knowledge = self._initialize_domain_knowledge()
        self.fairness_metrics = {}
        self.success_patterns = {}
        
        # Learning components
        self.case_database = []
        self.pattern_recognition_model = {}
        self.trust_scores = {}  # Trust scores for agents
        
        # Game theory components
        self.nash_solver = NashEquilibriumSolver()
        self.pareto_optimizer = ParetoOptimizer()
        
        # Performance tracking
        self.success_rate = 0.0
        self.average_rounds_to_agreement = 0
        self.fairness_score = 0.0
        
    def _initialize_domain_knowledge(self) -> Dict:
        """Initialize domain-specific mediation knowledge"""
        return {
            "settlement_benchmarks": {
                "excellent_credit": 0.7,    # 70% of debt
                "good_credit": 0.6,         # 60% of debt
                "fair_credit": 0.5,         # 50% of debt
                "poor_credit": 0.4          # 40% of debt
            },
            "timeline_standards": {
                "immediate": 30,
                "short_term": 90,
                "medium_term": 180,
                "long_term": 365
            },
            "interest_guidelines": {
                "hardship": 0.0,
                "standard": 0.05,
                "penalty": 0.10
            }
        }
    
    async def initiate_mediation(self, creditor_agent, debtor_agent) -> MediationSession:
        """Initiate a new mediation session"""
        session_id = f"mediation_{datetime.now().timestamp()}"
        
        session = MediationSession(
            session_id=session_id,
            creditor_id=creditor_agent.agent_id,
            debtor_id=debtor_agent.agent_id,
            start_time=datetime.now(),
            status="active"
        )
        
        self.active_sessions[session_id] = {
            "session": session,
            "creditor": creditor_agent,
            "debtor": debtor_agent,
            "history": []
        }
        
        # Assess situation
        assessment = await self._assess_negotiation_landscape(creditor_agent, debtor_agent)
        self.active_sessions[session_id]["assessment"] = assessment
        
        # Select mediation strategy
        self.strategy = self._select_mediation_strategy(assessment)
        
        return session
    
    async def _assess_negotiation_landscape(self, creditor, debtor) -> Dict:
        """Comprehensive assessment of negotiation situation"""
        # Get profiles
        debtor_profile = await debtor.get_profile()
        
        # Calculate ZOPA (Zone of Possible Agreement)
        creditor_min = creditor.min_acceptable
        debtor_max = debtor.max_affordable
        
        zopa_exists = debtor_max >= creditor_min
        zopa_size = max(0, debtor_max - creditor_min)
        
        # Power dynamics assessment
        power_balance = self._assess_power_balance(creditor, debtor, debtor_profile)
        
        # Emotional state assessment
        emotional_factors = {
            "debtor_stress": debtor.stress_level,
            "creditor_patience": creditor.patience_level / 20,  # Normalize
            "urgency_mismatch": abs(debtor.urgency_factor - (1 - creditor.patience_level / 20))
        }
        
        # Calculate fair settlement range
        fair_range = self._calculate_fair_settlement_range(
            creditor.initial_claim,
            debtor_profile,
            debtor.financial_profile
        )
        
        return {
            "zopa_exists": zopa_exists,
            "zopa_size": zopa_size,
            "creditor_min": creditor_min,
            "debtor_max": debtor_max,
            "power_balance": power_balance,
            "emotional_factors": emotional_factors,
            "fair_range": fair_range,
            "complexity_score": self._calculate_complexity_score(zopa_exists, power_balance, emotional_factors)
        }
    
    def _assess_power_balance(self, creditor, debtor, debtor_profile) -> Dict:
        """Assess negotiation power dynamics"""
        # Creditor power factors
        creditor_power = 0.5
        creditor_power += 0.1 if creditor.strategy == NegotiationStrategy.AGGRESSIVE else 0
        creditor_power += 0.1 * (creditor.initial_claim / 100000)  # Size of claim
        
        # Debtor power factors
        debtor_power = 0.5
        debtor_power += 0.1 if debtor_profile["credit_score"] > 700 else -0.1
        debtor_power += 0.1 if debtor.financial_profile.employment_status == "stable" else -0.1
        debtor_power += 0.1 if len(debtor.financial_profile.hardship_factors) > 2 else 0
        
        # Normalize
        total_power = creditor_power + debtor_power
        
        return {
            "creditor_power": creditor_power / total_power,
            "debtor_power": debtor_power / total_power,
            "balance_index": 1 - abs(creditor_power - debtor_power) / total_power
        }
    
    def _calculate_fair_settlement_range(self, claim_amount: float, debtor_profile: Dict, financial_profile) -> Dict:
        """Calculate objectively fair settlement range"""
        # Base on credit score
        credit_score = debtor_profile["credit_score"]
        if credit_score >= 750:
            base_rate = 0.75
        elif credit_score >= 650:
            base_rate = 0.60
        elif credit_score >= 550:
            base_rate = 0.45
        else:
            base_rate = 0.35
        
        # Adjust for financial hardship
        hardship_adjustment = len(financial_profile.hardship_factors) * 0.05
        base_rate -= min(hardship_adjustment, 0.15)
        
        # Consider payment capacity
        capacity_ratio = financial_profile.liquid_assets / claim_amount
        if capacity_ratio > 0.5:
            base_rate += 0.1
        
        fair_midpoint = claim_amount * base_rate
        fair_range = {
            "minimum": fair_midpoint * 0.85,
            "midpoint": fair_midpoint,
            "maximum": fair_midpoint * 1.15
        }
        
        return fair_range
    
    def _calculate_complexity_score(self, zopa_exists: bool, power_balance: Dict, emotional_factors: Dict) -> float:
        """Calculate case complexity score"""
        score = 0.0
        
        # ZOPA factor
        if not zopa_exists:
            score += 0.4
        
        # Power imbalance
        score += (1 - power_balance["balance_index"]) * 0.3
        
        # Emotional complexity
        avg_emotion = np.mean(list(emotional_factors.values()))
        score += avg_emotion * 0.3
        
        return score
    
    def _select_mediation_strategy(self, assessment: Dict) -> MediationStrategy:
        """Select appropriate mediation strategy"""
        complexity = assessment["complexity_score"]
        
        if not assessment["zopa_exists"]:
            return MediationStrategy.DIRECTIVE
        elif complexity > 0.7:
            return MediationStrategy.EVALUATIVE
        elif assessment["emotional_factors"]["urgency_mismatch"] > 0.5:
            return MediationStrategy.TRANSFORMATIVE
        else:
            return MediationStrategy.FACILITATIVE
    
    async def mediate_round(self, session_id: str, creditor_offer: 'Offer', debtor_counter: Optional['Offer']) -> Dict:
        """Mediate a single round of negotiation"""
        session_data = self.active_sessions[session_id]
        session = session_data["session"]
        session.rounds += 1
        
        # Record offers
        session_data["history"].append({
            "round": session.rounds,
            "creditor_offer": creditor_offer,
            "debtor_counter": debtor_counter,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check for impasse
        if self._detect_impasse(session_data["history"]):
            session.impasse_count += 1
            return await self._handle_impasse(session_id)
        
        # Generate mediation interventions
        interventions = self._generate_interventions(session_id, creditor_offer, debtor_counter)
        
        # Apply strategy-specific mediation
        if self.strategy == MediationStrategy.FACILITATIVE:
            result = await self._facilitative_mediation(session_id, interventions)
        elif self.strategy == MediationStrategy.EVALUATIVE:
            result = await self._evaluative_mediation(session_id, interventions)
        elif self.strategy == MediationStrategy.TRANSFORMATIVE:
            result = await self._transformative_mediation(session_id, interventions)
        else:  # DIRECTIVE
            result = await self._directive_mediation(session_id, interventions)
        
        return result
    
    def _detect_impasse(self, history: List[Dict]) -> bool:
        """Detect negotiation impasse"""
        if len(history) < 3:
            return False
        
        # Check for repetitive offers
        recent_offers = history[-3:]
        creditor_amounts = [h["creditor_offer"].amount for h in recent_offers]
        debtor_amounts = [h["debtor_counter"].amount if h["debtor_counter"] else 0 for h in recent_offers]
        
        # Minimal movement indicates impasse
        creditor_movement = max(creditor_amounts) - min(creditor_amounts)
        debtor_movement = max(debtor_amounts) - min(debtor_amounts)
        
        return creditor_movement < 0.01 * creditor_amounts[0] and debtor_movement < 0.01 * debtor_amounts[0]
    
    async def _handle_impasse(self, session_id: str) -> Dict:
        """Handle negotiation impasse"""
        session_data = self.active_sessions[session_id]
        assessment = session_data["assessment"]
        
        # Calculate middle ground using game theory
        nash_point = self.nash_solver.find_equilibrium(
            assessment["creditor_min"],
            assessment["debtor_max"],
            assessment["fair_range"]
        )
        
        # Generate breakthrough proposal
        breakthrough_offer = self._generate_breakthrough_proposal(session_id, nash_point)
        
        return {
            "type": "impasse_intervention",
            "proposed_solution": breakthrough_offer,
            "rationale": self._explain_proposal(breakthrough_offer, assessment),
            "alternative_options": self._generate_alternatives(session_id)
        }
    
    def _generate_interventions(self, session_id: str, creditor_offer: 'Offer', debtor_counter: Optional['Offer']) -> List[Dict]:
        """Generate mediation interventions"""
        interventions = []
        session_data = self.active_sessions[session_id]
        assessment = session_data["assessment"]
        
        # Reality testing
        if creditor_offer.amount > assessment["debtor_max"] * 1.2:
            interventions.append({
                "type": "reality_test",
                "target": "creditor",
                "message": "Consider debtor's demonstrated payment capacity",
                "data": {"max_capacity": assessment["debtor_max"]}
            })
        
        if debtor_counter and debtor_counter.amount < assessment["creditor_min"] * 0.8:
            interventions.append({
                "type": "reality_test",
                "target": "debtor",
                "message": "Offer may be below creditor's minimum threshold",
                "data": {"likely_minimum": assessment["creditor_min"]}
            })
        
        # Interest identification
        gap = creditor_offer.amount - (debtor_counter.amount if debtor_counter else 0)
        if gap > assessment["fair_range"]["midpoint"] * 0.3:
            interventions.append({
                "type": "interest_exploration",
                "message": "Explore non-monetary terms that could bridge the gap",
                "suggestions": self._generate_creative_solutions(session_id)
            })
        
        return interventions
    
    async def _facilitative_mediation(self, session_id: str, interventions: List[Dict]) -> Dict:
        """Facilitative mediation approach"""
        session_data = self.active_sessions[session_id]
        
        # Focus on improving communication
        communication_aids = {
            "reframing": self._reframe_positions(session_data["history"][-1]),
            "summarization": self._summarize_progress(session_data["history"]),
            "common_ground": self._identify_common_ground(session_data)
        }
        
        return {
            "type": "facilitative",
            "interventions": interventions,
            "communication_aids": communication_aids,
            "next_steps": ["Encourage direct dialogue", "Focus on interests, not positions"]
        }
    
    async def _evaluative_mediation(self, session_id: str, interventions: List[Dict]) -> Dict:
        """Evaluative mediation approach"""
        session_data = self.active_sessions[session_id]
        assessment = session_data["assessment"]
        
        # Provide expert evaluation
        evaluation = {
            "fair_settlement_range": assessment["fair_range"],
            "market_comparison": self._get_market_benchmarks(session_data),
            "legal_perspective": self._assess_legal_position(session_data),
            "likely_court_outcome": self._predict_litigation_outcome(session_data)
        }
        
        # Risk analysis
        risk_assessment = {
            "creditor_risks": self._assess_creditor_risks(session_data),
            "debtor_risks": self._assess_debtor_risks(session_data)
        }
        
        return {
            "type": "evaluative",
            "interventions": interventions,
            "expert_evaluation": evaluation,
            "risk_assessment": risk_assessment,
            "recommendation": self._generate_recommendation(evaluation, risk_assessment)
        }
    
    async def _transformative_mediation(self, session_id: str, interventions: List[Dict]) -> Dict:
        """Transformative mediation approach"""
        session_data = self.active_sessions[session_id]
        
        # Focus on empowerment and recognition
        empowerment_strategies = {
            "creditor": self._empower_party(session_data["creditor"]),
            "debtor": self._empower_party(session_data["debtor"])
        }
        
        recognition_opportunities = {
            "acknowledge_creditor_flexibility": True,
            "recognize_debtor_efforts": True,
            "mutual_understanding": self._facilitate_mutual_understanding(session_data)
        }
        
        return {
            "type": "transformative",
            "interventions": interventions,
            "empowerment": empowerment_strategies,
            "recognition": recognition_opportunities,
            "relationship_focus": "Rebuild trust and understanding"
        }
    
    async def _directive_mediation(self, session_id: str, interventions: List[Dict]) -> Dict:
        """Directive mediation approach"""
        session_data = self.active_sessions[session_id]
        assessment = session_data["assessment"]
        
        # Calculate optimal solution
        optimal_solution = self.pareto_optimizer.find_optimal_solution(
            assessment["creditor_min"],
            assessment["debtor_max"],
            assessment["fair_range"],
            session_data["history"]
        )
        
        # Generate directive proposal
        directive_offer = Offer(
            amount=optimal_solution["amount"],
            payment_terms=optimal_solution["payment_terms"],
            interest_rate=optimal_solution["interest_rate"],
            timeline_days=optimal_solution["timeline"]
        )
        
        return {
            "type": "directive",
            "interventions": interventions,
            "mediator_proposal": directive_offer,
            "justification": self._justify_directive_proposal(directive_offer, assessment),
            "implementation_plan": self._create_implementation_plan(directive_offer)
        }
    
    def _generate_breakthrough_proposal(self, session_id: str, nash_point: float) -> 'Offer':
        """Generate breakthrough proposal for impasse"""
        session_data = self.active_sessions[session_id]
        
        # Creative payment structure
        payment_structure = {
            "type": "performance_based",
            "base_amount": nash_point * 0.8,
            "performance_bonus": nash_point * 0.2,
            "conditions": ["timely_payments", "no_default"],
            "incentives": {"early_payment_discount": 0.05}
        }
        
        return Offer(
            amount=nash_point,
            payment_terms=payment_structure,
            interest_rate=0.03,  # Minimal interest
            timeline_days=180,   # Moderate timeline
            collateral={"good_faith_deposit": nash_point * 0.1}
        )
    
    def _generate_creative_solutions(self, session_id: str) -> List[Dict]:
        """Generate creative solutions to bridge gaps"""
        return [
            {
                "type": "payment_flexibility",
                "description": "Graduated payment plan starting low",
                "benefit": "Allows debtor to stabilize finances"
            },
            {
                "type": "performance_incentives",
                "description": "Reduced total if payments are timely",
                "benefit": "Motivates compliance"
            },
            {
                "type": "non_monetary",
                "description": "Partial payment through services/assets",
                "benefit": "Leverages debtor's non-cash resources"
            },
            {
                "type": "contingent_terms",
                "description": "Adjust payments based on income changes",
                "benefit": "Shares risk between parties"
            }
        ]
    
    def _update_trust_scores(self, agent_id: str, behavior: str):
        """Update trust scores based on agent behavior"""
        if agent_id not in self.trust_scores:
            self.trust_scores[agent_id] = 0.5
        
        if behavior == "constructive":
            self.trust_scores[agent_id] = min(1.0, self.trust_scores[agent_id] + 0.1)
        elif behavior == "obstructive":
            self.trust_scores[agent_id] = max(0.0, self.trust_scores[agent_id] - 0.1)
    
    def close_session(self, session_id: str, outcome: Dict) -> Dict:
        """Close mediation session and update learning"""
        session_data = self.active_sessions[session_id]
        session = session_data["session"]
        session.status = "completed"
        
        # Record outcome
        case_record = {
            "session": session,
            "outcome": outcome,
            "assessment": session_data["assessment"],
            "history": session_data["history"],
            "strategy_used": self.strategy.value
        }
        
        self.case_database.append(case_record)
        self.completed_sessions.append(session)
        
        # Update performance metrics
        self._update_performance_metrics(case_record)
        
        # Learn from case
        self._extract_patterns(case_record)
        
        # Clean up
        del self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "duration": (datetime.now() - session.start_time).total_seconds(),
            "rounds": session.rounds,
            "outcome": outcome,
            "performance_impact": self._calculate_performance_impact(case_record)
        }
    
    def _update_performance_metrics(self, case_record: Dict):
        """Update mediator performance metrics"""
        total_cases = len(self.completed_sessions)
        
        if case_record["outcome"]["status"] == "agreement":
            self.success_rate = ((self.success_rate * (total_cases - 1)) + 1) / total_cases
        else:
            self.success_rate = (self.success_rate * (total_cases - 1)) / total_cases
        
        # Update average rounds
        self.average_rounds_to_agreement = (
            (self.average_rounds_to_agreement * (total_cases - 1) + case_record["session"].rounds) / total_cases
        )
        
        # Update fairness score
        if "fairness_rating" in case_record["outcome"]:
            self.fairness_score = (
                (self.fairness_score * (total_cases - 1) + case_record["outcome"]["fairness_rating"]) / total_cases
            )
    
    def _extract_patterns(self, case_record: Dict):
        """Extract and learn patterns from completed case"""
        # Pattern extraction logic
        outcome = case_record["outcome"]["status"]
        initial_gap = case_record["history"][0]["creditor_offer"].amount - case_record["history"][0]["debtor_counter"].amount
        
        pattern_key = f"{case_record['assessment']['complexity_score']:.1f}_{self.strategy.value}_{outcome}"
        
        if pattern_key not in self.success_patterns:
            self.success_patterns[pattern_key] = {
                "count": 0,
                "avg_rounds": 0,
                "avg_satisfaction": 0
            }
        
        pattern = self.success_patterns[pattern_key]
        pattern["count"] += 1
        pattern["avg_rounds"] = (pattern["avg_rounds"] * (pattern["count"] - 1) + case_record["session"].rounds) / pattern["count"]


class NashEquilibriumSolver:
    """Solver for finding Nash equilibrium in negotiation"""
    
    def find_equilibrium(self, creditor_min: float, debtor_max: float, fair_range: Dict) -> float:
        """Find Nash equilibrium point"""
        if debtor_max < creditor_min:
            # No ZOPA, return fair midpoint
            return fair_range["midpoint"]
        
        # Nash bargaining solution
        # Assumes equal bargaining power
        return (creditor_min + debtor_max) / 2


class ParetoOptimizer:
    """Optimizer for finding Pareto optimal solutions"""
    
    def find_optimal_solution(self, creditor_min: float, debtor_max: float, 
                             fair_range: Dict, history: List[Dict]) -> Dict:
        """Find Pareto optimal solution"""
        # Simple implementation - can be enhanced with more sophisticated optimization
        optimal_amount = (fair_range["midpoint"] + (creditor_min + debtor_max) / 2) / 2
        
        return {
            "amount": optimal_amount,
            "payment_terms": {
                "type": "installment",
                "installments": 6,
                "amounts": [optimal_amount / 6] * 6,
                "due_dates": [(i + 1) * 30 for i in range(6)]
            },
            "interest_rate": 0.04,
            "timeline": 180
        }