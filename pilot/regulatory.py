import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from dataclasses import dataclass
from enum import Enum
import hashlib
from collections import defaultdict

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    UNDER_REVIEW = "under_review"

class RegulatoryFramework(Enum):
    FDCPA = "fair_debt_collection_practices"
    TCPA = "telephone_consumer_protection"
    FCRA = "fair_credit_reporting"
    GDPR = "general_data_protection"
    CCPA = "california_consumer_privacy"
    BASEL_III = "basel_iii_capital_requirements"

@dataclass
class ComplianceRule:
    rule_id: str
    framework: RegulatoryFramework
    description: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    automated_check: bool
    penalty_range: Tuple[float, float]

@dataclass
class ComplianceEvent:
    event_id: str
    timestamp: datetime
    agent_id: str
    rule_id: str
    status: ComplianceStatus
    details: Dict
    remediation_required: bool

@dataclass
class AuditRecord:
    record_id: str
    negotiation_id: str
    participants: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    events: List[Dict]
    compliance_status: ComplianceStatus
    hash_chain: str

class RegulatoryAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.compliance_rules = self._initialize_compliance_rules()
        self.active_monitoring = {}
        self.audit_trail = []
        self.violation_history = defaultdict(list)
        
        # Compliance monitoring
        self.real_time_monitors = {}
        self.compliance_thresholds = self._set_compliance_thresholds()
        self.risk_scores = {}
        
        # ML components for pattern detection
        self.anomaly_detector = AnomalyDetectionModel()
        self.fairness_analyzer = FairnessAnalyzer()
        self.pattern_recognizer = CompliancePatternRecognizer()
        
        # Blockchain for audit trail
        self.blockchain = ComplianceBlockchain()
        
        # Reporting and analytics
        self.compliance_metrics = {
            "total_negotiations_monitored": 0,
            "violations_detected": 0,
            "warnings_issued": 0,
            "average_compliance_score": 1.0
        }
        
    def _initialize_compliance_rules(self) -> Dict[str, ComplianceRule]:
        """Initialize comprehensive compliance rules"""
        rules = {}
        
        # FDCPA Rules
        rules["FDCPA_001"] = ComplianceRule(
            rule_id="FDCPA_001",
            framework=RegulatoryFramework.FDCPA,
            description="No harassment or abuse in debt collection",
            severity="critical",
            automated_check=True,
            penalty_range=(1000, 1000000)
        )
        
        rules["FDCPA_002"] = ComplianceRule(
            rule_id="FDCPA_002",
            framework=RegulatoryFramework.FDCPA,
            description="No false or misleading representations",
            severity="critical",
            automated_check=True,
            penalty_range=(1000, 1000000)
        )
        
        rules["FDCPA_003"] = ComplianceRule(
            rule_id="FDCPA_003",
            framework=RegulatoryFramework.FDCPA,
            description="Required disclosures in communications",
            severity="high",
            automated_check=True,
            penalty_range=(500, 500000)
        )
        
        # Interest Rate Limits
        rules["USURY_001"] = ComplianceRule(
            rule_id="USURY_001",
            framework=RegulatoryFramework.FDCPA,
            description="Interest rates must not exceed state usury limits",
            severity="critical",
            automated_check=True,
            penalty_range=(5000, 5000000)
        )
        
        # Data Protection Rules
        rules["GDPR_001"] = ComplianceRule(
            rule_id="GDPR_001",
            framework=RegulatoryFramework.GDPR,
            description="Obtain consent for data processing",
            severity="high",
            automated_check=True,
            penalty_range=(10000, 20000000)
        )
        
        rules["GDPR_002"] = ComplianceRule(
            rule_id="GDPR_002",
            framework=RegulatoryFramework.GDPR,
            description="Right to data erasure (right to be forgotten)",
            severity="high",
            automated_check=True,
            penalty_range=(10000, 20000000)
        )
        
        # Fair Credit Reporting
        rules["FCRA_001"] = ComplianceRule(
            rule_id="FCRA_001",
            framework=RegulatoryFramework.FCRA,
            description="Accurate credit reporting required",
            severity="high",
            automated_check=True,
            penalty_range=(2500, 2500000)
        )
        
        # Market Manipulation
        rules["MARKET_001"] = ComplianceRule(
            rule_id="MARKET_001",
            framework=RegulatoryFramework.BASEL_III,
            description="No market manipulation or insider trading",
            severity="critical",
            automated_check=True,
            penalty_range=(10000, 10000000)
        )
        
        # Fairness and Non-Discrimination
        rules["FAIR_001"] = ComplianceRule(
            rule_id="FAIR_001",
            framework=RegulatoryFramework.FDCPA,
            description="No discrimination based on protected characteristics",
            severity="critical",
            automated_check=True,
            penalty_range=(5000, 5000000)
        )
        
        return rules
    
    def _set_compliance_thresholds(self) -> Dict:
        """Set compliance monitoring thresholds"""
        return {
            "max_interest_rate": 0.25,  # 25% APR max
            "min_disclosure_requirements": ["total_amount", "interest_rate", "payment_terms", "penalties"],
            "max_contact_frequency": 7,  # Max contacts per week
            "quiet_hours": (21, 8),  # No contact 9 PM - 8 AM
            "max_settlement_pressure": 0.8,  # Max 80% of original debt in first offer
            "data_retention_days": 2555,  # 7 years
            "consent_required_actions": ["data_sharing", "credit_reporting", "automated_decision"]
        }
    
    async def monitor_negotiation(self, negotiation_id: str, creditor_agent, debtor_agent) -> AuditRecord:
        """Begin monitoring a negotiation for compliance"""
        audit_record = AuditRecord(
            record_id=f"audit_{datetime.now().timestamp()}",
            negotiation_id=negotiation_id,
            participants=[creditor_agent.agent_id, debtor_agent.agent_id],
            start_time=datetime.now(),
            end_time=None,
            events=[],
            compliance_status=ComplianceStatus.COMPLIANT,
            hash_chain=""
        )
        
        self.active_monitoring[negotiation_id] = {
            "audit_record": audit_record,
            "creditor": creditor_agent,
            "debtor": debtor_agent,
            "real_time_score": 1.0,
            "violations": []
        }
        
        # Initialize blockchain record
        self.blockchain.add_block({
            "type": "negotiation_start",
            "negotiation_id": negotiation_id,
            "participants": audit_record.participants,
            "timestamp": audit_record.start_time.isoformat()
        })
        
        self.compliance_metrics["total_negotiations_monitored"] += 1
        
        return audit_record
    
    async def check_offer_compliance(self, negotiation_id: str, offer: 'Offer', 
                                   offering_agent_id: str) -> ComplianceEvent:
        """Check if an offer complies with regulations"""
        monitoring_data = self.active_monitoring.get(negotiation_id)
        if not monitoring_data:
            return self._create_compliance_event(
                negotiation_id, "SYSTEM_001", ComplianceStatus.VIOLATION,
                {"error": "Unmonitored negotiation"}, True
            )
        
        violations = []
        warnings = []
        
        # Check interest rate compliance
        if offer.interest_rate > self.compliance_thresholds["max_interest_rate"]:
            violations.append({
                "rule_id": "USURY_001",
                "detail": f"Interest rate {offer.interest_rate:.2%} exceeds maximum {self.compliance_thresholds['max_interest_rate']:.2%}"
            })
        
        # Check for required disclosures
        if hasattr(offer, 'disclosures'):
            missing_disclosures = set(self.compliance_thresholds["min_disclosure_requirements"]) - set(offer.disclosures.keys())
            if missing_disclosures:
                violations.append({
                    "rule_id": "FDCPA_003",
                    "detail": f"Missing required disclosures: {missing_disclosures}"
                })
        
        # Check for fair settlement practices
        creditor = monitoring_data["creditor"]
        if hasattr(creditor, 'initial_claim'):
            settlement_ratio = offer.amount / creditor.initial_claim
            if settlement_ratio > self.compliance_thresholds["max_settlement_pressure"]:
                warnings.append({
                    "rule_id": "FDCPA_001",
                    "detail": f"High initial settlement demand: {settlement_ratio:.2%} of original debt"
                })
        
        # Check for discriminatory patterns using ML
        fairness_check = await self.fairness_analyzer.analyze_offer(offer, monitoring_data["debtor"])
        if fairness_check["bias_detected"]:
            violations.append({
                "rule_id": "FAIR_001",
                "detail": f"Potential discrimination detected: {fairness_check['bias_type']}"
            })
        
        # Determine overall compliance status
        if violations:
            status = ComplianceStatus.VIOLATION
            remediation = True
        elif warnings:
            status = ComplianceStatus.WARNING
            remediation = False
        else:
            status = ComplianceStatus.COMPLIANT
            remediation = False
        
        # Create compliance event
        event = self._create_compliance_event(
            negotiation_id, 
            violations[0]["rule_id"] if violations else warnings[0]["rule_id"] if warnings else "COMPLIANT",
            status,
            {
                "violations": violations,
                "warnings": warnings,
                "offer_details": {
                    "amount": offer.amount,
                    "interest_rate": offer.interest_rate,
                    "timeline_days": offer.timeline_days
                }
            },
            remediation
        )
        
        # Update monitoring data
        monitoring_data["events"].append(event)
        if violations:
            monitoring_data["violations"].extend(violations)
            self.compliance_metrics["violations_detected"] += len(violations)
        if warnings:
            self.compliance_metrics["warnings_issued"] += len(warnings)
        
        # Update real-time compliance score
        self._update_compliance_score(negotiation_id, status)
        
        # Add to blockchain
        self.blockchain.add_block({
            "type": "compliance_check",
            "negotiation_id": negotiation_id,
            "event_id": event.event_id,
            "status": status.value,
            "timestamp": event.timestamp.isoformat()
        })
        
        return event
    
    def _create_compliance_event(self, negotiation_id: str, rule_id: str, 
                               status: ComplianceStatus, details: Dict, 
                               remediation: bool) -> ComplianceEvent:
        """Create a compliance event"""
        return ComplianceEvent(
            event_id=f"compliance_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            rule_id=rule_id,
            status=status,
            details=details,
            remediation_required=remediation
        )
    
    def _update_compliance_score(self, negotiation_id: str, status: ComplianceStatus):
        """Update real-time compliance score"""
        monitoring_data = self.active_monitoring[negotiation_id]
        
        # Adjust score based on status
        if status == ComplianceStatus.VIOLATION:
            monitoring_data["real_time_score"] *= 0.7
        elif status == ComplianceStatus.WARNING:
            monitoring_data["real_time_score"] *= 0.9
        elif status == ComplianceStatus.COMPLIANT:
            monitoring_data["real_time_score"] = min(1.0, monitoring_data["real_time_score"] * 1.02)
        
        # Update average
        total_score = sum(m["real_time_score"] for m in self.active_monitoring.values())
        self.compliance_metrics["average_compliance_score"] = total_score / len(self.active_monitoring)
    
    async def check_communication_compliance(self, negotiation_id: str, 
                                           communication: Dict) -> ComplianceEvent:
        """Check communication compliance (frequency, timing, content)"""
        monitoring_data = self.active_monitoring.get(negotiation_id)
        if not monitoring_data:
            return None
        
        violations = []
        
        # Check timing (quiet hours)
        current_hour = datetime.now().hour
        quiet_start, quiet_end = self.compliance_thresholds["quiet_hours"]
        if quiet_start <= current_hour or current_hour < quiet_end:
            violations.append({
                "rule_id": "TCPA_001",
                "detail": f"Communication during quiet hours ({quiet_start}:00 - {quiet_end}:00)"
            })
        
        # Check frequency
        recent_comms = self._get_recent_communications(negotiation_id, days=7)
        if len(recent_comms) > self.compliance_thresholds["max_contact_frequency"]:
            violations.append({
                "rule_id": "FDCPA_001",
                "detail": f"Excessive communication frequency: {len(recent_comms)} in 7 days"
            })
        
        # Check content for harassment or misleading info
        content_analysis = await self.anomaly_detector.analyze_communication(communication)
        if content_analysis["harassment_score"] > 0.7:
            violations.append({
                "rule_id": "FDCPA_001",
                "detail": "Potential harassment detected in communication"
            })
        
        if content_analysis["misleading_score"] > 0.7:
            violations.append({
                "rule_id": "FDCPA_002",
                "detail": "Potentially misleading information detected"
            })
        
        status = ComplianceStatus.VIOLATION if violations else ComplianceStatus.COMPLIANT
        
        return self._create_compliance_event(
            negotiation_id, 
            violations[0]["rule_id"] if violations else "COMPLIANT",
            status,
            {"violations": violations, "communication": communication},
            len(violations) > 0
        )
    
    def _get_recent_communications(self, negotiation_id: str, days: int) -> List[Dict]:
        """Get recent communications for frequency analysis"""
        # In a real system, this would query a database
        # For now, return mock data based on monitoring history
        monitoring_data = self.active_monitoring.get(negotiation_id, {})
        events = monitoring_data.get("events", [])
        
        cutoff_date = datetime.now() - timedelta(days=days)
        return [e for e in events if e.timestamp > cutoff_date and "communication" in e.details]
    
    async def check_data_compliance(self, negotiation_id: str, data_action: Dict) -> ComplianceEvent:
        """Check data handling compliance (GDPR, CCPA)"""
        violations = []
        
        # Check for consent
        if data_action["action_type"] in self.compliance_thresholds["consent_required_actions"]:
            if not data_action.get("consent_obtained", False):
                violations.append({
                    "rule_id": "GDPR_001",
                    "detail": f"No consent for {data_action['action_type']}"
                })
        
        # Check data retention
        if "data_age_days" in data_action:
            if data_action["data_age_days"] > self.compliance_thresholds["data_retention_days"]:
                violations.append({
                    "rule_id": "GDPR_002",
                    "detail": f"Data retained beyond limit: {data_action['data_age_days']} days"
                })
        
        # Check for data minimization
        if "data_fields" in data_action:
            unnecessary_fields = self._check_data_minimization(data_action["data_fields"], data_action["purpose"])
            if unnecessary_fields:
                violations.append({
                    "rule_id": "GDPR_003",
                    "detail": f"Unnecessary data collected: {unnecessary_fields}"
                })
        
        status = ComplianceStatus.VIOLATION if violations else ComplianceStatus.COMPLIANT
        
        return self._create_compliance_event(
            negotiation_id,
            violations[0]["rule_id"] if violations else "COMPLIANT",
            status,
            {"violations": violations, "data_action": data_action},
            len(violations) > 0
        )
    
    def _check_data_minimization(self, fields: List[str], purpose: str) -> List[str]:
        """Check if collected data is minimal for stated purpose"""
        necessary_fields = {
            "payment_negotiation": ["name", "debt_amount", "contact_info"],
            "credit_check": ["name", "ssn", "address", "dob"],
            "identity_verification": ["name", "id_number", "address"]
        }
        
        required = set(necessary_fields.get(purpose, []))
        collected = set(fields)
        
        return list(collected - required)
    
    async def detect_market_manipulation(self, market_data: Dict) -> ComplianceEvent:
        """Detect potential market manipulation in debt trading"""
        manipulation_indicators = []
        
        # Check for wash trading
        if self._detect_wash_trading(market_data):
            manipulation_indicators.append({
                "type": "wash_trading",
                "confidence": 0.85,
                "detail": "Circular trading pattern detected"
            })
        
        # Check for price manipulation
        if self._detect_price_manipulation(market_data):
            manipulation_indicators.append({
                "type": "price_manipulation",
                "confidence": 0.75,
                "detail": "Abnormal price movements detected"
            })
        
        # Check for insider trading patterns
        insider_risk = await self.anomaly_detector.detect_insider_trading(market_data)
        if insider_risk["risk_score"] > 0.7:
            manipulation_indicators.append({
                "type": "insider_trading",
                "confidence": insider_risk["risk_score"],
                "detail": insider_risk["pattern_description"]
            })
        
        if manipulation_indicators:
            return self._create_compliance_event(
                market_data.get("market_id", "MARKET"),
                "MARKET_001",
                ComplianceStatus.VIOLATION,
                {"indicators": manipulation_indicators, "market_data": market_data},
                True
            )
        
        return self._create_compliance_event(
            market_data.get("market_id", "MARKET"),
            "COMPLIANT",
            ComplianceStatus.COMPLIANT,
            {"message": "No market manipulation detected"},
            False
        )
    
    def _detect_wash_trading(self, market_data: Dict) -> bool:
        """Detect wash trading patterns"""
        trades = market_data.get("recent_trades", [])
        
        # Look for circular trading patterns
        trade_graph = defaultdict(list)
        for trade in trades:
            trade_graph[trade["seller"]].append(trade["buyer"])
        
        # Simple cycle detection
        for start_node in trade_graph:
            if self._has_cycle(trade_graph, start_node, start_node, set()):
                return True
        
        return False
    
    def _has_cycle(self, graph: Dict, start: str, current: str, visited: set) -> bool:
        """Detect cycles in trading graph"""
        if current in visited and current == start:
            return True
        
        if current in visited:
            return False
        
        visited.add(current)
        
        for neighbor in graph.get(current, []):
            if self._has_cycle(graph, start, neighbor, visited):
                return True
        
        return False
    
    def _detect_price_manipulation(self, market_data: Dict) -> bool:
        """Detect price manipulation patterns"""
        prices = market_data.get("price_history", [])
        
        if len(prices) < 10:
            return False
        
        # Check for pump and dump pattern
        price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        # Rapid increase followed by rapid decrease
        max_increase = max(price_changes[:len(price_changes)//2])
        max_decrease = min(price_changes[len(price_changes)//2:])
        
        if max_increase > 0.2 and max_decrease < -0.15:
            return True
        
        return False
    
    async def generate_compliance_report(self, negotiation_id: str) -> Dict:
        """Generate comprehensive compliance report for a negotiation"""
        monitoring_data = self.active_monitoring.get(negotiation_id)
        if not monitoring_data:
            return {"error": "No monitoring data found"}
        
        audit_record = monitoring_data["audit_record"]
        
        # Calculate compliance statistics
        total_events = len(monitoring_data["events"])
        violations = [e for e in monitoring_data["events"] if e.status == ComplianceStatus.VIOLATION]
        warnings = [e for e in monitoring_data["events"] if e.status == ComplianceStatus.WARNING]
        
        # Generate risk assessment
        risk_assessment = self._assess_overall_risk(monitoring_data)
        
        # Generate recommendations
        recommendations = self._generate_remediation_recommendations(violations)
        
        report = {
            "negotiation_id": negotiation_id,
            "audit_record_id": audit_record.record_id,
            "participants": audit_record.participants,
            "duration": (datetime.now() - audit_record.start_time).total_seconds(),
            "compliance_score": monitoring_data["real_time_score"],
            "overall_status": self._determine_overall_status(monitoring_data),
            "statistics": {
                "total_events": total_events,
                "violations": len(violations),
                "warnings": len(warnings),
                "compliance_rate": (total_events - len(violations)) / total_events if total_events > 0 else 1.0
            },
            "violations_detail": [
                {
                    "rule_id": v.rule_id,
                    "timestamp": v.timestamp.isoformat(),
                    "details": v.details
                } for v in violations
            ],
            "risk_assessment": risk_assessment,
            "recommendations": recommendations,
            "blockchain_verification": self.blockchain.get_verification_hash(negotiation_id)
        }
        
        return report
    
    def _assess_overall_risk(self, monitoring_data: Dict) -> Dict:
        """Assess overall compliance risk"""
        violations = monitoring_data["violations"]
        
        # Count by severity
        severity_counts = defaultdict(int)
        for violation in violations:
            rule = self.compliance_rules.get(violation["rule_id"])
            if rule:
                severity_counts[rule.severity] += 1
        
        # Calculate risk score
        risk_score = (
            severity_counts["critical"] * 1.0 +
            severity_counts["high"] * 0.7 +
            severity_counts["medium"] * 0.4 +
            severity_counts["low"] * 0.2
        ) / max(len(violations), 1)
        
        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": "CRITICAL" if risk_score > 0.8 else "HIGH" if risk_score > 0.5 else "MEDIUM" if risk_score > 0.2 else "LOW",
            "severity_breakdown": dict(severity_counts),
            "primary_concerns": self._identify_primary_concerns(violations)
        }
    
    def _identify_primary_concerns(self, violations: List[Dict]) -> List[str]:
        """Identify primary compliance concerns"""
        concerns = set()
        
        for violation in violations:
            rule = self.compliance_rules.get(violation["rule_id"])
            if rule and rule.severity in ["critical", "high"]:
                concerns.add(rule.framework.value)
        
        return list(concerns)
    
    def _determine_overall_status(self, monitoring_data: Dict) -> str:
        """Determine overall compliance status"""
        if any(v["rule_id"] in ["FDCPA_001", "FDCPA_002", "MARKET_001"] for v in monitoring_data["violations"]):
            return "NON_COMPLIANT"
        elif monitoring_data["violations"]:
            return "PARTIALLY_COMPLIANT"
        elif any(e.status == ComplianceStatus.WARNING for e in monitoring_data["events"]):
            return "COMPLIANT_WITH_WARNINGS"
        else:
            return "FULLY_COMPLIANT"
    
    def _generate_remediation_recommendations(self, violations: List[ComplianceEvent]) -> List[Dict]:
        """Generate specific remediation recommendations"""
        recommendations = []
        
        for violation in violations:
            rule = self.compliance_rules.get(violation.rule_id)
            if not rule:
                continue
            
            if rule.framework == RegulatoryFramework.FDCPA:
                recommendations.append({
                    "priority": "HIGH",
                    "action": "Review and update debt collection practices",
                    "timeline": "Immediate",
                    "details": f"Address violation: {rule.description}"
                })
            
            elif rule.framework == RegulatoryFramework.GDPR:
                recommendations.append({
                    "priority": "CRITICAL",
                    "action": "Implement data protection measures",
                    "timeline": "Within 72 hours",
                    "details": f"Ensure compliance with: {rule.description}"
                })
            
            elif rule.rule_id == "MARKET_001":
                recommendations.append({
                    "priority": "CRITICAL",
                    "action": "Cease trading activity and conduct internal investigation",
                    "timeline": "Immediate",
                    "details": "Potential market manipulation detected"
                })
        
        return recommendations
    
    def close_monitoring(self, negotiation_id: str) -> Dict:
        """Close monitoring for a completed negotiation"""
        monitoring_data = self.active_monitoring.get(negotiation_id)
        if not monitoring_data:
            return {"error": "No active monitoring found"}
        
        # Finalize audit record
        audit_record = monitoring_data["audit_record"]
        audit_record.end_time = datetime.now()
        audit_record.compliance_status = ComplianceStatus.COMPLIANT if not monitoring_data["violations"] else ComplianceStatus.VIOLATION
        
        # Generate final hash
        audit_record.hash_chain = self.blockchain.finalize_chain(negotiation_id)
        
        # Archive to audit trail
        self.audit_trail.append(audit_record)
        
        # Update violation history
        for participant in audit_record.participants:
            if monitoring_data["violations"]:
                self.violation_history[participant].extend(monitoring_data["violations"])
        
        # Clean up active monitoring
        del self.active_monitoring[negotiation_id]
        
        return {
            "negotiation_id": negotiation_id,
            "final_status": audit_record.compliance_status.value,
            "duration": (audit_record.end_time - audit_record.start_time).total_seconds(),
            "final_score": monitoring_data["real_time_score"],
            "audit_hash": audit_record.hash_chain
        }


class AnomalyDetectionModel:
    """ML model for detecting anomalous patterns"""
    
    def __init__(self):
        self.harassment_keywords = ["threat", "sue", "arrest", "jail", "wages", "property"]
        self.misleading_patterns = ["guarantee", "risk-free", "immediate", "urgent", "final notice"]
        
    async def analyze_communication(self, communication: Dict) -> Dict:
        """Analyze communication for compliance issues"""
        content = communication.get("content", "").lower()
        
        # Harassment detection
        harassment_score = sum(1 for keyword in self.harassment_keywords if keyword in content) / len(self.harassment_keywords)
        
        # Misleading information detection
        misleading_score = sum(1 for pattern in self.misleading_patterns if pattern in content) / len(self.misleading_patterns)
        
        return {
            "harassment_score": harassment_score,
            "misleading_score": misleading_score,
            "sentiment": self._analyze_sentiment(content),
            "urgency_level": self._detect_urgency(content)
        }
    
    def _analyze_sentiment(self, content: str) -> str:
        """Simple sentiment analysis"""
        negative_words = ["threat", "bad", "consequences", "legal", "court"]
        positive_words = ["help", "assist", "solution", "work together", "flexible"]
        
        neg_count = sum(1 for word in negative_words if word in content)
        pos_count = sum(1 for word in positive_words if word in content)
        
        if neg_count > pos_count * 2:
            return "negative"
        elif pos_count > neg_count * 2:
            return "positive"
        else:
            return "neutral"
    
    def _detect_urgency(self, content: str) -> str:
        """Detect urgency level in communication"""
        urgent_words = ["immediately", "today", "now", "urgent", "final", "last chance"]
        urgent_count = sum(1 for word in urgent_words if word in content)
        
        if urgent_count >= 3:
            return "high"
        elif urgent_count >= 1:
            return "medium"
        else:
            return "low"
    
    async def detect_insider_trading(self, market_data: Dict) -> Dict:
        """Detect potential insider trading patterns"""
        # Simplified insider trading detection
        trades = market_data.get("recent_trades", [])
        
        # Look for unusual trading patterns before announcements
        suspicious_patterns = []
        
        for i, trade in enumerate(trades):
            if i < len(trades) - 1:
                next_trade = trades[i + 1]
                price_change = (next_trade["price"] - trade["price"]) / trade["price"]
                
                # Large trade followed by significant price movement
                if trade["volume"] > market_data.get("avg_volume", 0) * 2 and abs(price_change) > 0.1:
                    suspicious_patterns.append({
                        "trade_id": trade["id"],
                        "anomaly_score": min(abs(price_change) * 5, 1.0)
                    })
        
        if suspicious_patterns:
            max_score = max(p["anomaly_score"] for p in suspicious_patterns)
            return {
                "risk_score": max_score,
                "pattern_description": f"Detected {len(suspicious_patterns)} suspicious trades"
            }
        
        return {
            "risk_score": 0.0,
            "pattern_description": "No suspicious patterns detected"
        }


class FairnessAnalyzer:
    """Analyzer for detecting bias and ensuring fair treatment"""
    
    def __init__(self):
        self.protected_attributes = ["age", "gender", "race", "ethnicity", "religion", "disability"]
        self.baseline_terms = {}
        
    async def analyze_offer(self, offer: 'Offer', debtor_agent) -> Dict:
        """Analyze offer for potential bias"""
        bias_indicators = []
        
        # Check if offer terms significantly differ from baseline
        debtor_profile = await debtor_agent.get_profile()
        
        # Simple bias detection - in production, use more sophisticated ML
        if hasattr(debtor_agent, 'financial_profile'):
            profile = debtor_agent.financial_profile
            
            # Calculate expected terms based on financial factors only
            expected_rate = self._calculate_expected_rate(profile.credit_score)
            
            # Check for significant deviation
            if offer.interest_rate > expected_rate * 1.5:
                bias_indicators.append({
                    "type": "interest_rate_bias",
                    "severity": "high",
                    "deviation": offer.interest_rate / expected_rate
                })
        
        return {
            "bias_detected": len(bias_indicators) > 0,
            "bias_type": bias_indicators[0]["type"] if bias_indicators else None,
            "indicators": bias_indicators,
            "fairness_score": 1.0 - (len(bias_indicators) * 0.3)
        }
    
    def _calculate_expected_rate(self, credit_score: int) -> float:
        """Calculate expected interest rate based on credit score"""
        if credit_score >= 750:
            return 0.05
        elif credit_score >= 700:
            return 0.08
        elif credit_score >= 650:
            return 0.12
        elif credit_score >= 600:
            return 0.16
        else:
            return 0.20


class CompliancePatternRecognizer:
    """Recognize patterns in compliance violations"""
    
    def __init__(self):
        self.violation_patterns = defaultdict(list)
        
    def add_violation(self, agent_id: str, violation: Dict):
        """Add violation to pattern database"""
        self.violation_patterns[agent_id].append({
            "violation": violation,
            "timestamp": datetime.now()
        })
    
    def detect_patterns(self, agent_id: str) -> List[Dict]:
        """Detect violation patterns for an agent"""
        violations = self.violation_patterns.get(agent_id, [])
        
        if len(violations) < 3:
            return []
        
        patterns = []
        
        # Repeated violations
        violation_counts = defaultdict(int)
        for v in violations:
            violation_counts[v["violation"]["rule_id"]] += 1
        
        for rule_id, count in violation_counts.items():
            if count >= 3:
                patterns.append({
                    "type": "repeated_violation",
                    "rule_id": rule_id,
                    "count": count,
                    "severity": "high"
                })
        
        # Escalating violations
        if self._detect_escalation(violations):
            patterns.append({
                "type": "escalating_violations",
                "severity": "critical"
            })
        
        return patterns
    
    def _detect_escalation(self, violations: List[Dict]) -> bool:
        """Detect if violations are escalating in severity"""
        severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        
        severities = []
        for v in violations[-5:]:  # Last 5 violations
            rule_id = v["violation"]["rule_id"]
            # Would look up severity from rules in real implementation
            severities.append(2)  # Default medium
        
        # Check if severity is increasing
        return all(severities[i] <= severities[i+1] for i in range(len(severities)-1))


class ComplianceBlockchain:
    """Simple blockchain for compliance audit trail"""
    
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        
    def add_block(self, data: Dict) -> str:
        """Add block to chain"""
        block = {
            "index": len(self.chain),
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "previous_hash": self.chain[-1]["hash"] if self.chain else "0",
            "nonce": 0
        }
        
        block["hash"] = self._calculate_hash(block)
        self.chain.append(block)
        
        return block["hash"]
    
    def _calculate_hash(self, block: Dict) -> str:
        """Calculate block hash"""
        block_string = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def get_verification_hash(self, negotiation_id: str) -> str:
        """Get verification hash for negotiation"""
        relevant_blocks = [b for b in self.chain if b["data"].get("negotiation_id") == negotiation_id]
        
        if not relevant_blocks:
            return "NO_RECORDS"
        
        combined_hash = ""
        for block in relevant_blocks:
            combined_hash += block["hash"]
        
        return hashlib.sha256(combined_hash.encode()).hexdigest()
    
    def finalize_chain(self, negotiation_id: str) -> str:
        """Finalize chain for negotiation"""
        return self.get_verification_hash(negotiation_id)