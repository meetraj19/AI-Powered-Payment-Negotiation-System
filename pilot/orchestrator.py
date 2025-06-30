import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import uuid
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Import all agent modules
from creditor import CreditorAgent, NegotiationStrategy, Offer
from debitor import DebtorAgent, FinancialProfile, FinancialSituation
from mediator import MediatorAgent, MediationStrategy
from market_maker import MarketMakerAgent, DebtInstrument, MarketOrder
from regulatory import RegulatoryAgent, ComplianceStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"

class MessageType(Enum):
    NEGOTIATION_REQUEST = "negotiation_request"
    OFFER = "offer"
    COUNTER_OFFER = "counter_offer"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"
    MARKET_ORDER = "market_order"
    COMPLIANCE_CHECK = "compliance_check"
    SYSTEM_UPDATE = "system_update"

@dataclass
class Message:
    msg_id: str
    msg_type: MessageType
    sender_id: str
    receiver_id: str
    content: Dict
    timestamp: datetime
    priority: int = 5  # 1-10, 10 being highest

@dataclass
class NegotiationSession:
    session_id: str
    creditor_agent: CreditorAgent
    debtor_agent: DebtorAgent
    mediator_agent: Optional[MediatorAgent]
    regulatory_agent: RegulatoryAgent
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    outcome: Optional[Dict]
    compliance_record: Optional[Dict]

class PaymentNegotiationOrchestrator:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.system_state = SystemState.IDLE
        
        # Agent pools
        self.creditor_agents = {}
        self.debtor_agents = {}
        self.mediator_agents = {}
        self.market_maker_agents = {}
        self.regulatory_agents = {}
        
        # Active sessions
        self.active_sessions = {}
        self.session_history = []
        
        # Message handling
        self.message_queue = asyncio.Queue()
        self.priority_queue = queue.PriorityQueue()
        
        # Market infrastructure
        self.market_order_book = {"buy": [], "sell": []}
        self.market_stats = defaultdict(list)
        
        # System monitoring
        self.performance_metrics = {
            "total_negotiations": 0,
            "successful_negotiations": 0,
            "average_settlement_time": 0,
            "compliance_violations": 0,
            "system_uptime": datetime.now()
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config["max_workers"])
        
        # Initialize core agents
        self._initialize_core_agents()
        
    def _default_config(self) -> Dict:
        """Default system configuration"""
        return {
            "max_workers": 10,
            "max_concurrent_negotiations": 100,
            "enable_mediation": True,
            "enable_market_making": True,
            "compliance_mode": "strict",  # strict, moderate, lenient
            "auto_regulatory_reporting": True,
            "market_liquidity_target": 0.8,
            "system_risk_limit": 1000000,
            "message_timeout": 300,  # seconds
            "enable_ml_optimization": True
        }
    
    def _initialize_core_agents(self):
        """Initialize core system agents"""
        # Initialize regulatory agents
        self.regulatory_agents["REG_001"] = RegulatoryAgent("REG_001")
        
        # Initialize market maker agents
        if self.config["enable_market_making"]:
            self.market_maker_agents["MM_001"] = MarketMakerAgent("MM_001", 1000000)
            self.market_maker_agents["MM_002"] = MarketMakerAgent("MM_002", 500000)
        
        # Initialize mediator pool
        if self.config["enable_mediation"]:
            for i in range(5):
                mediator_id = f"MED_{i:03d}"
                self.mediator_agents[mediator_id] = MediatorAgent(mediator_id)
        
        logger.info(f"Initialized {len(self.regulatory_agents)} regulatory agents")
        logger.info(f"Initialized {len(self.market_maker_agents)} market maker agents")
        logger.info(f"Initialized {len(self.mediator_agents)} mediator agents")
    
    async def start_system(self):
        """Start the orchestrator system"""
        self.system_state = SystemState.ACTIVE
        logger.info("Payment Negotiation Orchestrator starting...")
        
        # Start background tasks
        asyncio.create_task(self._message_processor())
        asyncio.create_task(self._market_coordinator())
        asyncio.create_task(self._compliance_monitor())
        asyncio.create_task(self._performance_monitor())
        
        logger.info("System started successfully")
    
    async def create_creditor_agent(self, agent_id: str, initial_claim: float) -> CreditorAgent:
        """Create and register a new creditor agent"""
        agent = CreditorAgent(agent_id, initial_claim)
        self.creditor_agents[agent_id] = agent
        
        # Register with regulatory system
        await self._register_agent_with_regulatory(agent_id, "creditor")
        
        logger.info(f"Created creditor agent {agent_id} with claim ${initial_claim:,.2f}")
        return agent
    
    async def create_debtor_agent(self, agent_id: str, debt_amount: float, 
                                 financial_profile: FinancialProfile) -> DebtorAgent:
        """Create and register a new debtor agent"""
        agent = DebtorAgent(agent_id, debt_amount, financial_profile)
        self.debtor_agents[agent_id] = agent
        
        # Register with regulatory system
        await self._register_agent_with_regulatory(agent_id, "debtor")
        
        logger.info(f"Created debtor agent {agent_id} with debt ${debt_amount:,.2f}")
        return agent
    
    async def initiate_negotiation(self, creditor_id: str, debtor_id: str, 
                                 use_mediation: bool = None) -> str:
        """Initiate a new negotiation session"""
        if self.system_state != SystemState.ACTIVE:
            raise RuntimeError(f"System not active. Current state: {self.system_state}")
        
        # Check concurrent negotiation limit
        if len(self.active_sessions) >= self.config["max_concurrent_negotiations"]:
            raise RuntimeError("Maximum concurrent negotiations reached")
        
        # Get agents
        creditor = self.creditor_agents.get(creditor_id)
        debtor = self.debtor_agents.get(debtor_id)
        
        if not creditor or not debtor:
            raise ValueError("Invalid agent IDs")
        
        # Create session
        session_id = f"NEG_{datetime.now().timestamp()}_{uuid.uuid4().hex[:8]}"
        
        # Assign mediator if needed
        use_mediation = use_mediation if use_mediation is not None else self.config["enable_mediation"]
        mediator = None
        if use_mediation:
            mediator = await self._assign_mediator()
        
        # Get regulatory agent
        regulatory = self._get_regulatory_agent()
        
        # Create session
        session = NegotiationSession(
            session_id=session_id,
            creditor_agent=creditor,
            debtor_agent=debtor,
            mediator_agent=mediator,
            regulatory_agent=regulatory,
            status="active",
            start_time=datetime.now(),
            end_time=None,
            outcome=None,
            compliance_record=None
        )
        
        self.active_sessions[session_id] = session
        
        # Start regulatory monitoring
        audit_record = await regulatory.monitor_negotiation(session_id, creditor, debtor)
        
        # Initialize mediation if applicable
        if mediator:
            await mediator.initiate_mediation(creditor, debtor)
        
        # Start negotiation process
        asyncio.create_task(self._run_negotiation(session_id))
        
        self.performance_metrics["total_negotiations"] += 1
        
        logger.info(f"Initiated negotiation {session_id} between {creditor_id} and {debtor_id}")
        return session_id
    
    async def _run_negotiation(self, session_id: str):
        """Run the negotiation process"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        try:
            creditor = session.creditor_agent
            debtor = session.debtor_agent
            mediator = session.mediator_agent
            regulatory = session.regulatory_agent
            
            # Get debtor profile
            debtor_profile = await debtor.get_profile()
            
            # Generate initial offer
            initial_offer = creditor.generate_initial_offer(debtor_profile)
            
            # Compliance check
            compliance_event = await regulatory.check_offer_compliance(
                session_id, initial_offer, creditor.agent_id
            )
            
            if compliance_event.status == ComplianceStatus.VIOLATION:
                await self._handle_compliance_violation(session_id, compliance_event)
                return
            
            # Send offer through message system
            await self._send_message(Message(
                msg_id=f"MSG_{datetime.now().timestamp()}",
                msg_type=MessageType.OFFER,
                sender_id=creditor.agent_id,
                receiver_id=debtor.agent_id,
                content={"offer": initial_offer, "session_id": session_id},
                timestamp=datetime.now(),
                priority=7
            ))
            
            # Negotiation loop
            rounds = 0
            max_rounds = 20
            
            while rounds < max_rounds and session.status == "active":
                # Process debtor response
                counter_offer = await debtor.receive_offer(initial_offer)
                
                if counter_offer is None:
                    # Offer accepted
                    await self._finalize_negotiation(session_id, "accepted", initial_offer)
                    break
                
                # Compliance check on counter offer
                compliance_event = await regulatory.check_offer_compliance(
                    session_id, counter_offer, debtor.agent_id
                )
                
                if compliance_event.status == ComplianceStatus.VIOLATION:
                    await self._handle_compliance_violation(session_id, compliance_event)
                    break
                
                # Mediation if needed
                if mediator:
                    mediation_result = await mediator.mediate_round(
                        session_id, initial_offer, counter_offer
                    )
                    
                    if mediation_result["type"] == "impasse_intervention":
                        # Use mediator's proposed solution
                        initial_offer = mediation_result["proposed_solution"]
                        continue
                
                # Creditor evaluates counter offer
                accept, new_offer = creditor.evaluate_counter_offer(counter_offer)
                
                if accept:
                    await self._finalize_negotiation(session_id, "accepted", counter_offer)
                    break
                
                if new_offer:
                    initial_offer = new_offer
                else:
                    await self._finalize_negotiation(session_id, "rejected", None)
                    break
                
                rounds += 1
            
            if rounds >= max_rounds:
                await self._finalize_negotiation(session_id, "timeout", None)
            
        except Exception as e:
            logger.error(f"Error in negotiation {session_id}: {str(e)}")
            await self._finalize_negotiation(session_id, "error", None)
    
    async def _finalize_negotiation(self, session_id: str, status: str, final_offer: Optional[Offer]):
        """Finalize a negotiation session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        session.status = status
        session.end_time = datetime.now()
        session.outcome = {
            "status": status,
            "final_offer": final_offer,
            "duration": (session.end_time - session.start_time).total_seconds()
        }
        
        # Generate compliance report
        compliance_report = await session.regulatory_agent.generate_compliance_report(session_id)
        session.compliance_record = compliance_report
        
        # Close regulatory monitoring
        session.regulatory_agent.close_monitoring(session_id)
        
        # Update metrics
        if status == "accepted":
            self.performance_metrics["successful_negotiations"] += 1
            
            # Create debt instrument for market
            if final_offer and self.config["enable_market_making"]:
                await self._create_market_instrument(session, final_offer)
        
        # Archive session
        self.session_history.append(session)
        del self.active_sessions[session_id]
        
        # Update average settlement time
        self._update_settlement_metrics()
        
        logger.info(f"Finalized negotiation {session_id} with status: {status}")
    
    async def _create_market_instrument(self, session: NegotiationSession, offer: Offer):
        """Create tradeable debt instrument from settled negotiation"""
        instrument = DebtInstrument(
            instrument_id=f"DEBT_{session.session_id}",
            original_debt=session.creditor_agent.initial_claim,
            current_value=offer.amount,
            risk_rating=self._calculate_risk_rating(session),
            maturity_date=datetime.now() + timedelta(days=offer.timeline_days),
            debtor_profile=await session.debtor_agent.get_profile(),
            payment_history=[]
        )
        
        # Offer to market makers
        for mm_id, market_maker in self.market_maker_agents.items():
            market_result = await market_maker.create_market(instrument)
            
            if market_result["status"] == "success":
                logger.info(f"Market created for instrument {instrument.instrument_id} by {mm_id}")
                break
    
    def _calculate_risk_rating(self, session: NegotiationSession) -> str:
        """Calculate risk rating for debt instrument"""
        # Simplified risk rating calculation
        debtor = session.debtor_agent
        
        if debtor.financial_profile.credit_score >= 750:
            return "AAA"
        elif debtor.financial_profile.credit_score >= 700:
            return "AA"
        elif debtor.financial_profile.credit_score >= 650:
            return "A"
        elif debtor.financial_profile.credit_score >= 600:
            return "BBB"
        elif debtor.financial_profile.credit_score >= 550:
            return "BB"
        elif debtor.financial_profile.credit_score >= 500:
            return "B"
        else:
            return "C"
    
    async def _assign_mediator(self) -> Optional[MediatorAgent]:
        """Assign an available mediator"""
        # Find mediator with lowest active sessions
        min_sessions = float('inf')
        selected_mediator = None
        
        for mediator_id, mediator in self.mediator_agents.items():
            active_count = len(mediator.active_sessions)
            if active_count < min_sessions:
                min_sessions = active_count
                selected_mediator = mediator
        
        return selected_mediator
    
    def _get_regulatory_agent(self) -> RegulatoryAgent:
        """Get regulatory agent for monitoring"""
        # For now, return the first regulatory agent
        # In production, could load balance
        return list(self.regulatory_agents.values())[0]
    
    async def _register_agent_with_regulatory(self, agent_id: str, agent_type: str):
        """Register agent with regulatory system"""
        for reg_agent in self.regulatory_agents.values():
            # In real system, would register agent details
            logger.debug(f"Registered {agent_type} agent {agent_id} with regulatory system")
    
    async def _send_message(self, message: Message):
        """Send message through the system"""
        await self.message_queue.put(message)
        
        # High priority messages go to priority queue
        if message.priority >= 8:
            self.priority_queue.put((10 - message.priority, message))
    
    async def _message_processor(self):
        """Process messages between agents"""
        while self.system_state == SystemState.ACTIVE:
            try:
                # Check priority queue first
                if not self.priority_queue.empty():
                    _, message = self.priority_queue.get_nowait()
                else:
                    message = await asyncio.wait_for(
                        self.message_queue.get(), 
                        timeout=1.0
                    )
                
                await self._route_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
    
    async def _route_message(self, message: Message):
        """Route message to appropriate handler"""
        logger.debug(f"Routing message {message.msg_id} from {message.sender_id} to {message.receiver_id}")
        
        # Add routing logic based on message type
        if message.msg_type == MessageType.MARKET_ORDER:
            await self._handle_market_order(message)
        elif message.msg_type == MessageType.COMPLIANCE_CHECK:
            await self._handle_compliance_check(message)
        # Add more message type handlers as needed
    
    async def _handle_compliance_violation(self, session_id: str, compliance_event):
        """Handle compliance violations"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        self.performance_metrics["compliance_violations"] += 1
        
        # Log violation
        logger.warning(f"Compliance violation in session {session_id}: {compliance_event.details}")
        
        # Take remediation action
        if compliance_event.remediation_required:
            # Pause negotiation
            session.status = "paused_compliance"
            
            # Notify agents
            for agent_id in [session.creditor_agent.agent_id, session.debtor_agent.agent_id]:
                await self._send_message(Message(
                    msg_id=f"MSG_{datetime.now().timestamp()}",
                    msg_type=MessageType.SYSTEM_UPDATE,
                    sender_id="SYSTEM",
                    receiver_id=agent_id,
                    content={
                        "type": "compliance_violation",
                        "session_id": session_id,
                        "violation": compliance_event.details
                    },
                    timestamp=datetime.now(),
                    priority=9
                ))
    
    async def _market_coordinator(self):
        """Coordinate market making activities"""
        while self.system_state == SystemState.ACTIVE:
            try:
                # Update market conditions
                for market_maker in self.market_maker_agents.values():
                    market_maker.market_condition = market_maker.analyze_market_conditions()
                    
                    # Update ML models
                    if self.config["enable_ml_optimization"]:
                        market_maker.update_market_models()
                    
                    # Check hedging requirements
                    hedge_result = await market_maker.hedge_positions()
                    if hedge_result["hedging_required"]:
                        logger.info(f"Hedging required for {market_maker.agent_id}: {hedge_result['actions']}")
                
                # Aggregate market metrics
                await self._update_market_metrics()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in market coordination: {str(e)}")
    
    async def _update_market_metrics(self):
        """Update system-wide market metrics"""
        total_liquidity = 0
        total_volume = 0
        
        for market_maker in self.market_maker_agents.values():
            metrics = await market_maker.provide_liquidity_metrics()
            total_liquidity += metrics["total_bid_volume"] + metrics["total_ask_volume"]
            total_volume += market_maker.trades_executed
        
        self.market_stats["liquidity"].append({
            "timestamp": datetime.now(),
            "value": total_liquidity
        })
        
        self.market_stats["volume"].append({
            "timestamp": datetime.now(),
            "value": total_volume
        })
    
    async def _compliance_monitor(self):
        """Monitor system-wide compliance"""
        while self.system_state == SystemState.ACTIVE:
            try:
                # Check all active sessions for compliance
                for session_id, session in self.active_sessions.items():
                    if session.status == "active":
                        # Periodic compliance review
                        report = await session.regulatory_agent.generate_compliance_report(session_id)
                        
                        if report["overall_status"] == "NON_COMPLIANT":
                            await self._handle_compliance_violation(session_id, report)
                
                # Generate system compliance report
                if self.config["auto_regulatory_reporting"]:
                    await self._generate_system_compliance_report()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in compliance monitoring: {str(e)}")
    
    async def _generate_system_compliance_report(self):
        """Generate system-wide compliance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "active_negotiations": len(self.active_sessions),
            "compliance_violations": self.performance_metrics["compliance_violations"],
            "average_compliance_score": 0.0,
            "high_risk_sessions": []
        }
        
        # Calculate average compliance score
        scores = []
        for session in self.active_sessions.values():
            if hasattr(session, 'compliance_record') and session.compliance_record:
                scores.append(session.compliance_record.get("compliance_score", 1.0))
        
        if scores:
            report["average_compliance_score"] = np.mean(scores)
        
        logger.info(f"System compliance report: {report}")
    
    async def _performance_monitor(self):
        """Monitor system performance"""
        while self.system_state == SystemState.ACTIVE:
            try:
                # Log performance metrics
                uptime = (datetime.now() - self.performance_metrics["system_uptime"]).total_seconds()
                
                logger.info(f"""
                System Performance Metrics:
                - Uptime: {uptime / 3600:.2f} hours
                - Total Negotiations: {self.performance_metrics['total_negotiations']}
                - Success Rate: {self.performance_metrics['successful_negotiations'] / max(self.performance_metrics['total_negotiations'], 1) * 100:.2f}%
                - Active Sessions: {len(self.active_sessions)}
                - Compliance Violations: {self.performance_metrics['compliance_violations']}
                """)
                
                await asyncio.sleep(600)  # Log every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
    
    def _update_settlement_metrics(self):
        """Update average settlement time"""
        if self.session_history:
            settlement_times = [
                s.outcome["duration"] 
                for s in self.session_history 
                if s.outcome and "duration" in s.outcome
            ]
            
            if settlement_times:
                self.performance_metrics["average_settlement_time"] = np.mean(settlement_times)
    
    async def submit_market_order(self, order: MarketOrder) -> Dict:
        """Submit order to market"""
        # Find best market maker to handle order
        best_mm = None
        best_price = float('inf') if order.order_type == "buy" else 0
        
        for mm_id, market_maker in self.market_maker_agents.items():
            evaluation = await market_maker.evaluate_debt_instrument(order.instrument)
            
            if order.order_type == "buy" and evaluation["market_value"] < best_price:
                best_price = evaluation["market_value"]
                best_mm = market_maker
            elif order.order_type == "sell" and evaluation["market_value"] > best_price:
                best_price = evaluation["market_value"]
                best_mm = market_maker
        
        if best_mm:
            trade_result = await best_mm.execute_trade(order)
            return trade_result or {"status": "no_match"}
        
        return {"status": "no_market_maker"}
    
    async def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "system_state": self.system_state.value,
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(self.active_sessions),
            "total_agents": {
                "creditors": len(self.creditor_agents),
                "debtors": len(self.debtor_agents),
                "mediators": len(self.mediator_agents),
                "market_makers": len(self.market_maker_agents),
                "regulatory": len(self.regulatory_agents)
            },
            "performance_metrics": self.performance_metrics,
            "market_stats": {
                "latest_liquidity": self.market_stats["liquidity"][-1] if self.market_stats["liquidity"] else None,
                "latest_volume": self.market_stats["volume"][-1] if self.market_stats["volume"] else None
            },
            "config": self.config
        }
    
    async def emergency_shutdown(self, reason: str):
        """Emergency system shutdown"""
        logger.critical(f"Emergency shutdown initiated: {reason}")
        self.system_state = SystemState.EMERGENCY
        
        # Pause all active negotiations
        for session_id, session in self.active_sessions.items():
            session.status = "emergency_paused"
            
            # Notify all agents
            await self._send_message(Message(
                msg_id=f"MSG_{datetime.now().timestamp()}",
                msg_type=MessageType.SYSTEM_UPDATE,
                sender_id="SYSTEM",
                receiver_id="ALL",
                content={
                    "type": "emergency_shutdown",
                    "reason": reason,
                    "session_id": session_id
                },
                timestamp=datetime.now(),
                priority=10
            ))
        
        # Save state for recovery
        await self._save_system_state()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.critical("Emergency shutdown complete")
    
    async def _save_system_state(self):
        """Save system state for recovery"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "active_sessions": {
                sid: {
                    "creditor_id": s.creditor_agent.agent_id,
                    "debtor_id": s.debtor_agent.agent_id,
                    "status": s.status,
                    "start_time": s.start_time.isoformat()
                } for sid, s in self.active_sessions.items()
            },
            "performance_metrics": self.performance_metrics,
            "market_stats": self.market_stats
        }
        
        # In production, would save to persistent storage
        with open("system_state_backup.json", "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info("System state saved")
    
    async def maintenance_mode(self, duration_minutes: int):
        """Enter maintenance mode"""
        self.system_state = SystemState.MAINTENANCE
        logger.info(f"Entering maintenance mode for {duration_minutes} minutes")
        
        # Notify all active sessions
        for session_id in self.active_sessions:
            await self._send_message(Message(
                msg_id=f"MSG_{datetime.now().timestamp()}",
                msg_type=MessageType.SYSTEM_UPDATE,
                sender_id="SYSTEM",
                receiver_id="ALL",
                content={
                    "type": "maintenance_mode",
                    "duration_minutes": duration_minutes,
                    "session_id": session_id
                },
                timestamp=datetime.now(),
                priority=8
            ))
        
        # Schedule return to active state
        asyncio.create_task(self._maintenance_timer(duration_minutes))
    
    async def _maintenance_timer(self, duration_minutes: int):
        """Timer for maintenance mode"""
        await asyncio.sleep(duration_minutes * 60)
        self.system_state = SystemState.ACTIVE
        logger.info("Maintenance mode ended, system active")


# Example usage
async def main():
    """Example of using the orchestrator"""
    # Create orchestrator
    orchestrator = PaymentNegotiationOrchestrator()
    
    # Start system
    await orchestrator.start_system()
    
    # Create agents
    creditor = await orchestrator.create_creditor_agent("CRED_001", 50000)
    
    financial_profile = FinancialProfile(
        monthly_income=5000,
        monthly_obligations=3000,
        liquid_assets=10000,
        credit_score=650,
        debt_to_income=0.4,
        employment_status="stable",
        hardship_factors=["medical_expenses"]
    )
    
    debtor = await orchestrator.create_debtor_agent("DEBT_001", 50000, financial_profile)
    
    # Initiate negotiation
    session_id = await orchestrator.initiate_negotiation("CRED_001", "DEBT_001", use_mediation=True)
    
    print(f"Started negotiation session: {session_id}")
    
    # Wait for negotiation to complete
    await asyncio.sleep(10)
    
    # Get system status
    status = await orchestrator.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())