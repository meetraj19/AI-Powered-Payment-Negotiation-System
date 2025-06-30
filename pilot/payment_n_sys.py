#!/usr/bin/env python3
"""
CrewAI-based Payment Negotiation Multi-Agent System
Integrates creditor, debtor, mediator, market maker, regulatory agents
and adds PDF report generation capabilities
"""

import os
import sys
import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CrewAI imports
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool
except ImportError:
    logger.error("Please install crewai: pip install crewai")
    sys.exit(1)

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    logger.error("Please install langchain-openai: pip install langchain-openai")
    sys.exit(1)

# PDF generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
except ImportError:
    logger.error("Please install reportlab: pip install reportlab")
    sys.exit(1)

# Import all agent modules (assuming they're in the same directory)
try:
    from creditor import CreditorAgent, NegotiationStrategy, Offer
    from debitor import DebtorAgent, FinancialProfile, FinancialSituation
    from mediator import MediatorAgent, MediationStrategy
    from market_maker import MarketMakerAgent, DebtInstrument, MarketOrder
    from regulatory import RegulatoryAgent, ComplianceStatus
except ImportError as e:
    logger.error(f"Failed to import agent modules: {e}")
    logger.error("Make sure all agent module files are in the same directory")
    sys.exit(1)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenAIConfig:
    """OpenAI configuration management"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    def create_llm(self):
        """Create and return OpenAI LLM instance"""
        return ChatOpenAI(
            openai_api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=60,
            max_retries=3
        )

# Initialize OpenAI
openai_config = OpenAIConfig()
llm = openai_config.create_llm()

@dataclass
class NegotiationContext:
    """Shared context for the negotiation"""
    creditor_id: str
    debtor_id: str
    initial_debt: float
    financial_profile: FinancialProfile
    negotiation_history: List[Dict] = None
    current_offer: Optional[Offer] = None
    compliance_status: str = "compliant"
    mediation_active: bool = False
    final_agreement: Optional[Dict] = None
    
    def __post_init__(self):
        if self.negotiation_history is None:
            self.negotiation_history = []

# Tools for agents to interact with the original agent classes
@tool("creditor_negotiation_tool")
def creditor_negotiation_tool(action: str, **kwargs) -> str:
    """Tool for creditor agent to perform negotiation actions
    
    Args:
        action: The action to perform ('generate_initial_offer' or 'evaluate_counter_offer')
        For 'generate_initial_offer': creditor_id, initial_debt, debtor_profile
        For 'evaluate_counter_offer': counter_offer details
    """
    try:
        if action == "generate_initial_offer":
            # Extract parameters with defaults
            creditor_id = kwargs.get('creditor_id', 'CRED_001')
            initial_debt = float(kwargs.get('initial_debt', 50000))
            
            # Create creditor agent
            creditor = CreditorAgent(creditor_id, initial_debt)
            
            # Build debtor profile from various possible parameter names
            debtor_profile = kwargs.get('debtor_profile', {})
            if not debtor_profile:
                # Build profile from individual parameters
                debtor_profile = {
                    'credit_score': int(kwargs.get('credit_score', 650)),
                    'monthly_income': float(kwargs.get('monthly_income', 5000)),
                    'monthly_obligations': float(kwargs.get('monthly_obligations', kwargs.get('monthly_expenses', 3000))),
                    'debt_to_income': float(kwargs.get('debt_to_income', 0.4)),
                    'employment_status': kwargs.get('employment_status', 'stable'),
                    'financial_situation': kwargs.get('financial_situation', 'moderate')
                }
            
            offer = creditor.generate_initial_offer(debtor_profile)
            return json.dumps({
                "success": True,
                "offer": {
                    "amount": offer.amount,
                    "interest_rate": offer.interest_rate,
                    "timeline_days": offer.timeline_days,
                    "payment_terms": offer.payment_terms
                }
            })
        
        elif action == "evaluate_counter_offer":
            # Extract counter offer details
            counter_offer_data = {
                'amount': float(kwargs.get('amount', 0)),
                'interest_rate': float(kwargs.get('interest_rate', 0.05)),
                'timeline_days': int(kwargs.get('timeline_days', 180)),
                'payment_terms': kwargs.get('payment_terms', {})
            }
            
            counter_offer = Offer(**counter_offer_data)
            
            # Need creditor agent - get from context or create new
            creditor_id = kwargs.get('creditor_id', 'CRED_001')
            initial_debt = float(kwargs.get('initial_debt', counter_offer.amount / 0.6))
            creditor = CreditorAgent(creditor_id, initial_debt)
            
            accept, new_offer = creditor.evaluate_counter_offer(counter_offer)
            result = {"success": True, "accept": accept}
            if new_offer:
                result["new_offer"] = {
                    "amount": new_offer.amount,
                    "interest_rate": new_offer.interest_rate,
                    "timeline_days": new_offer.timeline_days
                }
            return json.dumps(result)
        
        else:
            return json.dumps({"success": False, "error": f"Unknown action: {action}"})
            
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "action": action, "kwargs": str(kwargs)})

@tool("debtor_negotiation_tool") 
def debtor_negotiation_tool(action: str, **kwargs) -> str:
    """Tool for debtor agent to respond to offers
    
    Args:
        action: The action to perform ('evaluate_offer')
        For 'evaluate_offer': offer details and financial profile
    """
    try:
        if action == "evaluate_offer":
            # Extract parameters
            debtor_id = kwargs.get('debtor_id', 'DEBT_001')
            initial_debt = float(kwargs.get('initial_debt', 50000))
            
            # Build financial profile - handle different parameter names
            profile_data = {
                'monthly_income': float(kwargs.get('monthly_income', 5000)),
                'monthly_obligations': float(kwargs.get('monthly_obligations', kwargs.get('monthly_expenses', 3000))),
                'liquid_assets': float(kwargs.get('liquid_assets', 10000)),
                'credit_score': int(kwargs.get('credit_score', 650)),
                'debt_to_income': float(kwargs.get('debt_to_income', 0.4)),
                'employment_status': kwargs.get('employment_status', 'stable'),
                'hardship_factors': kwargs.get('hardship_factors', [])
            }
            
            # Create financial profile
            financial_profile = FinancialProfile(**profile_data)
            
            # Create debtor agent
            debtor = DebtorAgent(debtor_id, initial_debt, financial_profile)
            
            # Extract offer details
            offer_data = kwargs.get('offer', {})
            if not isinstance(offer_data, dict):
                # If offer is passed directly as parameters
                offer_data = {
                    'amount': float(kwargs.get('amount', initial_debt * 0.7)),
                    'interest_rate': float(kwargs.get('interest_rate', 0.05)),
                    'timeline_days': int(kwargs.get('timeline_days', 180)),
                    'payment_terms': kwargs.get('payment_terms', {'type': 'installment'})
                }
            
            offer = Offer(**offer_data)
            
            # Using asyncio to handle async method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            counter_offer = loop.run_until_complete(debtor.receive_offer(offer))
            
            if counter_offer is None:
                return json.dumps({"success": True, "accept": True})
            else:
                return json.dumps({
                    "success": True,
                    "accept": False,
                    "counter_offer": {
                        "amount": counter_offer.amount,
                        "interest_rate": counter_offer.interest_rate,
                        "timeline_days": counter_offer.timeline_days,
                        "payment_terms": counter_offer.payment_terms
                    }
                })
        
        else:
            return json.dumps({"success": False, "error": f"Unknown action: {action}"})
            
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "action": action, "kwargs": str(kwargs)})

@tool("mediation_tool")
def mediation_tool(action: str, **kwargs) -> str:
    """Tool for mediator to facilitate negotiation
    
    Args:
        action: The action to perform ('analyze_impasse')
        negotiation_history: List of negotiation rounds
    """
    try:
        mediator = MediatorAgent("MEDIATOR_001")
        
        if action == "analyze_impasse":
            # Get negotiation history
            history = kwargs.get('negotiation_history', [])
            
            if len(history) >= 3:
                # Check if negotiation is stuck
                recent_offers = history[-3:]
                amounts = []
                
                for h in recent_offers:
                    if isinstance(h, dict) and 'offer' in h and 'amount' in h['offer']:
                        amounts.append(float(h['offer']['amount']))
                    elif isinstance(h, dict) and 'amount' in h:
                        amounts.append(float(h['amount']))
                
                if amounts and (max(amounts) - min(amounts)) < 0.01 * amounts[0]:
                    avg_amount = sum(amounts) / len(amounts)
                    return json.dumps({
                        "success": True,
                        "impasse_detected": True,
                        "suggestion": "Consider alternative payment structures or timeline adjustments",
                        "recommended_amount": avg_amount
                    })
            
            return json.dumps({"success": True, "impasse_detected": False})
        
        else:
            return json.dumps({"success": False, "error": f"Unknown action: {action}"})
            
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

@tool("compliance_check_tool")
def compliance_check_tool(action: str, **kwargs) -> str:
    """Tool for regulatory compliance checking
    
    Args:
        action: The action to perform ('check_offer')
        offer: The offer details to check
    """
    try:
        regulatory = RegulatoryAgent("REG_001")
        
        if action == "check_offer":
            # Extract offer details
            offer = kwargs.get('offer', {})
            if not isinstance(offer, dict):
                offer = {
                    'amount': float(kwargs.get('amount', 0)),
                    'interest_rate': float(kwargs.get('interest_rate', 0.05)),
                    'timeline_days': int(kwargs.get('timeline_days', 180)),
                    'payment_terms': kwargs.get('payment_terms', {})
                }
            
            violations = []
            
            # Check interest rate compliance
            interest_rate = float(offer.get('interest_rate', 0))
            if interest_rate > 0.25:
                violations.append("Interest rate exceeds 25% APR maximum")
            
            # Check for required disclosures
            if 'payment_terms' not in offer or not offer['payment_terms']:
                violations.append("Missing required payment terms disclosure")
            
            # Check settlement amount reasonableness
            if 'amount' in offer and 'initial_debt' in kwargs:
                settlement_ratio = float(offer['amount']) / float(kwargs['initial_debt'])
                if settlement_ratio > 0.95:
                    violations.append("Initial settlement demand may be considered excessive")
            
            if violations:
                return json.dumps({
                    "success": True,
                    "compliant": False,
                    "violations": violations,
                    "remediation": "Address violations before proceeding"
                })
            
            return json.dumps({
                "success": True, 
                "compliant": True, 
                "violations": []
            })
        
        else:
            return json.dumps({"success": False, "error": f"Unknown action: {action}"})
            
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

@tool("market_analysis_tool")
def market_analysis_tool(action: str, **kwargs) -> str:
    """Tool for market maker to analyze debt instruments
    
    Args:
        action: The action to perform ('evaluate_settlement')
        settlement: The settlement details
        financial_profile: The debtor's financial profile
    """
    try:
        market_maker = MarketMakerAgent("MM_001", 1000000)
        
        if action == "evaluate_settlement":
            # Extract settlement details
            settlement = kwargs.get('settlement', {})
            if not isinstance(settlement, dict):
                settlement = {
                    'amount': float(kwargs.get('amount', 0)),
                    'interest_rate': float(kwargs.get('interest_rate', 0.05)),
                    'timeline_days': int(kwargs.get('timeline_days', 180))
                }
            
            # Extract financial profile
            financial_profile = kwargs.get('financial_profile', {})
            credit_score = int(financial_profile.get('credit_score', kwargs.get('credit_score', 650)))
            
            # Calculate fair market value
            risk_premium = max(0.02, (850 - credit_score) / 1000)
            fair_value = float(settlement.get('amount', 0)) * (1 - risk_premium)
            
            # Determine risk rating
            if credit_score >= 750:
                risk_rating = "AAA"
            elif credit_score >= 700:
                risk_rating = "AA"
            elif credit_score >= 650:
                risk_rating = "A"
            elif credit_score >= 600:
                risk_rating = "BBB"
            else:
                risk_rating = "BB"
            
            return json.dumps({
                "success": True,
                "fair_market_value": fair_value,
                "risk_rating": risk_rating,
                "liquidity_score": 0.7,
                "tradeable": True
            })
        
        else:
            return json.dumps({"success": False, "error": f"Unknown action: {action}"})
            
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

# Define CrewAI Agents
creditor_agent = Agent(
    role='Senior Creditor Representative',
    goal='Negotiate optimal debt recovery while maintaining compliance and fairness',
    backstory="""You are an experienced creditor representative with 15 years in debt negotiation.
    You understand the importance of recovering funds while treating debtors fairly and maintaining
    long-term relationships. You use data-driven approaches and adaptive strategies.""",
    tools=[creditor_negotiation_tool],
    llm=llm,
    verbose=True
)

debtor_agent = Agent(
    role='Debtor Advocate',
    goal='Negotiate fair and affordable settlement terms based on financial capacity',
    backstory="""You represent debtors in financial distress, ensuring they get fair treatment
    and payment terms they can actually meet. You analyze financial situations comprehensively
    and advocate for sustainable payment plans.""",
    tools=[debtor_negotiation_tool],
    llm=llm,
    verbose=True
)

mediator_agent = Agent(
    role='Professional Mediator',
    goal='Facilitate productive negotiations and help parties reach mutually beneficial agreements',
    backstory="""You are a certified mediator specializing in financial disputes. You excel at
    identifying common ground, breaking impasses, and suggesting creative solutions that satisfy
    both parties' core interests.""",
    tools=[mediation_tool],
    llm=llm,
    verbose=True
)

regulatory_agent = Agent(
    role='Compliance Officer',
    goal='Ensure all negotiations comply with relevant laws and regulations',
    backstory="""You are a regulatory compliance expert ensuring all negotiations follow FDCPA,
    TCPA, GDPR, and other relevant regulations. You prevent violations and protect both parties
    from legal risks.""",
    tools=[compliance_check_tool],
    llm=llm,
    verbose=True
)

market_maker_agent = Agent(
    role='Market Analyst',
    goal='Evaluate settlements for market tradability and provide liquidity analysis',
    backstory="""You analyze settled debts for their market value and tradability. You understand
    risk assessment, market conditions, and help create liquid markets for debt instruments.""",
    tools=[market_analysis_tool],
    llm=llm,
    verbose=True
)

# PDF Report Generator Agent
pdf_generator_agent = Agent(
    role='Report Generator Specialist',
    goal='Create comprehensive PDF reports documenting negotiation outcomes and compliance',
    backstory="""You are an expert in creating detailed, professional reports that document
    complex negotiations. You ensure all important details, compliance records, and agreements
    are clearly presented in an accessible format.""",
    llm=llm,
    verbose=True
)

class PDFReportGenerator:
    """Generate comprehensive PDF reports for negotiations"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12
        ))
    
    def generate_negotiation_report(self, context: NegotiationContext, 
                                   negotiation_result: Dict,
                                   crew_output: str = None,
                                   filename: str = "negotiation_report.pdf"):
        """Generate comprehensive PDF report"""
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []
        
        # Title Page
        story.append(Paragraph("Payment Negotiation Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Report metadata
        metadata_text = f"""
        <para align="center">
        <font size="12">
        Creditor: {context.creditor_id} | Debtor: {context.debtor_id}<br/>
        Original Debt: ${context.initial_debt:,.2f}<br/>
        Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </font>
        </para>
        """
        story.append(Paragraph(metadata_text, self.styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        # If we have crew output, include it as the main content
        if crew_output:
            story.append(Paragraph("Negotiation Process Summary", self.styles['SectionHeader']))
            story.append(Spacer(1, 0.2*inch))
            
            # Clean and format the crew output
            crew_text = crew_output.replace('**', '')  # Remove markdown bold
            crew_text = crew_text.replace('---', '')   # Remove dividers
            
            # Clean up any HTML entities
            crew_text = crew_text.replace('&lt;', '<')
            crew_text = crew_text.replace('&gt;', '>')
            crew_text = crew_text.replace('&amp;', '&')
            crew_text = crew_text.replace('&nbsp;', ' ')
            
            # Split into sections
            sections = crew_text.split('\n\n')
            for section in sections:
                if section.strip():
                    # Clean the section text
                    section_text = section.strip()
                    
                    # Check if it's a header (starts with number)
                    if section_text and section_text[0].isdigit() and '.' in section_text[:3]:
                        # It's a numbered section header
                        story.append(Paragraph(section_text, self.styles['SectionHeader']))
                    else:
                        # Regular paragraph - escape any special characters for ReportLab
                        section_text = section_text.replace('<', '&lt;')
                        section_text = section_text.replace('>', '&gt;')
                        section_text = section_text.replace('&', '&amp;')
                        story.append(Paragraph(section_text, self.styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            
            story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        summary_data = [
            ["Creditor ID", context.creditor_id],
            ["Debtor ID", context.debtor_id],
            ["Original Debt", f"${context.initial_debt:,.2f}"],
            ["Settlement Amount", f"${negotiation_result.get('final_amount', 0):,.2f}"],
            ["Settlement Percentage", f"{(negotiation_result.get('final_amount', 0)/context.initial_debt*100):.1f}%"],
            ["Negotiation Status", negotiation_result.get('status', 'Unknown')]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Debtor Financial Profile
        story.append(Paragraph("Debtor Financial Profile", self.styles['SectionHeader']))
        profile = context.financial_profile
        profile_data = [
            ["Monthly Income", f"${profile.monthly_income:,.2f}"],
            ["Monthly Obligations", f"${profile.monthly_obligations:,.2f}"],
            ["Disposable Income", f"${(profile.monthly_income - profile.monthly_obligations):,.2f}"],
            ["Credit Score", str(profile.credit_score)],
            ["Debt-to-Income Ratio", f"{profile.debt_to_income:.1%}"],
            ["Employment Status", profile.employment_status],
            ["Liquid Assets", f"${profile.liquid_assets:,.2f}"],
            ["Hardship Factors", ", ".join(profile.hardship_factors) if profile.hardship_factors else "None"]
        ]
        
        profile_table = Table(profile_data, colWidths=[3*inch, 3*inch])
        profile_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold')
        ]))
        story.append(profile_table)
        story.append(PageBreak())
        
        # Negotiation History
        story.append(Paragraph("Negotiation History", self.styles['SectionHeader']))
        if context.negotiation_history:
            history_data = [["Round", "Creditor Offer", "Debtor Counter", "Status"]]
            for i, record in enumerate(context.negotiation_history):
                history_data.append([
                    str(i + 1),
                    f"${record.get('creditor_offer', {}).get('amount', 0):,.2f}",
                    f"${record.get('debtor_counter', {}).get('amount', 0):,.2f}" if record.get('debtor_counter') else "N/A",
                    record.get('status', 'Pending')
                ])
            
            history_table = Table(history_data, colWidths=[1*inch, 2*inch, 2*inch, 1.5*inch])
            history_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(history_table)
        
        story.append(Spacer(1, 0.3*inch))
        
        # Final Agreement Details
        if negotiation_result.get('status') == 'success':
            story.append(Paragraph("Final Agreement Details", self.styles['SectionHeader']))
            agreement_text = f"""
            The parties have reached a settlement agreement with the following terms:
            <br/><br/>
            <b>Settlement Amount:</b> ${negotiation_result.get('final_amount', 0):,.2f}<br/>
            <b>Payment Timeline:</b> {negotiation_result.get('timeline_days', 0)} days<br/>
            <b>Interest Rate:</b> {negotiation_result.get('interest_rate', 0):.2%}<br/>
            <b>Payment Structure:</b> {negotiation_result.get('payment_type', 'Installment')}<br/>
            """
            story.append(Paragraph(agreement_text, self.styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Compliance Summary
        story.append(Paragraph("Compliance Summary", self.styles['SectionHeader']))
        compliance_text = f"""
        <b>Overall Compliance Status:</b> {negotiation_result.get('compliance_status', 'Compliant')}<br/>
        <b>Violations Detected:</b> {negotiation_result.get('violations_count', 0)}<br/>
        <b>Warnings Issued:</b> {negotiation_result.get('warnings_count', 0)}<br/>
        <b>Regulatory Frameworks Applied:</b> FDCPA, TCPA, GDPR, FCRA<br/>
        """
        story.append(Paragraph(compliance_text, self.styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Market Analysis
        if negotiation_result.get('market_analysis'):
            story.append(Paragraph("Market Analysis", self.styles['SectionHeader']))
            market = negotiation_result['market_analysis']
            market_text = f"""
            <b>Fair Market Value:</b> ${market.get('fair_market_value', 0):,.2f}<br/>
            <b>Risk Rating:</b> {market.get('risk_rating', 'N/A')}<br/>
            <b>Liquidity Score:</b> {market.get('liquidity_score', 0):.2f}<br/>
            <b>Tradeable Status:</b> {'Yes' if market.get('tradeable') else 'No'}<br/>
            """
            story.append(Paragraph(market_text, self.styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 1*inch))
        footer_text = f"""
        <para align="center">
        <font size="10">
        Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        This report is confidential and intended for authorized parties only.
        </font>
        </para>
        """
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return filename

# Define Tasks
def create_negotiation_task(context: NegotiationContext):
    return Task(
        description=f"""
        Initiate negotiation between creditor {context.creditor_id} and debtor {context.debtor_id}.
        Original debt amount: ${context.initial_debt}
        
        Generate an initial offer considering the debtor's financial profile.
        
        Use the creditor_negotiation_tool with:
        - action: 'generate_initial_offer'
        - creditor_id: '{context.creditor_id}'
        - initial_debt: {context.initial_debt}
        - credit_score: {context.financial_profile.credit_score}
        - monthly_income: {context.financial_profile.monthly_income}
        - monthly_obligations: {context.financial_profile.monthly_obligations}
        - debt_to_income: {context.financial_profile.debt_to_income}
        - employment_status: '{context.financial_profile.employment_status}'
        
        Return the offer details including amount, interest_rate, timeline_days, and payment_terms.
        """,
        agent=creditor_agent,
        expected_output="Initial settlement offer with amount, interest rate, and payment terms"
    )

def create_evaluation_task(context: NegotiationContext):
    return Task(
        description=f"""
        Evaluate the current offer from the creditor and determine if it's acceptable
        based on the debtor's financial capacity. If not acceptable, generate a counter-offer.
        
        Use the debtor_negotiation_tool with:
        - action: 'evaluate_offer'
        - debtor_id: '{context.debtor_id}'
        - initial_debt: {context.initial_debt}
        - monthly_income: {context.financial_profile.monthly_income}
        - monthly_obligations: {context.financial_profile.monthly_obligations}
        - liquid_assets: {context.financial_profile.liquid_assets}
        - credit_score: {context.financial_profile.credit_score}
        - debt_to_income: {context.financial_profile.debt_to_income}
        - employment_status: '{context.financial_profile.employment_status}'
        - hardship_factors: {context.financial_profile.hardship_factors}
        
        The offer to evaluate should include amount, interest_rate, timeline_days, and payment_terms.
        
        Return accept/reject decision and counter-offer details if applicable.
        """,
        agent=debtor_agent,
        expected_output="Accept/reject decision and counter-offer if applicable"
    )

def create_mediation_task(context: NegotiationContext):
    return Task(
        description="""
        Monitor the negotiation progress and intervene if an impasse is detected.
        Analyze the negotiation history and suggest creative solutions if parties are stuck.
        
        Use the mediation_tool with:
        - action: 'analyze_impasse'
        - negotiation_history: The list of all offers and counter-offers made so far
        
        Look for patterns where offers are not converging and suggest breakthrough solutions.
        """,
        agent=mediator_agent,
        expected_output="Mediation analysis and recommendations if needed"
    )

def create_compliance_task(context: NegotiationContext):
    return Task(
        description=f"""
        Review all offers and ensure compliance with relevant regulations including
        FDCPA, TCPA, GDPR, and usury laws. Check interest rates, required disclosures,
        and fair treatment standards.
        
        Use the compliance_check_tool with:
        - action: 'check_offer'
        - initial_debt: {context.initial_debt}
        
        Check each offer for:
        - Interest rate compliance (max 25% APR)
        - Required disclosures (payment terms must be included)
        - Fair settlement practices
        
        Return compliance status and any violations found.
        """,
        agent=regulatory_agent,
        expected_output="Compliance assessment with any violations or warnings"
    )

def create_market_analysis_task(context: NegotiationContext):
    return Task(
        description=f"""
        Analyze the final settlement for market tradability. Evaluate the debt instrument's
        fair market value, risk rating, and liquidity potential.
        
        Use the market_analysis_tool with:
        - action: 'evaluate_settlement'
        - credit_score: {context.financial_profile.credit_score}
        
        The settlement details should include amount, interest_rate, and timeline_days.
        
        Return market analysis including fair value, risk rating, and tradability assessment.
        """,
        agent=market_maker_agent,
        expected_output="Market analysis including fair value and tradability assessment"
    )

def create_report_generation_task(context: NegotiationContext):
    return Task(
        description=f"""
        Generate a comprehensive PDF report documenting the entire negotiation process.
        
        Compile all the information from the previous tasks and create a detailed report that includes:
        
        1. Executive summary with final settlement details
        2. Debtor financial profile analysis  
        3. Complete negotiation history
        4. Compliance assessment summary
        5. Market analysis results
        6. Final agreement terms
        
        The report should incorporate all the findings from:
        - The creditor's initial offer and subsequent negotiations
        - The debtor's responses and counter-offers
        - The mediator's interventions and recommendations
        - The compliance officer's assessments and violations
        - The market analyst's valuation and risk ratings
        
        Present the information in a clear, professional format suitable for all stakeholders.
        Include specific details like:
        - Settlement amount: The final agreed amount
        - Interest rate: The agreed interest rate
        - Payment timeline: Number of days for repayment
        - Payment structure: Installment details
        - Compliance status: Any violations or warnings
        - Market assessment: Risk rating, liquidity score, fair market value
        
        The report should be comprehensive and include all relevant details from the negotiation process.
        """,
        agent=pdf_generator_agent,
        expected_output="Comprehensive PDF report documenting the negotiation with all details from the crew's analysis"
    )

def parse_crew_output(crew_output: str) -> Dict[str, Any]:
    """
    Parse the crew output to extract negotiation details
    """
    import re
    
    # Initialize result dictionary
    result = {
        'final_amount': 0,
        'interest_rate': 0,
        'timeline_days': 0,
        'payment_type': 'Installment',
        'installment_amount': 0,
        'num_installments': 0,
        'compliance_status': 'Unknown',
        'violations': [],
        'risk_rating': 'Unknown',
        'liquidity_score': 0,
        'fair_market_value': 0,
        'tradeable': False
    }
    
    try:
        # Extract settlement amount - handle both "settlement amount of" and "Settlement Amount:"
        amount_patterns = [
            r'settlement amount of \$?([\d,]+\.?\d*)',
            r'Settlement Amount:\s*\$?([\d,]+\.?\d*)',
            r'Final settlement:\s*\$?([\d,]+\.?\d*)'
        ]
        for pattern in amount_patterns:
            amount_match = re.search(pattern, crew_output, re.IGNORECASE)
            if amount_match:
                result['final_amount'] = float(amount_match.group(1).replace(',', ''))
                break
        
        # Extract interest rate - handle both "interest rate of X%" and "Interest Rate: X%"
        interest_patterns = [
            r'interest rate of (\d+\.?\d*)%',
            r'Interest Rate:\s*(\d+\.?\d*)%',
            r'interest rate:\s*(\d+\.?\d*)%'
        ]
        for pattern in interest_patterns:
            interest_match = re.search(pattern, crew_output, re.IGNORECASE)
            if interest_match:
                result['interest_rate'] = float(interest_match.group(1)) / 100
                break
        
        # Extract timeline
        timeline_patterns = [
            r'timeline of (\d+) days',
            r'over (\d+) days',
            r'timeline:\s*(\d+) days',
            r'Payment Timeline:\s*(\d+) days'
        ]
        for pattern in timeline_patterns:
            timeline_match = re.search(pattern, crew_output, re.IGNORECASE)
            if timeline_match:
                result['timeline_days'] = int(timeline_match.group(1))
                break
        
        # Extract installment details
        installment_patterns = [
            r'(\d+) installments of \$?([\d,]+\.?\d*)',
            r'includes (\d+) installments of \$?([\d,]+\.?\d*)',
            r'(\d+) payments of \$?([\d,]+\.?\d*)'
        ]
        for pattern in installment_patterns:
            installment_match = re.search(pattern, crew_output, re.IGNORECASE)
            if installment_match:
                result['num_installments'] = int(installment_match.group(1))
                result['installment_amount'] = float(installment_match.group(2).replace(',', ''))
                break
        
        # Extract compliance status
        if 'not fully compliant' in crew_output.lower() or 'non-compliant' in crew_output.lower():
            result['compliance_status'] = 'Non-Compliant'
        elif 'compliant' in crew_output.lower():
            result['compliance_status'] = 'Compliant'
            
        # Extract specific violations
        if 'missing required payment terms disclosure' in crew_output.lower():
            result['violations'].append('Missing payment terms disclosure')
        if 'missing payment terms' in crew_output.lower():
            result['violations'].append('Missing payment terms disclosure')
        
        # Extract risk rating - handle various formats
        risk_patterns = [
            r"risk rating of '([A-Z]+)'",
            r'Risk Rating:\s*([A-Z]+)',
            r'rating of ([A-Z]+)',
            r"rated '([A-Z]+)'"
        ]
        for pattern in risk_patterns:
            risk_match = re.search(pattern, crew_output, re.IGNORECASE)
            if risk_match:
                result['risk_rating'] = risk_match.group(1).upper()
                break
        
        # Extract liquidity score
        liquidity_patterns = [
            r'liquidity score (?:is|of) ([\d.]+)',
            r'Liquidity Score:\s*([\d.]+)',
            r'liquidity:\s*([\d.]+)'
        ]
        for pattern in liquidity_patterns:
            liquidity_match = re.search(pattern, crew_output, re.IGNORECASE)
            if liquidity_match:
                result['liquidity_score'] = float(liquidity_match.group(1))
                break
        
        # Extract fair market value
        market_value_patterns = [
            r'fair market value of \$?([\d,]+\.?\d*)',
            r'Fair Market Value:\s*\$?([\d,]+\.?\d*)',
            r'market value:\s*\$?([\d,]+\.?\d*)'
        ]
        for pattern in market_value_patterns:
            market_value_match = re.search(pattern, crew_output, re.IGNORECASE)
            if market_value_match:
                result['fair_market_value'] = float(market_value_match.group(1).replace(',', ''))
                break
        
        # Check if tradeable
        if 'tradeable' in crew_output.lower():
            # Check if it's preceded by "not" or "non"
            tradeable_context = crew_output.lower()[max(0, crew_output.lower().index('tradeable')-20):crew_output.lower().index('tradeable')]
            if 'not' not in tradeable_context and 'non' not in tradeable_context:
                result['tradeable'] = True
            
    except Exception as e:
        logger.error(f"Error parsing crew output: {e}")
    
    return result

# Main execution function
def run_payment_negotiation(
    creditor_id: str,
    debtor_id: str,
    initial_debt: float,
    financial_profile: FinancialProfile,
    max_rounds: int = 10
) -> Dict[str, Any]:
    """
    Run a complete payment negotiation using CrewAI
    """
    try:
        # Initialize context
        context = NegotiationContext(
            creditor_id=creditor_id,
            debtor_id=debtor_id,
            initial_debt=initial_debt,
            financial_profile=financial_profile
        )
        
        logger.info(f"Starting negotiation: {creditor_id} vs {debtor_id} for ${initial_debt:,.2f}")
        
        # Create crew for negotiation
        negotiation_crew = Crew(
            agents=[
                creditor_agent,
                debtor_agent,
                mediator_agent,
                regulatory_agent,
                market_maker_agent,
                pdf_generator_agent
            ],
            tasks=[
                create_negotiation_task(context),
                create_evaluation_task(context),
                create_mediation_task(context),
                create_compliance_task(context),
                create_market_analysis_task(context),
                create_report_generation_task(context)
            ],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute negotiation
        logger.info("Executing negotiation crew...")
        result = negotiation_crew.kickoff()
        
        # Convert result to string if needed
        crew_output_str = str(result)
        
        # Parse the crew output to extract negotiation details
        parsed_results = parse_crew_output(crew_output_str)
        
        # Build negotiation result dictionary
        negotiation_result = {
            'status': 'success' if parsed_results['final_amount'] > 0 else 'pending',
            'final_amount': parsed_results['final_amount'],
            'timeline_days': parsed_results['timeline_days'],
            'interest_rate': parsed_results['interest_rate'],
            'payment_type': parsed_results['payment_type'],
            'installment_details': {
                'num_installments': parsed_results['num_installments'],
                'installment_amount': parsed_results['installment_amount']
            },
            'compliance_status': parsed_results['compliance_status'],
            'violations_count': len(parsed_results['violations']),
            'violations': parsed_results['violations'],
            'warnings_count': 1 if parsed_results['violations'] else 0,
            'negotiation_rounds': len(context.negotiation_history),
            'market_analysis': {
                'fair_market_value': parsed_results['fair_market_value'],
                'risk_rating': parsed_results['risk_rating'],
                'liquidity_score': parsed_results['liquidity_score'],
                'tradeable': parsed_results['tradeable']
            }
        }
        
        # Generate PDF report with crew output
        logger.info("Generating PDF report...")
        pdf_generator = PDFReportGenerator()
        report_filename = pdf_generator.generate_negotiation_report(
            context, 
            negotiation_result,
            crew_output_str,  # Pass the crew output to include in PDF
            f"negotiation_{creditor_id}_{debtor_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        logger.info(f"Negotiation completed successfully. Report saved to: {report_filename}")
        
        # Save crew output to text file for reference
        crew_output_filename = f"crew_output_{creditor_id}_{debtor_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(crew_output_filename, 'w') as f:
            f.write("CrewAI Negotiation Output\n")
            f.write("=" * 50 + "\n\n")
            f.write(crew_output_str)
        
        logger.info(f"Crew output saved to: {crew_output_filename}")
        
        return {
            'success': True,
            'negotiation_result': negotiation_result,
            'crew_output': crew_output_str,
            'crew_output_file': crew_output_filename,
            'report_filename': report_filename,
            'context': context
        }
        
    except Exception as e:
        logger.error(f"Error during negotiation: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'negotiation_result': None,
            'crew_output': None,
            'crew_output_file': None,
            'report_filename': None,
            'context': None
        }

# Example usage
if __name__ == "__main__":
    # Check if API key is set
    if os.environ.get("OPENAI_API_KEY") == "your-openai-api-key-here":
        logger.warning("Please set your OpenAI API key in the OPENAI_API_KEY environment variable")
        logger.warning("Example: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    # Example financial profile
    example_profile = FinancialProfile(
        monthly_income=5000,
        monthly_obligations=3000,
        liquid_assets=10000,
        credit_score=650,
        debt_to_income=0.4,
        employment_status="stable",
        hardship_factors=["medical_expenses", "job_reduction"]
    )
    
    logger.info("Starting payment negotiation system...")
    
    # Run negotiation
    result = run_payment_negotiation(
        creditor_id="CRED_001",
        debtor_id="DEBT_001",
        initial_debt=50000,
        financial_profile=example_profile
    )
    
    if result['success']:
        print(f"\n✅ Negotiation completed successfully!")
        print(f"\n📊 Negotiation Results:")
        print(f"   Final settlement: ${result['negotiation_result']['final_amount']:,.2f}")
        print(f"   Settlement percentage: {result['negotiation_result']['final_amount']/50000*100:.1f}%")
        print(f"   Payment timeline: {result['negotiation_result']['timeline_days']} days")
        print(f"   Interest rate: {result['negotiation_result']['interest_rate']:.1%}")
        
        if result['negotiation_result']['installment_details']['num_installments'] > 0:
            print(f"   Payment plan: {result['negotiation_result']['installment_details']['num_installments']} installments")
            print(f"   Installment amount: ${result['negotiation_result']['installment_details']['installment_amount']:,.2f}")
        
        print(f"\n📋 Compliance Status:")
        print(f"   Status: {result['negotiation_result']['compliance_status']}")
        print(f"   Violations: {result['negotiation_result']['violations_count']}")
        if result['negotiation_result']['violations']:
            for violation in result['negotiation_result']['violations']:
                print(f"   - {violation}")
        
        print(f"\n📈 Market Analysis:")
        market = result['negotiation_result']['market_analysis']
        print(f"   Risk Rating: {market['risk_rating']}")
        print(f"   Liquidity Score: {market['liquidity_score']}")
        print(f"   Fair Market Value: ${market['fair_market_value']:,.2f}")
        print(f"   Tradeable: {'Yes' if market['tradeable'] else 'No'}")
        
        print(f"\n📄 Reports Generated:")
        print(f"   PDF Report: {result['report_filename']}")
        print(f"   Crew Output: {result['crew_output_file']}")
        print(f"\n💡 The PDF report includes:")
        print(f"   - Complete CrewAI negotiation analysis")
        print(f"   - Settlement details and payment terms")
        print(f"   - Compliance assessment")
        print(f"   - Market valuation")
        print(f"   - Full negotiation history")
    else:
        print(f"\n❌ Negotiation failed: {result['error']}")
        print("Please check the logs for more details.")