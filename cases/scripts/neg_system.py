from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Type
from datetime import datetime, timedelta
import json
import logging
import os
from enum import Enum
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Data Models
class DebtNegotiation(BaseModel):
    negotiation_id: str
    creditor_id: str
    debtor_id: str
    original_amount: float
    current_offer: Optional[float] = None
    interest_rate: Optional[float] = None
    payment_timeline_days: Optional[int] = None
    status: str = "active"
    compliance_violations: List[str] = Field(default_factory=list)
    mediation_required: bool = False

class DebtorProfile(BaseModel):
    debtor_id: str
    credit_score: int
    monthly_income: float
    monthly_obligations: float
    liquid_assets: float
    employment_status: str
    hardship_factors: List[str] = Field(default_factory=list)
    max_affordable_payment: Optional[float] = None

class MarketData(BaseModel):
    instrument_id: str
    debt_amount: float
    settlement_percentage: float
    risk_rating: str
    market_value: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None

# Input Models for Tools
class DebtSettlementInput(BaseModel):
    original_amount: float
    credit_score: int
    monthly_income: float
    monthly_obligations: float

class ComplianceInput(BaseModel):
    offer_amount: float
    interest_rate: float
    payment_terms: str
    debtor_state: str = "CA"

class MarketAnalysisInput(BaseModel):
    debt_amount: float
    settlement_percentage: float
    risk_rating: str
    market_conditions: str = "normal"

class PaymentPlanInput(BaseModel):
    settlement_amount: float
    monthly_capacity: float
    preferred_timeline_months: int = 12

class PDFReportInput(BaseModel):
    negotiation_id: str
    report_content: str
    report_type: str = "negotiation_summary"
    output_filename: Optional[str] = None

# Custom Tools for CrewAI
class DebtSettlementTool(BaseTool):
    name: str = "debt_settlement_calculator"
    description: str = "Calculate a fair debt settlement based on financial factors"
    args_schema: Type[BaseModel] = DebtSettlementInput

    def _run(self, original_amount: float, credit_score: int, 
             monthly_income: float, monthly_obligations: float) -> Dict[str, Any]:
        try:
            # Calculate debt-to-income ratio
            dti = monthly_obligations / monthly_income if monthly_income > 0 else 1.0
            
            # Base settlement percentage based on credit score
            if credit_score >= 750:
                base_percentage = 0.75
            elif credit_score >= 700:
                base_percentage = 0.65
            elif credit_score >= 650:
                base_percentage = 0.55
            elif credit_score >= 600:
                base_percentage = 0.45
            else:
                base_percentage = 0.35
            
            # Adjust for DTI
            if dti > 0.8:
                base_percentage *= 0.8
            elif dti > 0.6:
                base_percentage *= 0.9
            
            settlement_amount = original_amount * base_percentage
            
            return {
                "recommended_settlement": round(settlement_amount, 2),
                "settlement_percentage": round(base_percentage, 3),
                "monthly_payment_capacity": round((monthly_income - monthly_obligations) * 0.5, 2),
                "risk_assessment": "high" if credit_score < 650 or dti > 0.7 else "moderate" if credit_score < 700 else "low"
            }
        except Exception as e:
            logger.error(f"Error in debt settlement calculation: {e}")
            return {"error": str(e)}

class ComplianceTool(BaseTool):
    name: str = "compliance_checker"
    description: str = "Check if negotiation terms comply with regulations"
    args_schema: Type[BaseModel] = ComplianceInput

    def _run(self, offer_amount: float, interest_rate: float, 
             payment_terms: str, debtor_state: str = "CA") -> Dict[str, Any]:
        try:
            violations = []
            warnings = []
            
            # Check usury laws (simplified - varies by state)
            max_interest_rates = {
                "CA": 0.10,  # California: 10% for consumer loans
                "NY": 0.16,  # New York: 16%
                "TX": 0.18,  # Texas: 18%
                "FL": 0.18,  # Florida: 18%
                "IL": 0.09,  # Illinois: 9%
            }
            
            state_max_rate = max_interest_rates.get(debtor_state.upper(), 0.25)
            
            if interest_rate > state_max_rate:
                violations.append(f"Interest rate {interest_rate:.2%} exceeds state maximum of {state_max_rate:.2%}")
            
            # Check for predatory lending practices
            if interest_rate > 0.36:  # 36% APR is often considered predatory
                violations.append("Interest rate exceeds predatory lending threshold")
            
            # Check payment terms
            if "acceleration" in payment_terms.lower() and "notice" not in payment_terms.lower():
                warnings.append("Acceleration clause requires proper notice provisions")
            
            # Check for minimum settlement amount (should be reasonable)
            if offer_amount < 100:
                warnings.append("Settlement amount appears unusually low")
            
            return {
                "compliant": len(violations) == 0,
                "violations": violations,
                "warnings": warnings,
                "risk_level": "high" if violations else "medium" if warnings else "low"
            }
        except Exception as e:
            logger.error(f"Error in compliance check: {e}")
            return {"error": str(e)}

class MarketAnalysisTool(BaseTool):
    name: str = "market_analyzer"
    description: str = "Analyze market value of settled debt for trading"
    args_schema: Type[BaseModel] = MarketAnalysisInput

    def _run(self, debt_amount: float, settlement_percentage: float, 
             risk_rating: str, market_conditions: str = "normal") -> Dict[str, Any]:
        try:
            # Base market value calculation
            settlement_value = debt_amount * settlement_percentage
            
            # Risk adjustment factors
            risk_factors = {
                "AAA": 0.95,
                "AA": 0.90,
                "A": 0.85,
                "BBB": 0.75,
                "BB": 0.65,
                "B": 0.50,
                "C": 0.35,
                "DEFAULT": 0.5
            }
            
            risk_adjustment = risk_factors.get(risk_rating.upper(), risk_factors["DEFAULT"])
            
            # Market condition adjustments
            market_adjustments = {
                "bull": 1.05,
                "normal": 1.0,
                "bear": 0.95,
                "volatile": 0.90
            }
            
            market_adjustment = market_adjustments.get(market_conditions.lower(), 1.0)
            
            market_value = settlement_value * risk_adjustment * market_adjustment
            
            # Calculate bid-ask spread
            spread_percentage = 0.02 + (1 - risk_adjustment) * 0.03  # 2-5% spread based on risk
            
            return {
                "market_value": round(market_value, 2),
                "bid_price": round(market_value * (1 - spread_percentage / 2), 2),
                "ask_price": round(market_value * (1 + spread_percentage / 2), 2),
                "spread_percentage": round(spread_percentage, 4),
                "liquidity_score": risk_adjustment,
                "recommendation": "buy" if risk_adjustment > 0.7 else "hold" if risk_adjustment > 0.5 else "sell"
            }
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return {"error": str(e)}

class PaymentPlanTool(BaseTool):
    name: str = "payment_plan_generator"
    description: str = "Generate a structured payment plan"
    args_schema: Type[BaseModel] = PaymentPlanInput

    def _run(self, settlement_amount: float, monthly_capacity: float, 
             preferred_timeline_months: int = 12) -> Dict[str, Any]:
        try:
            # Calculate different payment options
            options = []
            
            # Option 1: Preferred timeline
            if preferred_timeline_months > 0:
                monthly_payment = settlement_amount / preferred_timeline_months
                if monthly_payment <= monthly_capacity:
                    options.append({
                        "type": "installment",
                        "monthly_payment": round(monthly_payment, 2),
                        "num_payments": preferred_timeline_months,
                        "total_amount": round(settlement_amount, 2),
                        "feasibility": "recommended"
                    })
            
            # Option 2: Maximum affordable timeline
            if monthly_capacity > 0:
                max_timeline = int(settlement_amount / monthly_capacity) + 1
                if max_timeline <= 60:  # Max 5 years
                    actual_monthly = settlement_amount / max_timeline
                    options.append({
                        "type": "installment",
                        "monthly_payment": round(actual_monthly, 2),
                        "num_payments": max_timeline,
                        "total_amount": round(settlement_amount, 2),
                        "feasibility": "affordable"
                    })
            
            # Option 3: Lump sum with discount
            lump_sum_discount = 0.15  # 15% discount for immediate payment
            options.append({
                "type": "lump_sum",
                "payment_amount": round(settlement_amount * (1 - lump_sum_discount), 2),
                "discount_percentage": lump_sum_discount,
                "feasibility": "requires_liquid_assets"
            })
            
            return {
                "payment_options": options,
                "recommended_option": options[0] if options else None,
                "monthly_capacity": monthly_capacity
            }
        except Exception as e:
            logger.error(f"Error generating payment plan: {e}")
            return {"error": str(e)}

class PDFGeneratorTool(BaseTool):
    name: str = "pdf_report_generator"
    description: str = "Generate a professional PDF report from negotiation content"
    args_schema: Type[BaseModel] = PDFReportInput

    def _run(self, negotiation_id: str, report_content: str, 
             report_type: str = "negotiation_summary", output_filename: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Set default filename if not provided
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{report_type}_{negotiation_id}_{timestamp}.pdf"
            
            # Ensure output directory exists
            os.makedirs("reports", exist_ok=True)
            filepath = os.path.join("reports", output_filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=letter,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Create custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                spaceBefore=20,
                textColor=colors.darkblue
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=12,
                alignment=TA_LEFT
            )
            
            # Build PDF content
            story = []
            
            # Title
            title = f"Debt Negotiation Report - {negotiation_id.upper()}"
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 20))
            
            # Header information
            header_info = [
                ['Report Type:', report_type.replace('_', ' ').title()],
                ['Negotiation ID:', negotiation_id],
                ['Generated On:', datetime.now().strftime("%B %d, %Y at %I:%M %p")],
                ['Generated By:', 'AI Debt Negotiation System']
            ]
            
            header_table = Table(header_info, colWidths=[2*inch, 4*inch])
            header_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(header_table)
            story.append(Spacer(1, 30))
            
            # Process report content
            sections = self._parse_report_content(report_content)
            
            for section_title, section_content in sections.items():
                # Add section heading
                story.append(Paragraph(section_title, heading_style))
                
                # Add section content
                if isinstance(section_content, list):
                    for item in section_content:
                        story.append(Paragraph(f"• {item}", normal_style))
                elif isinstance(section_content, dict):
                    # Create table for dictionary data
                    table_data = [[k, str(v)] for k, v in section_content.items()]
                    if table_data:
                        content_table = Table(table_data, colWidths=[2.5*inch, 3.5*inch])
                        content_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 0), (-1, -1), 9),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(content_table)
                else:
                    # Regular text content
                    paragraphs = str(section_content).split('\n')
                    for para in paragraphs:
                        if para.strip():
                            story.append(Paragraph(para.strip(), normal_style))
                
                story.append(Spacer(1, 15))
            
            # Footer
            story.append(Spacer(1, 30))
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=colors.grey
            )
            story.append(Paragraph("This report was generated by the AI Debt Negotiation System", footer_style))
            story.append(Paragraph(f"Report contains confidential financial information", footer_style))
            
            # Build PDF
            doc.build(story)
            
            return {
                "success": True,
                "filename": output_filename,
                "filepath": filepath,
                "file_size_bytes": os.path.getsize(filepath),
                "pages_generated": "estimated 1-3 pages",
                "generation_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return {"error": str(e), "success": False}
    
    def _parse_report_content(self, content: str) -> Dict[str, Any]:
        """Parse the report content into structured sections"""
        try:
            # Try to parse as JSON first
            if content.strip().startswith('{'):
                data = json.loads(content)
                return data
        except:
            pass
        
        # Parse as text with sections
        sections = {}
        current_section = "Summary"
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line looks like a section header
            if (line.endswith(':') or 
                line.isupper() or 
                any(keyword in line.lower() for keyword in ['analysis', 'summary', 'recommendation', 'terms', 'proposal'])):
                
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = line.rstrip(':')
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections if sections else {"Report Content": content}

# CrewAI Agents
class PaymentNegotiationAgents:
    def __init__(self, llm):
        self.llm = llm
        # Initialize tools
        self.debt_settlement_tool = DebtSettlementTool()
        self.compliance_tool = ComplianceTool()
        self.market_analysis_tool = MarketAnalysisTool()
        self.payment_plan_tool = PaymentPlanTool()
        self.pdf_generator_tool = PDFGeneratorTool()
    
    def creditor_agent(self) -> Agent:
        return Agent(
            role='Senior Debt Recovery Specialist',
            goal='Maximize debt recovery while maintaining ethical practices and regulatory compliance',
            backstory="""You are an experienced debt recovery specialist with 15 years in the financial industry. 
            You understand the importance of balancing firm negotiation with empathy and legal compliance. 
            Your expertise includes assessing debtor capabilities, structuring payment plans, and maximizing 
            recovery rates while maintaining positive relationships. You always prioritize sustainable 
            agreements over aggressive collection tactics.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.debt_settlement_tool, self.payment_plan_tool]
        )
    
    def debtor_advocate_agent(self) -> Agent:
        return Agent(
            role='Financial Hardship Counselor',
            goal='Protect debtor interests while negotiating fair and sustainable payment arrangements',
            backstory="""You are a certified financial counselor specializing in helping individuals 
            facing financial hardship. With deep knowledge of consumer rights and debt negotiation strategies, 
            you advocate for fair settlements that consider the debtor's actual ability to pay. You understand 
            bankruptcy alternatives and always seek solutions that allow debtors to maintain dignity while 
            resolving their obligations.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.debt_settlement_tool, self.payment_plan_tool]
        )
    
    def mediator_agent(self) -> Agent:
        return Agent(
            role='Professional Debt Mediator',
            goal='Facilitate fair agreements between creditors and debtors through balanced negotiation',
            backstory="""You are a certified mediator with expertise in financial disputes. Your approach 
            combines deep understanding of both creditor and debtor perspectives with proven mediation 
            techniques. You excel at finding creative solutions that satisfy both parties while ensuring 
            all agreements are legally sound and sustainable. Your success rate in reaching mutually 
            beneficial agreements exceeds 85%.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self.debt_settlement_tool, self.payment_plan_tool]
        )
    
    def compliance_officer_agent(self) -> Agent:
        return Agent(
            role='Regulatory Compliance Officer',
            goal='Ensure all negotiations comply with federal and state regulations',
            backstory="""You are a regulatory compliance expert with comprehensive knowledge of FDCPA, 
            TCPA, FCRA, and state-specific debt collection laws. Your role is to review all negotiation 
            terms for legal compliance and flag any potential violations. You stay updated on changing 
            regulations and ensure that all parties operate within legal boundaries while protecting 
            consumer rights.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.compliance_tool]
        )
    
    def market_analyst_agent(self) -> Agent:
        return Agent(
            role='Debt Market Analyst',
            goal='Analyze and price debt instruments for secondary market trading',
            backstory="""You are a quantitative analyst specializing in distressed debt markets. With 
            expertise in risk assessment, market dynamics, and pricing models, you evaluate settled debts 
            for their investment potential. You understand market cycles, risk-return profiles, and can 
            accurately price debt instruments while providing liquidity to the market.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.market_analysis_tool]
        )
    
    def report_generator_agent(self) -> Agent:
        return Agent(
            role='Financial Report Specialist',
            goal='Generate comprehensive and professional PDF reports of debt negotiations',
            backstory="""You are a skilled financial report writer with expertise in creating clear, 
            professional documentation of debt negotiations and settlements. You excel at organizing 
            complex financial information into structured, easy-to-understand reports that serve both 
            legal and business purposes. Your reports are known for their clarity, completeness, and 
            professional presentation. You understand the importance of proper documentation in 
            financial negotiations and ensure all key details are captured accurately.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.pdf_generator_tool]
        )

# CrewAI Tasks
class PaymentNegotiationTasks:
    def creditor_evaluation_task(self, agent: Agent, negotiation: DebtNegotiation, 
                                debtor_profile: DebtorProfile) -> Task:
        return Task(
            description=f"""Evaluate the debtor's financial situation and propose an initial settlement offer.
            
            Use the debt_settlement_calculator tool with these parameters:
            - original_amount: {negotiation.original_amount}
            - credit_score: {debtor_profile.credit_score}
            - monthly_income: {debtor_profile.monthly_income}
            - monthly_obligations: {debtor_profile.monthly_obligations}
            
            Then use the payment_plan_generator tool to create payment options.
            
            Debtor Profile Context:
            - Liquid Assets: ${debtor_profile.liquid_assets:,.2f}
            - Employment Status: {debtor_profile.employment_status}
            - Hardship Factors: {', '.join(debtor_profile.hardship_factors) if debtor_profile.hardship_factors else 'None'}
            
            Provide a detailed assessment including:
            1. Recommended settlement amount and percentage
            2. Proposed interest rate (if applicable)
            3. Payment timeline options
            4. Risk assessment
            5. Justification for the proposal
            """,
            agent=agent,
            expected_output="A comprehensive settlement proposal with detailed financial analysis and justification"
        )
    
    def debtor_response_task(self, agent: Agent, creditor_offer: str, 
                           debtor_profile: DebtorProfile) -> Task:
        return Task(
            description=f"""Review the creditor's settlement offer and provide a counter-proposal that 
            protects the debtor's interests while being realistic.
            
            Creditor's Offer Summary:
            {creditor_offer}
            
            Use the available tools to calculate fair settlement amounts and payment plans.
            
            Debtor's Financial Situation:
            - Maximum Affordable Payment: ${debtor_profile.max_affordable_payment or 'To be determined'}
            - Current Hardships: {', '.join(debtor_profile.hardship_factors) if debtor_profile.hardship_factors else 'None'}
            - Monthly Income: ${debtor_profile.monthly_income:,.2f}
            - Monthly Obligations: ${debtor_profile.monthly_obligations:,.2f}
            
            Provide:
            1. Analysis of the creditor's offer fairness
            2. Counter-offer with justification
            3. Alternative payment structures if needed
            4. Documentation of hardship factors
            5. Bankruptcy comparison if relevant
            """,
            agent=agent,
            expected_output="A detailed counter-proposal with supporting documentation and alternatives"
        )
    
    def mediation_task(self, agent: Agent, negotiation: DebtNegotiation, 
                      creditor_position: str, debtor_position: str) -> Task:
        return Task(
            description=f"""Mediate between the creditor and debtor to reach a fair agreement.
            
            Negotiation ID: {negotiation.negotiation_id}
            Original Amount: ${negotiation.original_amount:,.2f}
            
            Current Positions:
            Creditor's Position: {creditor_position}
            Debtor's Position: {debtor_position}
            
            Use the debt settlement and payment plan tools to find compromise solutions.
            
            Your mediation should:
            1. Acknowledge both parties' valid concerns
            2. Propose 2-3 compromise solutions using the available tools
            3. Highlight benefits of agreement vs. litigation
            4. Suggest implementation timeline
            5. Ensure sustainability of any agreement
            """,
            agent=agent,
            expected_output="A mediation report with multiple compromise solutions and implementation plan"
        )
    
    def compliance_review_task(self, agent: Agent, proposed_terms: Dict[str, Any], 
                             state: str = "CA") -> Task:
        return Task(
            description=f"""Review the proposed settlement terms for regulatory compliance.
            
            Use the compliance_checker tool with the proposed terms.
            
            Proposed Terms Summary:
            {json.dumps(proposed_terms, indent=2)}
            
            Debtor State: {state}
            
            Provide:
            - Compliance status (Compliant/Non-compliant)
            - List of any violations
            - Required modifications
            - Risk assessment
            - Documentation requirements
            """,
            agent=agent,
            expected_output="A comprehensive compliance report with specific violations and remediation steps"
        )
    
    def market_analysis_task(self, agent: Agent, settled_debt: MarketData) -> Task:
        return Task(
            description=f"""Analyze the settled debt for secondary market trading potential.
            
            Use the market_analyzer tool with these parameters:
            - debt_amount: {settled_debt.debt_amount}
            - settlement_percentage: {settled_debt.settlement_percentage}
            - risk_rating: {settled_debt.risk_rating}
            
            Provide:
            1. Market value assessment
            2. Recommended bid/ask prices
            3. Liquidity analysis
            4. Risk-return profile
            5. Trading recommendation
            6. Comparable market analysis
            """,
            agent=agent,
            expected_output="A detailed market analysis report with pricing recommendations and risk assessment"
        )
    
    def pdf_generation_task(self, agent: Agent, negotiation_id: str, 
                           negotiation_results: str, report_type: str = "negotiation_summary") -> Task:
        return Task(
            description=f"""Generate a comprehensive PDF report of the debt negotiation.
            
            Negotiation ID: {negotiation_id}
            Report Type: {report_type}
            
            Use the pdf_report_generator tool to create a professional PDF report.
            
            The report should include:
            1. Executive summary of the negotiation
            2. Financial analysis and recommendations
            3. Settlement terms and conditions
            4. Compliance verification
            5. Payment plan details
            6. Risk assessment
            7. Next steps and follow-up actions
            
            Negotiation Results to Include:
            {negotiation_results}
            
            Ensure the report is:
            - Professional and well-formatted
            - Complete with all key information
            - Easy to understand for all stakeholders
            - Legally compliant and properly documented
            """,
            agent=agent,
            expected_output="A comprehensive PDF report file with professional formatting and complete negotiation documentation"
        )

# Main Crew Orchestrator
class PaymentNegotiationCrew:
    def __init__(self, openai_api_key: str = None):
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4o-mini",  # Using a more cost-effective model
            openai_api_key=openai_api_key
        )
        
        self.agents = PaymentNegotiationAgents(self.llm)
        self.tasks = PaymentNegotiationTasks()
    
    def negotiate_debt_settlement(self, negotiation: DebtNegotiation, 
                                debtor_profile: DebtorProfile, 
                                generate_pdf: bool = True) -> Dict[str, Any]:
        """Run a complete debt negotiation process"""
        try:
            # Create agents
            creditor = self.agents.creditor_agent()
            compliance_officer = self.agents.compliance_officer_agent()
            report_generator = self.agents.report_generator_agent()
            
            # Create tasks
            creditor_task = self.tasks.creditor_evaluation_task(creditor, negotiation, debtor_profile)
            
            # Create crew for initial negotiation
            initial_crew = Crew(
                agents=[creditor, compliance_officer],
                tasks=[creditor_task],
                process=Process.sequential,
                verbose=True
            )
            
            # Run initial negotiation
            result = initial_crew.kickoff()
            
            pdf_result = None
            if generate_pdf:
                # Generate PDF report
                pdf_task = self.tasks.pdf_generation_task(
                    report_generator, 
                    negotiation.negotiation_id, 
                    str(result),
                    "settlement_negotiation"
                )
                
                pdf_crew = Crew(
                    agents=[report_generator],
                    tasks=[pdf_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                pdf_result = pdf_crew.kickoff()
            
            logger.info(f"Negotiation {negotiation.negotiation_id} completed")
            
            return {
                "negotiation_id": negotiation.negotiation_id,
                "result": str(result),
                "pdf_report": str(pdf_result) if pdf_result else None,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in negotiation {negotiation.negotiation_id}: {e}")
            return {
                "negotiation_id": negotiation.negotiation_id,
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
    
    def run_complex_negotiation(self, negotiation: DebtNegotiation, 
                              debtor_profile: DebtorProfile,
                              generate_pdf: bool = True) -> Dict[str, Any]:
        """Run a complex negotiation with mediation"""
        try:
            # Create all agents
            creditor = self.agents.creditor_agent()
            debtor_advocate = self.agents.debtor_advocate_agent()
            mediator = self.agents.mediator_agent()
            compliance_officer = self.agents.compliance_officer_agent()
            report_generator = self.agents.report_generator_agent()
            
            # Phase 1: Initial positions
            creditor_task = self.tasks.creditor_evaluation_task(creditor, negotiation, debtor_profile)
            
            # Create initial crew
            phase1_crew = Crew(
                agents=[creditor],
                tasks=[creditor_task],
                process=Process.sequential
            )
            
            creditor_position = phase1_crew.kickoff()
            
            # Phase 2: Debtor response
            debtor_task = self.tasks.debtor_response_task(
                debtor_advocate,
                str(creditor_position),
                debtor_profile
            )
            
            phase2_crew = Crew(
                agents=[debtor_advocate],
                tasks=[debtor_task],
                process=Process.sequential
            )
            
            debtor_position = phase2_crew.kickoff()
            
            # Phase 3: Mediation
            mediation_task = self.tasks.mediation_task(
                mediator,
                negotiation,
                str(creditor_position),
                str(debtor_position)
            )
            
            phase3_crew = Crew(
                agents=[mediator, compliance_officer],
                tasks=[mediation_task],
                process=Process.sequential
            )
            
            mediation_result = phase3_crew.kickoff()
            
            # Compile all results
            full_negotiation_results = {
                "creditor_position": str(creditor_position),
                "debtor_position": str(debtor_position),
                "mediation_result": str(mediation_result)
            }
            
            pdf_result = None
            if generate_pdf:
                # Generate comprehensive PDF report
                pdf_task = self.tasks.pdf_generation_task(
                    report_generator, 
                    negotiation.negotiation_id, 
                    json.dumps(full_negotiation_results, indent=2),
                    "complex_negotiation_mediation"
                )
                
                pdf_crew = Crew(
                    agents=[report_generator],
                    tasks=[pdf_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                pdf_result = pdf_crew.kickoff()
            
            return {
                "negotiation_id": negotiation.negotiation_id,
                "creditor_position": str(creditor_position),
                "debtor_position": str(debtor_position),
                "mediation_result": str(mediation_result),
                "pdf_report": str(pdf_result) if pdf_result else None,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in complex negotiation {negotiation.negotiation_id}: {e}")
            return {
                "negotiation_id": negotiation.negotiation_id,
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_debt_portfolio(self, debt_instruments: List[MarketData]) -> Dict[str, Any]:
        """Analyze a portfolio of debt instruments"""
        try:
            market_analyst = self.agents.market_analyst_agent()
            
            tasks = []
            for instrument in debt_instruments:
                task = self.tasks.market_analysis_task(market_analyst, instrument)
                tasks.append(task)
            
            crew = Crew(
                agents=[market_analyst],
                tasks=tasks,
                process=Process.sequential
            )
            
            results = crew.kickoff()
            
            return {
                "portfolio_analysis": str(results),
                "instruments_analyzed": len(debt_instruments),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Example Usage
def example_usage():
    """Example usage of the debt negotiation system"""
    # Initialize the crew (you'll need to provide your OpenAI API key)
    api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    
    if api_key == "your-openai-api-key-here":
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    crew_system = PaymentNegotiationCrew(openai_api_key=api_key)
    
    # Create a negotiation case
    negotiation = DebtNegotiation(
        negotiation_id="NEG_001",
        creditor_id="CRED_001",
        debtor_id="DEBT_001",
        original_amount=50000.00
    )
    
    # Create debtor profile
    debtor_profile = DebtorProfile(
        debtor_id="DEBT_001",
        credit_score=650,
        monthly_income=5000.00,
        monthly_obligations=3500.00,
        liquid_assets=5000.00,
        employment_status="stable",
        hardship_factors=["medical_expenses", "reduced_income"],
        max_affordable_payment=500.00
    )
    
    # Run simple negotiation with PDF generation
    print("Starting debt negotiation with PDF report generation...")
    result = crew_system.negotiate_debt_settlement(negotiation, debtor_profile, generate_pdf=True)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    if result.get("pdf_report"):
        print(f"\n📄 PDF Report generated successfully!")
        print(f"Check the 'reports' folder for the generated PDF file.")
    
    # Run complex negotiation with mediation and PDF (uncomment to run)
    # print("\nStarting complex negotiation with mediation and PDF report...")
    # complex_result = crew_system.run_complex_negotiation(negotiation, debtor_profile, generate_pdf=True)
    # print(f"Complex Result: {json.dumps(complex_result, indent=2)}")

if __name__ == "__main__":
    print("=== AI Debt Negotiation System with PDF Reports ===")
    print("Required dependencies: pip install crewai langchain-openai pydantic reportlab")
    print("=" * 60)
    example_usage()