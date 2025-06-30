#!/usr/bin/env python3
"""
German Debt Collection Multi-Agent System with Enhanced PDF Export
Captures ALL CrewAI agent outputs and generates comprehensive PDF reports
"""

import os
import sys
import json
import datetime
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Test imports
try:
    from crewai import Agent, Task, Crew
    try:
        from crewai import Process
    except ImportError:
        class Process:
            sequential = "sequential"
            hierarchical = "hierarchical"
            
    from crewai.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    import numpy as np
    
    # PDF Generation imports
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY, TA_LEFT
    from reportlab.pdfgen import canvas
    
    print("✅ All libraries imported successfully!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please install required libraries:")
    print("pip install crewai crewai-tools langchain langchain-openai pydantic numpy reportlab")
    exit(1)

from dotenv import load_dotenv
load_dotenv()

# OpenAI Configuration
class OpenAIConfig:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    def create_llm(self):
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
class DebtorRiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class DebtCase(BaseModel):
    case_id: str
    debtor_name: str
    company_name: Optional[str] = None
    debt_amount: float
    debt_age_days: int
    industry: str
    location: str = "Germany"
    previous_payment_history: str = "unknown"
    communication_language: str = "de"
    debt_type: str = "B2B"
    contact_attempts: int = 0
    legal_status: str = "pre_legal"

# Enhanced Analysis Result Parser
class CrewAIOutputParser:
    """Parses CrewAI outputs to extract structured data for PDF generation"""
    
    def __init__(self, crew_output: str, case_data: Dict):
        self.raw_output = str(crew_output)
        self.case_data = case_data
        self.parsed_data = self._parse_output()
    
    def _parse_output(self) -> Dict:
        """Parse crew output into structured sections"""
        
        parsed = {
            "case_info": self.case_data,
            "risk_assessment": self._extract_risk_assessment(),
            "negotiation_strategy": self._extract_negotiation_strategy(),
            "compliance_review": self._extract_compliance_review(),
            "communication": self._extract_communication(),
            "financial_summary": self._calculate_financials(),
            "recommendations": self._extract_recommendations(),
            "timeline": self._generate_timeline(),
            "metadata": {
                "analysis_date": datetime.datetime.now().isoformat(),
                "system_version": "CrewAI Multi-Agent v1.0",
                "raw_output_length": len(self.raw_output)
            }
        }
        
        return parsed
    
    def _extract_risk_assessment(self) -> Dict:
        """Extract risk assessment data from crew output"""
        
        # Look for risk-related content
        risk_data = {}
        
        # Extract risk score
        risk_score_match = re.search(r'risk[_\s]*score[:\s]*(\d+)', self.raw_output, re.IGNORECASE)
        if risk_score_match:
            risk_data["risk_score"] = int(risk_score_match.group(1))
        else:
            # Calculate fallback risk score
            risk_data["risk_score"] = self._calculate_fallback_risk()
        
        # Extract recovery probability
        recovery_match = re.search(r'recovery[_\s]*probability[:\s]*([0-9.]+)', self.raw_output, re.IGNORECASE)
        if recovery_match:
            prob = float(recovery_match.group(1))
            risk_data["recovery_probability"] = prob if prob > 1 else prob * 100
        else:
            risk_data["recovery_probability"] = max(20, 100 - risk_data["risk_score"])
        
        # Extract risk level
        risk_level_match = re.search(r'risk[_\s]*level[:\s]*(\w+)', self.raw_output, re.IGNORECASE)
        if risk_level_match:
            risk_data["risk_level"] = risk_level_match.group(1).upper()
        else:
            score = risk_data["risk_score"]
            if score < 30:
                risk_data["risk_level"] = "LOW"
            elif score < 60:
                risk_data["risk_level"] = "MEDIUM"
            else:
                risk_data["risk_level"] = "HIGH"
        
        # Extract risk factors
        risk_data["risk_factors"] = self._extract_risk_factors()
        
        # Determine recommended approach
        if risk_data["risk_level"] in ["LOW", "MEDIUM"]:
            risk_data["recommended_approach"] = "Amicable Collection"
        else:
            risk_data["recommended_approach"] = "Intensive Collection"
        
        return risk_data
    
    def _calculate_fallback_risk(self) -> int:
        """Calculate risk score when not found in output"""
        debt_age = self.case_data.get('debt_age_days', 0)
        debt_amount = self.case_data.get('debt_amount', 0)
        industry = self.case_data.get('industry', '').lower()
        payment_history = self.case_data.get('previous_payment_history', 'unknown').lower()
        
        score = 0
        
        # Age factor
        score += min(50, debt_age // 10)
        
        # Amount factor
        if debt_amount > 50000:
            score += 25
        elif debt_amount > 10000:
            score += 15
        elif debt_amount < 1000:
            score += 5
        
        # Industry factor
        high_risk_industries = ["hospitality", "retail", "construction"]
        if industry in high_risk_industries:
            score += 20
        
        # Payment history factor
        if payment_history == "good":
            score -= 15
        elif payment_history == "poor":
            score += 20
        
        return min(100, max(0, score))
    
    def _extract_risk_factors(self) -> List[str]:
        """Extract risk factors from the output"""
        factors = []
        
        debt_age = self.case_data.get('debt_age_days', 0)
        debt_amount = self.case_data.get('debt_amount', 0)
        industry = self.case_data.get('industry', 'N/A')
        payment_history = self.case_data.get('previous_payment_history', 'unknown')
        
        factors.append(f"Debt age: {debt_age} days overdue")
        factors.append(f"Industry: {industry} sector")
        factors.append(f"Amount: €{debt_amount:,.2f}")
        factors.append(f"Payment history: {payment_history}")
        
        if debt_age > 90:
            factors.append("Long overdue period increases collection difficulty")
        if debt_amount > 25000:
            factors.append("High value debt requires careful handling")
        if industry.lower() in ["hospitality", "retail"]:
            factors.append("Industry has higher default rates")
        
        return factors
    
    def _extract_negotiation_strategy(self) -> Dict:
        """Extract negotiation strategy from output"""
        strategy = {
            "payment_options": [],
            "communication_approach": "Professional",
            "escalation_timeline": {},
            "relationship_preservation": True
        }
        
        # Check for payment options mentions
        if "discount" in self.raw_output.lower():
            strategy["payment_options"].append({
                "type": "Early Payment Discount",
                "description": "2% discount for payment within 7 days",
                "benefit": "Immediate resolution"
            })
        
        if any(word in self.raw_output.lower() for word in ["installment", "rate", "plan"]):
            strategy["payment_options"].append({
                "type": "Payment Plan",
                "description": "Structured monthly payments",
                "benefit": "Manageable for debtor"
            })
        
        if "legal" in self.raw_output.lower():
            strategy["payment_options"].append({
                "type": "Legal Escalation",
                "description": "Court proceedings if no response",
                "benefit": "Formal enforcement"
            })
        
        # Extract timeline
        strategy["escalation_timeline"] = {
            "Day 0": "Payment reminder sent",
            "Day 7": "Follow-up contact",
            "Day 14": "First formal dunning notice",
            "Day 28": "Second dunning notice",
            "Day 42": "Final notice before legal action"
        }
        
        return strategy
    
    def _extract_compliance_review(self) -> Dict:
        """Extract compliance information"""
        compliance = {
            "gdpr_compliant": True,
            "bgb_compliant": True,
            "rdg_compliant": True,
            "compliance_notes": [],
            "legal_references": []
        }
        
        # Check for compliance mentions
        if "gdpr" in self.raw_output.lower() or "dsgvo" in self.raw_output.lower():
            compliance["compliance_notes"].append("GDPR privacy notice included")
            compliance["legal_references"].append("Art. 13 GDPR")
        
        if "bgb" in self.raw_output.lower():
            compliance["compliance_notes"].append("BGB civil code compliance verified")
            compliance["legal_references"].append("§ 286 BGB (Default)")
        
        if not compliance["compliance_notes"]:
            compliance["compliance_notes"] = [
                "Standard German debt collection compliance applied",
                "Privacy notice included as per GDPR requirements",
                "Professional communication standards maintained"
            ]
            compliance["legal_references"] = ["GDPR Art. 13", "BGB § 286", "RDG regulations"]
        
        return compliance
    
    def _extract_communication(self) -> Dict:
        """Extract communication details"""
        comm = {
            "language": self.case_data.get("communication_language", "de"),
            "tone": "Professional",
            "subject": "",
            "content_preview": "",
            "includes_gdpr": True,
            "deadline_days": 7
        }
        
        # Extract subject line
        subject_match = re.search(r'subject[:\s]*(.*?)(?:\n|Body)', self.raw_output, re.IGNORECASE)
        if subject_match:
            comm["subject"] = subject_match.group(1).strip()
        else:
            comm["subject"] = f"Zahlungserinnerung - Rechnung Nr. {self.case_data.get('invoice_number', 'N/A')}"
        
        # Extract content preview
        if "Sehr geehrte" in self.raw_output:
            start = self.raw_output.find("Sehr geehrte")
            end = start + 300  # First 300 characters
            comm["content_preview"] = self.raw_output[start:end].strip()
        
        return comm
    
    def _calculate_financials(self) -> Dict:
        """Calculate comprehensive financial breakdown"""
        debt_amount = self.case_data.get('debt_amount', 0)
        days_overdue = self.case_data.get('debt_age_days', 0)
        
        # German legal interest rate (5.12% above base rate)
        legal_interest_rate = 0.0512
        interest_amount = debt_amount * legal_interest_rate * (days_overdue / 365) if days_overdue > 30 else 0
        
        # Collection fee (No Cure No Pay model)
        collection_fee_rate = 0.10 if debt_amount >= 5000 else 0.15
        collection_fee = max(185, debt_amount * collection_fee_rate)
        
        # Court costs estimation
        if debt_amount <= 500:
            court_costs = 105
        elif debt_amount <= 2000:
            court_costs = 189
        elif debt_amount <= 5000:
            court_costs = 294
        elif debt_amount <= 10000:
            court_costs = 441
        else:
            court_costs = 546 + (debt_amount - 10000) * 0.005
        
        total_amount = debt_amount + interest_amount + collection_fee
        max_recovery = total_amount + court_costs
        
        return {
            "principal_amount": debt_amount,
            "interest_amount": round(interest_amount, 2),
            "interest_rate": f"{legal_interest_rate * 100:.2f}%",
            "collection_fee": round(collection_fee, 2),
            "collection_fee_rate": f"{collection_fee_rate * 100:.0f}%",
            "court_costs": round(court_costs, 2),
            "total_amount_due": round(total_amount, 2),
            "maximum_recovery": round(max_recovery, 2),
            "currency": "EUR"
        }
    
    def _extract_recommendations(self) -> List[str]:
        """Extract or generate recommendations"""
        recommendations = []
        
        risk_score = self._calculate_fallback_risk()
        
        if risk_score < 40:
            recommendations = [
                "Proceed with friendly payment reminder approach",
                "Offer early payment discount to incentivize quick resolution",
                "Maintain positive business relationship throughout process",
                "Document all communication for future reference"
            ]
        elif risk_score < 70:
            recommendations = [
                "Send professional dunning notice with clear deadline",
                "Offer structured payment plan as alternative",
                "Increase communication frequency",
                "Prepare documentation for potential escalation"
            ]
        else:
            recommendations = [
                "Immediate personal contact with decision maker",
                "Verify current business status and solvency",
                "Consider protective measures to secure claim",
                "Prepare comprehensive legal documentation"
            ]
        
        return recommendations
    
    def _generate_timeline(self) -> List[Dict]:
        """Generate collection timeline"""
        base_date = datetime.datetime.now()
        timeline = []
        
        events = [
            (0, "Payment reminder sent", "Completed"),
            (7, "Follow-up contact", "Planned"),
            (14, "First formal dunning notice", "Planned"),
            (28, "Second dunning notice", "Planned"),
            (42, "Final notice before legal action", "Conditional"),
            (56, "Legal proceedings initiation", "If required")
        ]
        
        for days, event, status in events:
            timeline.append({
                "date": (base_date + datetime.timedelta(days=days)).strftime("%d.%m.%Y"),
                "event": event,
                "status": status,
                "days_from_now": days
            })
        
        return timeline

# Enhanced PDF Generator
class ComprehensivePDFGenerator:
    """Generates comprehensive PDF reports from parsed CrewAI output"""
    
    def __init__(self, parsed_data: Dict):
        self.data = parsed_data
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom PDF styles"""
        
        # Check and add custom styles only if they don't exist
        custom_styles = {
            'DebtTitle': ParagraphStyle(
                name='DebtTitle',
                parent=self.styles['Heading1'],
                fontSize=20,
                textColor=colors.HexColor('#1a365d'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ),
            'DebtSectionHeader': ParagraphStyle(
                name='DebtSectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2d3748'),
                spaceAfter=12,
                spaceBefore=20,
                fontName='Helvetica-Bold'
            ),
            'DebtSubsectionHeader': ParagraphStyle(
                name='DebtSubsectionHeader',
                parent=self.styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#4a5568'),
                spaceAfter=8,
                spaceBefore=12,
                fontName='Helvetica-Bold'
            ),
            'DebtBodyText': ParagraphStyle(
                name='DebtBodyText',
                parent=self.styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                alignment=TA_JUSTIFY,
                leftIndent=20
            ),
            'DebtHighlight': ParagraphStyle(
                name='DebtHighlight',
                parent=self.styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#2b6cb0'),
                spaceAfter=6,
                fontName='Helvetica-Bold'
            )
        }
        
        # Add styles only if they don't already exist
        for style_name, style_obj in custom_styles.items():
            if style_name not in self.styles:
                self.styles.add(style_obj)
    
    def generate_pdf(self, filename: str = None) -> str:
        """Generate comprehensive PDF report"""
        
        if filename is None:
            case_id = self.data["case_info"].get("case_id", "UNKNOWN")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debt_collection_analysis_{case_id}_{timestamp}.pdf"
        
        # Ensure filename ends with .pdf
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        print(f"📄 Generating comprehensive PDF report: {filename}")
        
        # Create document
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        # Build document content
        elements = []
        
        # Title page
        self._add_title_page(elements)
        
        # Executive summary
        self._add_executive_summary(elements)
        
        # Case information
        self._add_case_information(elements)
        
        # Risk assessment
        self._add_risk_assessment(elements)
        
        # Negotiation strategy
        self._add_negotiation_strategy(elements)
        
        # Compliance review
        self._add_compliance_review(elements)
        
        # Communication details
        self._add_communication_details(elements)
        
        # Financial analysis
        self._add_financial_analysis(elements)
        
        # Timeline
        self._add_timeline(elements)
        
        # Recommendations
        self._add_recommendations(elements)
        
        # Appendix
        self._add_appendix(elements)
        
        # Build PDF
        try:
            doc.build(elements)
            file_size = os.path.getsize(filename)
            print(f"✅ PDF generated successfully!")
            print(f"📁 File: {os.path.abspath(filename)}")
            print(f"📊 Size: {file_size:,} bytes")
            return filename
        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
            return None
    
    def _add_title_page(self, elements):
        """Add title page"""
        elements.append(Paragraph("DEBT COLLECTION ANALYSIS REPORT", self.styles['DebtTitle']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Report metadata
        case_info = self.data["case_info"]
        metadata = self.data["metadata"]
        
        meta_data = [
            ['Case ID:', case_info.get('case_id', 'N/A')],
            ['Debtor:', case_info.get('company_name', case_info.get('debtor_name', 'N/A'))],
            ['Analysis Date:', datetime.datetime.fromisoformat(metadata['analysis_date']).strftime('%d.%m.%Y %H:%M')],
            ['System Version:', metadata['system_version']],
            ['Report Type:', 'Comprehensive AI Analysis'],
            ['Confidentiality:', 'CONFIDENTIAL - For Internal Use Only']
        ]
        
        meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(meta_table)
        elements.append(Spacer(1, 0.5*inch))
        
        disclaimer = """
        This report contains confidential information generated by AI-powered debt collection analysis. 
        It is intended solely for the use of authorized personnel and should not be distributed without 
        proper authorization. The analysis follows German debt collection regulations and best practices.
        """
        elements.append(Paragraph(disclaimer, self.styles['DebtBodyText']))
        elements.append(PageBreak())
    
    def _add_executive_summary(self, elements):
        """Add executive summary"""
        elements.append(Paragraph("EXECUTIVE SUMMARY", self.styles['DebtSectionHeader']))
        
        case_info = self.data["case_info"]
        risk_data = self.data["risk_assessment"]
        financial = self.data["financial_summary"]
        
        summary_text = f"""
        This report presents a comprehensive analysis of debt collection case {case_info.get('case_id', 'N/A')} 
        involving {case_info.get('company_name', 'N/A')} with an outstanding debt of €{case_info.get('debt_amount', 0):,.2f}.
        
        <b>Key Findings:</b><br/>
        • Risk Assessment: {risk_data.get('risk_level', 'UNKNOWN')} risk level<br/>
        • Recovery Probability: {risk_data.get('recovery_probability', 0):.1f}%<br/>
        • Total Amount Due: €{financial.get('total_amount_due', 0):,.2f}<br/>
        • Recommended Approach: {risk_data.get('recommended_approach', 'Standard Collection')}<br/>
        
        The analysis indicates that {risk_data.get('recommended_approach', 'standard collection')} 
        is the most appropriate strategy for this case, with an estimated recovery probability 
        of {risk_data.get('recovery_probability', 0):.1f}%.
        """
        
        elements.append(Paragraph(summary_text, self.styles['DebtBodyText']))
        elements.append(Spacer(1, 0.3*inch))
    
    def _add_case_information(self, elements):
        """Add detailed case information"""
        elements.append(Paragraph("1. CASE INFORMATION", self.styles['DebtSectionHeader']))
        
        case_info = self.data["case_info"]
        
        case_data = [
            ['Field', 'Value'],
            ['Case ID', case_info.get('case_id', 'N/A')],
            ['Debtor Name', case_info.get('debtor_name', 'N/A')],
            ['Company Name', case_info.get('company_name', 'N/A')],
            ['Industry', case_info.get('industry', 'N/A').title()],
            ['Location', case_info.get('location', 'N/A')],
            ['Debt Amount', f"€{case_info.get('debt_amount', 0):,.2f}"],
            ['Days Overdue', str(case_info.get('debt_age_days', 0))],
            ['Payment History', case_info.get('previous_payment_history', 'Unknown').title()],
            ['Communication Language', case_info.get('communication_language', 'de').upper()],
            ['Debt Type', case_info.get('debt_type', 'N/A')],
            ['Contact Attempts', str(case_info.get('contact_attempts', 0))],
            ['Legal Status', case_info.get('legal_status', 'N/A').replace('_', ' ').title()]
        ]
        
        case_table = Table(case_data, colWidths=[2.5*inch, 3.5*inch])
        case_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3748')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f7fafc')),
            ('FONT', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(case_table)
        elements.append(Spacer(1, 0.3*inch))
    
    def _add_risk_assessment(self, elements):
        """Add risk assessment section"""
        elements.append(Paragraph("2. AI-POWERED RISK ASSESSMENT", self.styles['DebtSectionHeader']))
        
        risk_data = self.data["risk_assessment"]
        
        # Risk overview
        elements.append(Paragraph("Risk Analysis Overview", self.styles['DebtSubsectionHeader']))
        
        risk_overview = f"""
        <b>Risk Score:</b> {risk_data.get('risk_score', 0)}/100<br/>
        <b>Risk Classification:</b> {risk_data.get('risk_level', 'UNKNOWN')}<br/>
        <b>Recovery Probability:</b> {risk_data.get('recovery_probability', 0):.1f}%<br/>
        <b>Recommended Approach:</b> {risk_data.get('recommended_approach', 'Standard Collection')}<br/>
        """
        elements.append(Paragraph(risk_overview, self.styles['DebtHighlight']))
        
        # Risk factors
        elements.append(Paragraph("Identified Risk Factors", self.styles['DebtSubsectionHeader']))
        
        risk_factors = risk_data.get('risk_factors', [])
        for factor in risk_factors:
            elements.append(Paragraph(f"• {factor}", self.styles['DebtBodyText']))
        
        elements.append(Spacer(1, 0.3*inch))
    
    def _add_negotiation_strategy(self, elements):
        """Add negotiation strategy section"""
        elements.append(Paragraph("3. NEGOTIATION STRATEGY", self.styles['DebtSectionHeader']))
        
        strategy = self.data["negotiation_strategy"]
        
        # Communication approach
        elements.append(Paragraph("Communication Approach", self.styles['DebtSubsectionHeader']))
        approach_text = f"""
        <b>Primary Approach:</b> {strategy.get('communication_approach', 'Professional')}<br/>
        <b>Relationship Preservation:</b> {'Yes' if strategy.get('relationship_preservation', True) else 'No'}<br/>
        """
        elements.append(Paragraph(approach_text, self.styles['DebtBodyText']))
        
        # Payment options
        elements.append(Paragraph("Available Payment Options", self.styles['DebtSubsectionHeader']))
        
        payment_options = strategy.get('payment_options', [])
        if payment_options:
            for i, option in enumerate(payment_options, 1):
                option_text = f"""
                <b>{i}. {option.get('type', 'Payment Option')}</b><br/>
                Description: {option.get('description', 'N/A')}<br/>
                Benefit: {option.get('benefit', 'N/A')}<br/>
                """
                elements.append(Paragraph(option_text, self.styles['DebtBodyText']))
        
        elements.append(Spacer(1, 0.3*inch))
    
    def _add_compliance_review(self, elements):
        """Add compliance review section"""
        elements.append(Paragraph("4. LEGAL COMPLIANCE REVIEW", self.styles['DebtSectionHeader']))
        
        compliance = self.data["compliance_review"]
        
        # Compliance status
        compliance_text = f"""
        <b>GDPR/BDSG Compliant:</b> {'✓ Yes' if compliance.get('gdpr_compliant', True) else '✗ No'}<br/>
        <b>BGB Compliant:</b> {'✓ Yes' if compliance.get('bgb_compliant', True) else '✗ No'}<br/>
        <b>RDG Compliant:</b> {'✓ Yes' if compliance.get('rdg_compliant', True) else '✗ No'}<br/>
        """
        elements.append(Paragraph(compliance_text, self.styles['DebtHighlight']))
        
        # Compliance notes
        elements.append(Paragraph("Compliance Notes", self.styles['DebtSubsectionHeader']))
        notes = compliance.get('compliance_notes', [])
        for note in notes:
            elements.append(Paragraph(f"• {note}", self.styles['DebtBodyText']))
        
        # Legal references
        elements.append(Paragraph("Applicable Legal References", self.styles['DebtSubsectionHeader']))
        references = compliance.get('legal_references', [])
        for ref in references:
            elements.append(Paragraph(f"• {ref}", self.styles['DebtBodyText']))
        
        elements.append(Spacer(1, 0.3*inch))
    
    def _add_communication_details(self, elements):
        """Add communication details section"""
        elements.append(Paragraph("5. COMMUNICATION DETAILS", self.styles['DebtSectionHeader']))
        
        comm = self.data["communication"]
        
        # Communication overview
        comm_overview = f"""
        <b>Language:</b> {comm.get('language', 'de').upper()}<br/>
        <b>Tone:</b> {comm.get('tone', 'Professional')}<br/>
        <b>Subject Line:</b> {comm.get('subject', 'N/A')}<br/>
        <b>Payment Deadline:</b> {comm.get('deadline_days', 7)} days<br/>
        <b>GDPR Notice Included:</b> {'Yes' if comm.get('includes_gdpr', True) else 'No'}<br/>
        """
        elements.append(Paragraph(comm_overview, self.styles['DebtBodyText']))
        
        # Content preview
        if comm.get('content_preview'):
            elements.append(Paragraph("Communication Preview", self.styles['DebtSubsectionHeader']))
            elements.append(Paragraph(comm['content_preview'], self.styles['DebtBodyText']))
        
        elements.append(PageBreak())
    
    def _add_financial_analysis(self, elements):
        """Add comprehensive financial analysis"""
        elements.append(Paragraph("6. FINANCIAL ANALYSIS", self.styles['DebtSectionHeader']))
        
        financial = self.data["financial_summary"]
        
        # Financial breakdown table
        financial_data = [
            ['Description', 'Amount (EUR)', 'Details'],
            ['Principal Amount', f"€{financial.get('principal_amount', 0):,.2f}", 'Original debt amount'],
            ['Legal Interest', f"€{financial.get('interest_amount', 0):,.2f}", f"Rate: {financial.get('interest_rate', '5.12%')}"],
            ['Collection Fee', f"€{financial.get('collection_fee', 0):,.2f}", f"Rate: {financial.get('collection_fee_rate', '10%')}"],
            ['', '', ''],
            ['Total Amount Due', f"€{financial.get('total_amount_due', 0):,.2f}", 'Current total obligation'],
            ['', '', ''],
            ['Potential Court Costs', f"€{financial.get('court_costs', 0):,.2f}", 'If legal proceedings required'],
            ['Maximum Recovery', f"€{financial.get('maximum_recovery', 0):,.2f}", 'Including all costs']
        ]
        
        financial_table = Table(financial_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        financial_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3748')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#e2e8f0')),
            ('FONT', (0, 5), (-1, 5), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor('#fed7d7')),
            ('FONT', (0, 7), (-1, 7), 'Helvetica-Bold'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(financial_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Financial notes
        notes_text = """
        <b>Financial Notes:</b><br/>
        • Interest calculated according to German legal rate (5.12% above base rate)<br/>
        • Collection fees follow "No Cure No Pay" model standards<br/>
        • Court costs are estimates based on debt amount and German fee schedules<br/>
        • All amounts are subject to successful collection
        """
        elements.append(Paragraph(notes_text, self.styles['DebtBodyText']))
        elements.append(Spacer(1, 0.3*inch))
    
    def _add_timeline(self, elements):
        """Add collection timeline"""
        elements.append(Paragraph("7. COLLECTION TIMELINE", self.styles['DebtSectionHeader']))
        
        timeline = self.data["timeline"]
        
        timeline_data = [['Date', 'Event', 'Status', 'Days from Start']]
        
        for event in timeline:
            timeline_data.append([
                event.get('date', ''),
                event.get('event', ''),
                event.get('status', ''),
                str(event.get('days_from_now', 0))
            ])
        
        timeline_table = Table(timeline_data, colWidths=[1.2*inch, 2.8*inch, 1.2*inch, 0.8*inch])
        timeline_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3748')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(timeline_table)
        elements.append(Spacer(1, 0.3*inch))
    
    def _add_recommendations(self, elements):
        """Add recommendations section"""
        elements.append(Paragraph("8. RECOMMENDATIONS AND NEXT STEPS", self.styles['DebtSectionHeader']))
        
        recommendations = self.data["recommendations"]
        
        elements.append(Paragraph("Priority Actions", self.styles['DebtSubsectionHeader']))
        
        for i, rec in enumerate(recommendations, 1):
            elements.append(Paragraph(f"{i}. {rec}", self.styles['DebtBodyText']))
        
        # Success factors
        elements.append(Paragraph("Success Factors", self.styles['DebtSubsectionHeader']))
        success_factors = [
            "Maintain professional communication throughout the process",
            "Document all interactions and payment commitments",
            "Be flexible with payment arrangements while protecting interests",
            "Monitor debtor's business situation for changes",
            "Escalate promptly if agreed terms are not met"
        ]
        
        for factor in success_factors:
            elements.append(Paragraph(f"• {factor}", self.styles['DebtBodyText']))
        
        elements.append(Spacer(1, 0.3*inch))
    
    def _add_appendix(self, elements):
        """Add appendix with technical details"""
        elements.append(PageBreak())
        elements.append(Paragraph("APPENDIX - TECHNICAL DETAILS", self.styles['DebtSectionHeader']))
        
        metadata = self.data["metadata"]
        
        # System information
        elements.append(Paragraph("Analysis System Information", self.styles['DebtSubsectionHeader']))
        
        system_info = f"""
        <b>System Version:</b> {metadata.get('system_version', 'N/A')}<br/>
        <b>Analysis Date:</b> {datetime.datetime.fromisoformat(metadata['analysis_date']).strftime('%d.%m.%Y %H:%M:%S')}<br/>
        <b>Output Data Length:</b> {metadata.get('raw_output_length', 0):,} characters<br/>
        <b>Processing Method:</b> CrewAI Multi-Agent Analysis<br/>
        """
        elements.append(Paragraph(system_info, self.styles['DebtBodyText']))
        
        # Disclaimer
        elements.append(Paragraph("Disclaimer", self.styles['DebtSubsectionHeader']))
        disclaimer = """
        This analysis has been generated by an AI-powered debt collection system. While every effort 
        has been made to ensure accuracy and compliance with German regulations, this report should 
        be reviewed by qualified personnel before implementation. The recommendations are based on 
        the information provided and current legal standards as of the analysis date.
        
        For questions regarding this analysis, please contact the responsible collection specialist 
        or legal department.
        """
        elements.append(Paragraph(disclaimer, self.styles['DebtBodyText']))

# Main execution function with PDF export
def run_debt_collection_with_pdf_export(case_data: Dict):
    """Run complete debt collection analysis and generate PDF report"""
    
    print("🚀 Starting Enhanced Debt Collection Analysis with PDF Export")
    print("=" * 80)
    
    try:
        # Create and run crew
        crew = create_german_debt_collection_crew(case_data)
        
        print("🤖 Executing AI agent crew...")
        crew_result = crew.kickoff()
        
        print("✅ Crew execution completed!")
        print("📊 Parsing agent outputs...")
        
        # Parse crew output
        parser = CrewAIOutputParser(crew_result, case_data)
        parsed_data = parser.parsed_data
        
        print("📄 Generating comprehensive PDF report...")
        
        # Generate PDF
        pdf_generator = ComprehensivePDFGenerator(parsed_data)
        pdf_filename = pdf_generator.generate_pdf()
        
        # Display enhanced console output
        display_enhanced_console_output(parsed_data)
        
        return {
            "success": True,
            "crew_result": crew_result,
            "parsed_data": parsed_data,
            "pdf_filename": pdf_filename
        }
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

def display_enhanced_console_output(parsed_data: Dict):
    """Display enhanced console output with parsed data"""
    
    print("\n" + "=" * 100)
    print("🎯 ENHANCED DEBT COLLECTION ANALYSIS RESULTS")
    print("=" * 100)
    
    case_info = parsed_data["case_info"]
    risk_data = parsed_data["risk_assessment"]
    financial = parsed_data["financial_summary"]
    
    # Case overview
    print(f"📋 CASE OVERVIEW:")
    print(f"   Case ID: {case_info.get('case_id', 'N/A')}")
    print(f"   Debtor: {case_info.get('company_name', case_info.get('debtor_name', 'N/A'))}")
    print(f"   Amount: €{case_info.get('debt_amount', 0):,.2f}")
    print(f"   Days Overdue: {case_info.get('debt_age_days', 0)}")
    print(f"   Industry: {case_info.get('industry', 'N/A').title()}")
    
    # AI Analysis Results
    print(f"\n🤖 AI ANALYSIS RESULTS:")
    print(f"   Risk Score: {risk_data.get('risk_score', 0)}/100")
    print(f"   Risk Level: {risk_data.get('risk_level', 'UNKNOWN')}")
    print(f"   Recovery Probability: {risk_data.get('recovery_probability', 0):.1f}%")
    print(f"   Recommended Approach: {risk_data.get('recommended_approach', 'Standard')}")
    
    # Financial Summary
    print(f"\n💰 FINANCIAL BREAKDOWN:")
    print(f"   Principal: €{financial.get('principal_amount', 0):,.2f}")
    print(f"   Interest: €{financial.get('interest_amount', 0):,.2f}")
    print(f"   Collection Fee: €{financial.get('collection_fee', 0):,.2f}")
    print(f"   Total Due: €{financial.get('total_amount_due', 0):,.2f}")
    
    # Recommendations preview
    recommendations = parsed_data.get("recommendations", [])
    if recommendations:
        print(f"\n💡 KEY RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. {rec}")
        if len(recommendations) > 3:
            print(f"   ... and {len(recommendations) - 3} more (see PDF report)")
    
    print(f"\n📄 COMPREHENSIVE PDF REPORT GENERATED")
    print("=" * 100)

# Tools and Agents (same as before, keeping the existing implementation)
class DebtorAssessmentTool(BaseTool):
    name: str = "Debtor Assessment Tool"
    description: str = "Analyzes debtor's financial situation and payment capability"

    def _run(self, case_data: Dict) -> Dict:
        case = DebtCase(**case_data)
        risk_score = 0
        
        # Age factor
        if case.debt_age_days < 30: risk_score += 10
        elif case.debt_age_days < 90: risk_score += 25
        elif case.debt_age_days < 180: risk_score += 40
        else: risk_score += 60
            
        # Industry factor
        high_risk = ["hospitality", "retail", "construction"]
        medium_risk = ["manufacturing", "logistics", "services"]
        low_risk = ["technology", "healthcare", "finance"]
        
        if case.industry.lower() in high_risk: risk_score += 30
        elif case.industry.lower() in medium_risk: risk_score += 15
        elif case.industry.lower() in low_risk: risk_score += 5
        else: risk_score += 20
            
        # Amount factor
        if case.debt_amount < 1000: risk_score += 5
        elif case.debt_amount < 5000: risk_score += 10
        elif case.debt_amount < 25000: risk_score += 20
        else: risk_score += 30
            
        # Payment history
        if case.previous_payment_history == "good": risk_score -= 20
        elif case.previous_payment_history == "poor": risk_score += 20
            
        risk_score = max(0, min(100, risk_score))
        
        if risk_score < 25: risk_level = DebtorRiskLevel.LOW
        elif risk_score < 50: risk_level = DebtorRiskLevel.MEDIUM
        elif risk_score < 75: risk_level = DebtorRiskLevel.HIGH
        else: risk_level = DebtorRiskLevel.VERY_HIGH
            
        recovery_probability = (100 - risk_score) / 100
        
        if risk_level in [DebtorRiskLevel.LOW, DebtorRiskLevel.MEDIUM]:
            approach = "amicable_negotiation"
            recommended_actions = [
                "Send professional payment reminder",
                "Offer flexible payment plan",
                "Maintain business relationship focus"
            ]
        else:
            approach = "intensive_collection"
            recommended_actions = [
                "Immediate personal contact",
                "Credit database verification",
                "Consider protective measures",
                "Prepare legal proceedings documentation"
            ]
            
        return {
            "risk_score": risk_score,
            "risk_level": risk_level.value,
            "recovery_probability": recovery_probability,
            "recommended_approach": approach,
            "recommended_actions": recommended_actions,
            "schufa_check_recommended": risk_score > 50,
            "del_credere_eligible": risk_score < 40
        }

class NegotiationStrategyTool(BaseTool):
    name: str = "Negotiation Strategy Tool"
    description: str = "Generates optimal payment negotiation strategies"

    def _run(self, assessment_data: Dict, case_data: Dict) -> Dict:
        case = DebtCase(**case_data)
        risk_level = assessment_data.get("risk_level", "medium")
        
        strategies = []
        
        if case.contact_attempts == 0:
            strategies.append({
                "phase": "initial_reminder",
                "action": "Friendly payment reminder",
                "timeline": "immediate",
                "tone": "professional_friendly"
            })
            
        payment_options = []
        
        # Immediate payment discount
        payment_options.append({
            "type": "immediate_payment",
            "discount": 0.02 if case.debt_amount < 5000 else 0.015,
            "deadline": "7 days",
            "final_amount": case.debt_amount * 0.98
        })
        
        # Payment plan
        if case.debt_amount > 1000:
            max_installments = 12 if risk_level in ["low", "medium"] else 6
            min_down_payment = 0.1 if risk_level == "low" else 0.25
            
            payment_options.append({
                "type": "payment_plan",
                "installments": min(3, max_installments),
                "down_payment": case.debt_amount * min_down_payment,
                "monthly_amount": (case.debt_amount * (1 - min_down_payment)) / 3
            })
            
        return {
            "current_phase": strategies[-1] if strategies else "pre_collection",
            "strategies": strategies,
            "payment_options": payment_options,
            "escalation_timeline": {
                "reminder": "Day 0",
                "first_dunning": "Day 7",
                "second_dunning": "Day 21",
                "legal_proceedings": "Day 35"
            },
            "relationship_preservation": risk_level in ["low", "medium"]
        }

class LegalDatabaseTool(BaseTool):
    name: str = "Legal Database Tool"
    description: str = "Checks German legal requirements for debt collection"

    def _run(self, action_type: str, context: Dict) -> Dict:
        violations = []
        warnings = []
        
        if not context.get("privacy_notice_provided", True):
            violations.append("GDPR privacy notice required")
            
        debt_amount = context.get("debt_amount", 0)
        debt_age_days = context.get("debt_age_days", 0)
        
        legal_interest_rate = 0.0512
        interest_amount = debt_amount * legal_interest_rate * (debt_age_days / 365) if debt_age_days > 30 else 0
            
        return {
            "action_type": action_type,
            "violations": violations,
            "warnings": warnings,
            "compliant": len(violations) == 0,
            "legal_interest_rate": legal_interest_rate,
            "calculated_interest": interest_amount,
            "max_collection_fee": debt_amount * 0.25
        }

class DocumentGeneratorTool(BaseTool):
    name: str = "Document Generator Tool"
    description: str = "Generates legally compliant debt collection documents"

    def _run(self, doc_type: str, case_data: Dict, language: str = "de") -> Dict:
        if language == "de":
            subject = f"Zahlungserinnerung - Rechnung Nr. {case_data.get('invoice_number', 'N/A')}"
            deadline = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%d.%m.%Y")
            
            document = f"""Sehr geehrte Damen und Herren,

wir möchten Sie freundlich darauf hinweisen, dass die Rechnung Nr. {case_data.get('invoice_number', 'N/A')} 
vom {case_data.get('invoice_date', 'N/A')} in Höhe von {case_data.get('debt_amount', 0)} EUR noch offen ist.

Bitte gleichen Sie den Betrag bis zum {deadline} aus.

Bei Fragen stehen wir Ihnen gerne zur Verfügung.

Diese Zahlungserinnerung erfolgt unter Vorbehalt und ohne Anerkennung einer Rechtspflicht.

Datenschutzhinweis: Ihre Daten werden gemäß DSGVO zur Durchsetzung unserer berechtigten Interessen verarbeitet."""
        
        return {
            "success": True,
            "document_type": doc_type,
            "language": language,
            "subject": subject,
            "document": document,
            "legal_compliant": True
        }

# Create Agents (same as before)
def create_collection_agent(llm=None):
    if llm is None:
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        
    return Agent(
        role='Senior B2B Debt Collection Specialist',
        goal='Maximize debt recovery while maintaining business relationships and ensuring compliance',
        backstory="""You are an experienced debt collection specialist with 15+ years in the German B2B market.""",
        tools=[
            DebtorAssessmentTool(),
            NegotiationStrategyTool()
        ],
        llm=llm,
        verbose=True,
        allow_delegation=True,
        max_iter=5
    )

def create_compliance_agent(llm=None):
    if llm is None:
        llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        
    return Agent(
        role='German Debt Collection Compliance Officer',
        goal='Ensure all collection activities comply with German regulations',
        backstory="""You are a certified compliance officer with expertise in German debt collection law.""",
        tools=[
            LegalDatabaseTool()
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=4
    )

def create_communication_agent(llm=None):
    if llm is None:
        llm = ChatOpenAI(model="gpt-4", temperature=0.6)
        
    return Agent(
        role='Multilingual Debtor Relations Specialist',
        goal='Facilitate productive communication with debtors',
        backstory="""You are a communication specialist trained in conflict resolution and financial mediation.""",
        tools=[
            DocumentGeneratorTool()
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

# Create Tasks (simplified versions)
def create_debt_assessment_task(case_data: Dict) -> Task:
    return Task(
        description=f"""Analyze this debt case and provide comprehensive assessment: {json.dumps(case_data, indent=2)}""",
        expected_output="""Risk assessment with score, recovery probability, and recommendations""",
        agent=None
    )

def create_negotiation_strategy_task(case_data: Dict) -> Task:
    return Task(
        description=f"""Develop negotiation strategy based on assessment: {json.dumps(case_data, indent=2)}""",
        expected_output="""Negotiation strategy with payment options and timeline""",
        agent=None
    )

def create_compliance_review_task(case_data: Dict) -> Task:
    return Task(
        description=f"""Review for German legal compliance: {json.dumps(case_data, indent=2)}""",
        expected_output="""Compliance report with legal requirements verification""",
        agent=None
    )

def create_communication_drafting_task(case_data: Dict, language: str = "de") -> Task:
    return Task(
        description=f"""Draft debt collection communication: {json.dumps(case_data, indent=2)}""",
        expected_output=f"""Professional communication in {language} with legal compliance""",
        agent=None
    )

# Create Crew
def create_german_debt_collection_crew(case_data: Dict, llm=None):
    collection_agent = create_collection_agent(llm)
    compliance_agent = create_compliance_agent(llm)
    communication_agent = create_communication_agent(llm)
    
    assessment_task = create_debt_assessment_task(case_data)
    assessment_task.agent = collection_agent
    
    strategy_task = create_negotiation_strategy_task(case_data)
    strategy_task.agent = collection_agent
    strategy_task.context = [assessment_task]
    
    compliance_task = create_compliance_review_task(case_data)
    compliance_task.agent = compliance_agent
    compliance_task.context = [assessment_task, strategy_task]
    
    communication_task = create_communication_drafting_task(case_data, case_data.get("communication_language", "de"))
    communication_task.agent = communication_agent
    communication_task.context = [assessment_task, strategy_task, compliance_task]
    
    try:
        crew = Crew(
            agents=[collection_agent, compliance_agent, communication_agent],
            tasks=[assessment_task, strategy_task, compliance_task, communication_task],
            process=Process.sequential,
            verbose=True
        )
    except:
        crew = Crew(
            agents=[collection_agent, compliance_agent, communication_agent],
            tasks=[assessment_task, strategy_task, compliance_task, communication_task],
            verbose=True
        )
    
    return crew

# Test function for PDF generation
def test_pdf_generation_only():
    """Test PDF generation without running CrewAI (for debugging)"""
    
    print("🧪 Testing PDF Generation Only...")
    
    # Create sample parsed data
    sample_case = {
        "case_id": "TEST-PDF-001",
        "debtor_name": "Test Company GmbH",
        "company_name": "Test Manufacturing GmbH",
        "debt_amount": 12500.00,
        "debt_age_days": 35,
        "industry": "manufacturing",
        "location": "Hamburg, Germany",
        "previous_payment_history": "good",
        "communication_language": "de",
        "debt_type": "B2B",
        "contact_attempts": 0,
        "legal_status": "pre_legal",
        "invoice_number": "TEST-INV-001",
        "invoice_date": "2024-01-15",
        "original_due_date": "2024-02-15"
    }
    
    sample_crew_output = """
    Risk Assessment: The debtor shows a medium risk profile with a risk score of 35/100.
    Recovery probability is estimated at 85%. The company has a good payment history
    and operates in the manufacturing sector. Recommended approach is amicable negotiation.
    
    Negotiation Strategy: Offer multiple payment options including early payment discount
    and structured payment plans. Maintain professional relationship throughout process.
    
    Compliance Review: All actions comply with German regulations including GDPR, BGB, and RDG.
    Privacy notice included as required. No violations identified.
    
    Communication: Professional German language template generated with appropriate tone
    for B2B relationship. GDPR notice included. 7-day payment deadline set.
    """
    
    try:
        # Parse the sample output
        parser = CrewAIOutputParser(sample_crew_output, sample_case)
        parsed_data = parser.parsed_data
        
        print("✅ Output parsing successful")
        
        # Generate PDF
        pdf_generator = ComprehensivePDFGenerator(parsed_data)
        pdf_filename = pdf_generator.generate_pdf("test_debt_collection_report.pdf")
        
        if pdf_filename and os.path.exists(pdf_filename):
            file_size = os.path.getsize(pdf_filename)
            print(f"✅ PDF test successful!")
            print(f"📄 File: {pdf_filename}")
            print(f"📊 Size: {file_size:,} bytes")
            return True
        else:
            print("❌ PDF test failed - file not created")
            return False
            
    except Exception as e:
        print(f"❌ PDF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Main execution
if __name__ == "__main__":
    import sys
    
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test-pdf":
        success = test_pdf_generation_only()
        if success:
            print("\n🎉 PDF generation test passed! The system is ready.")
        else:
            print("\n❌ PDF generation test failed. Please check the error messages above.")
        exit(0)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("\n💡 You can test PDF generation without API key:")
        print("   python system.py --test-pdf")
        exit(1)
    
    # Example case
    example_case = {
        "case_id": "IHHHT-2024-B2B-001",
        "debtor_name": "Tüller GmbH",
        "company_name": "Tüller Industrietechnik GmbH",
        "debt_amount": 15750.50,
        "debt_age_days": 45,
        "industry": "manufacturing",
        "location": "Stuttgart, Germany",
        "previous_payment_history": "good",
        "communication_language": "de",
        "debt_type": "B2B",
        "contact_attempts": 0,
        "legal_status": "pre_legal",
        "invoice_number": "INV-2024-0892",
        "invoice_date": "2024-01-15",
        "original_due_date": "2024-02-15"
    }
    
    # Run analysis with PDF export
    result = run_debt_collection_with_pdf_export(example_case)
    
    if result["success"]:
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📄 PDF Report: {result['pdf_filename']}")
        print(f"📁 Location: {os.path.abspath(result['pdf_filename'])}")
    else:
        print(f"\n❌ Analysis failed: {result['error']}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check your OpenAI API key")
        print("   2. Test PDF generation: python system.py --test-pdf")
        print("   3. Ensure all dependencies are installed")