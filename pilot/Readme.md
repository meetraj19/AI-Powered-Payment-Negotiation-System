# CrewAI Payment Negotiation Multi-Agent System

A sophisticated multi-agent system for automated payment negotiation between creditors and debtors, built using CrewAI framework. The system incorporates AI-powered agents for negotiation, mediation, compliance checking, market analysis, and comprehensive PDF report generation.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd payment-negotiation-system
python setup.py

# 2. Set API key
export OPENAI_API_KEY='sk-your-api-key-here'

# 3. Run demo
python example_usage.py
```

For detailed setup instructions, see the [Installation](#installation) section or check out [QUICKSTART.md](QUICKSTART.md).

## 🌟 Features

- **Automated Negotiation**: AI agents representing creditors and debtors negotiate optimal settlement terms
- **Professional Mediation**: Mediator agent intervenes to resolve impasses and suggest creative solutions
- **Regulatory Compliance**: Real-time compliance checking against FDCPA, TCPA, GDPR, and other regulations
- **Market Analysis**: Evaluate settlements for fair market value and tradability
- **Comprehensive Reporting**: Generate detailed PDF reports documenting the entire negotiation process
- **Machine Learning Integration**: Agents use ML models for decision-making and pattern recognition

## 📋 Table of Contents

- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Agents Overview](#agents-overview)
- [Output Files](#output-files)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CrewAI Orchestrator                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  Creditor   │  │   Debtor    │  │  Mediator   │       │
│  │   Agent     │  │   Agent     │  │   Agent     │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Regulatory  │  │Market Maker │  │PDF Generator│       │
│  │   Agent     │  │   Agent     │  │   Agent     │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- All agent module files (provided separately):
  - `creditor.py`
  - `debitor.py`
  - `mediator.py`
  - `market_maker.py`
  - `regulatory.py`
  - `orchestrator.py`

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd payment-negotiation-system
```

2. **Run the setup script**
```bash
python setup.py
```
This will:
- Check Python version (3.8+ required)
- Create necessary directories
- Verify all required files are present
- Install missing dependencies
- Check API key configuration

3. **Set up your OpenAI API key**
```bash
export OPENAI_API_KEY='sk-your-api-key-here'  # Linux/Mac
# or
set OPENAI_API_KEY=sk-your-api-key-here  # Windows
```

4. **Run the example**
```bash
python example_usage.py
```

## ⚙️ Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### System Configuration
The system uses default configurations that can be modified in the code:
```python
{
    "max_workers": 10,
    "max_concurrent_negotiations": 100,
    "enable_mediation": True,
    "enable_market_making": True,
    "compliance_mode": "strict"
}
```

For advanced configuration options, see [config.example.json](config.example.json) for a complete template of customizable settings.

## 💻 Usage

### Basic Example

```python
from crewai_payment_negotiation import run_payment_negotiation
from debitor import FinancialProfile

# Create debtor financial profile
profile = FinancialProfile(
    monthly_income=5000,
    monthly_obligations=3000,
    liquid_assets=10000,
    credit_score=650,
    debt_to_income=0.4,
    employment_status="stable",
    hardship_factors=["medical_expenses"]
)

# Run negotiation
result = run_payment_negotiation(
    creditor_id="CRED_001",
    debtor_id="DEBT_001",
    initial_debt=50000,
    financial_profile=profile
)

if result['success']:
    print(f"Settlement reached: ${result['negotiation_result']['final_amount']:,.2f}")
    print(f"Report saved to: {result['report_filename']}")
```

### Command Line Usage

```bash
python crewai_payment_negotiation.py
```

This will run a demo negotiation with example data.

### Interactive Demo

```bash
python example_usage.py
```

This provides an interactive demo where you can:
- Choose from different debtor profiles
- Set custom debt amounts
- See real-time negotiation progress
- Get detailed analysis and reports

## 📊 Example Output

A successful negotiation produces:

```
✅ Negotiation completed successfully!

📊 Settlement Summary:
   Original Debt: $50,000.00
   Settlement Amount: $34,850.91
   Settlement Rate: 69.7%
   Savings: $15,149.09

💳 Payment Terms:
   Timeline: 120 days
   Interest Rate: 6.0%
   Installments: 4
   Monthly Payment: $8,712.73

✔️  Compliance Status: Compliant

📈 Market Analysis:
   Risk Rating: A
   Fair Market Value: $31,365.82
   Tradeable: Yes

📄 Reports Generated:
   PDF Report: negotiation_CRED_001_DEBT_001_20241210_143022.pdf
   Crew Output: crew_output_CRED_001_DEBT_001_20241210_143022.txt
```

## 👥 Agents Overview

### 1. **Creditor Agent**
- **Role**: Senior Creditor Representative
- **Goal**: Negotiate optimal debt recovery while maintaining compliance
- **Strategies**: Adaptive negotiation based on debtor profile
- **Tools**: `creditor_negotiation_tool`

### 2. **Debtor Agent**
- **Role**: Debtor Advocate
- **Goal**: Negotiate fair and affordable settlement terms
- **Capabilities**: Analyzes financial capacity and generates counter-offers
- **Tools**: `debtor_negotiation_tool`

### 3. **Mediator Agent**
- **Role**: Professional Mediator
- **Goal**: Facilitate productive negotiations and resolve impasses
- **Strategies**: Facilitative, evaluative, transformative, and directive mediation
- **Tools**: `mediation_tool`

### 4. **Regulatory Agent**
- **Role**: Compliance Officer
- **Goal**: Ensure all negotiations comply with relevant laws
- **Frameworks**: FDCPA, TCPA, GDPR, FCRA, Basel III
- **Tools**: `compliance_check_tool`

### 5. **Market Maker Agent**
- **Role**: Market Analyst
- **Goal**: Evaluate settlements for market tradability
- **Capabilities**: Risk assessment, liquidity analysis, fair value calculation
- **Tools**: `market_analysis_tool`

### 6. **PDF Generator Agent**
- **Role**: Report Generator Specialist
- **Goal**: Create comprehensive PDF reports
- **Output**: Professional PDF documentation of negotiations

## 📁 Output Files

The system generates two main output files:

### 1. **PDF Report** (`negotiation_CRED_XXX_DEBT_XXX_YYYYMMDD_HHMMSS.pdf`)
Contains:
- Executive summary
- Negotiation process analysis from CrewAI
- Debtor financial profile
- Complete negotiation history
- Compliance assessment
- Market analysis
- Final agreement terms

### 2. **Crew Output** (`crew_output_CRED_XXX_DEBT_XXX_YYYYMMDD_HHMMSS.txt`)
Contains:
- Raw output from the CrewAI agents
- Detailed analysis and reasoning
- Step-by-step negotiation progress

## 📂 File Structure

```
payment-negotiation-system/
├── Core System Files
│   ├── crewai_payment_negotiation.py  # Main CrewAI implementation
│   ├── creditor.py                    # Creditor agent module
│   ├── debitor.py                     # Debtor agent module  
│   ├── mediator.py                    # Mediator agent module
│   ├── market_maker.py                # Market maker agent module
│   ├── regulatory.py                  # Regulatory agent module
│   └── orchestrator.py                # System orchestrator module
├── Setup & Examples
│   ├── setup.py                       # Setup and validation script
│   ├── example_usage.py               # Interactive demo script
│   └── config.example.json            # Configuration template
├── Documentation
│   ├── README.md                      # This file
│   ├── QUICKSTART.md                  # Quick start guide
│   └── CHANGELOG.md                   # Version history
├── Configuration
│   ├── requirements.txt               # Python dependencies
│   ├── .gitignore                     # Git ignore file
│   └── .env.example                   # Example environment variables
└── Output (auto-generated)
    ├── output/                        # Generated reports directory
    │   ├── negotiation_*.pdf          # PDF reports
    │   └── crew_output_*.txt          # Raw CrewAI outputs
    └── logs/                          # System logs
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Error: Module not found**
   - Ensure all agent module files are in the same directory
   - Check that all dependencies are installed: `pip install -r requirements.txt`

2. **OpenAI API Key Error**
   - Verify your API key is set correctly
   - Check that your API key has sufficient credits

3. **Tool Execution Failures**
   - The system includes robust error handling and will retry failed tool calls
   - Check the console output for specific error messages

4. **PDF Generation Issues**
   - Ensure ReportLab is properly installed: `pip install reportlab`
   - Check write permissions in the output directory

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📈 System Capabilities

### What the System Can Do
- **Automated Negotiation**: Handle complex multi-round negotiations
- **Adaptive Strategies**: Adjust tactics based on debtor profiles
- **Compliance Checking**: Real-time regulatory compliance validation
- **Market Valuation**: Assess fair market value of settlements
- **Comprehensive Reporting**: Generate detailed PDF documentation
- **Multiple Scenarios**: Handle various financial situations and hardships

### Current Limitations
- Requires OpenAI API access and credits
- English language only (currently)
- Simplified regulatory framework (US-focused)
- No real-time market data integration
- No actual payment processing

## 🔒 Security Considerations

- Never commit API keys to version control
- Use environment variables for sensitive data
- Sanitize all user inputs
- Ensure compliance with data protection laws
- Implement access controls in production

## 📄 License

This project is provided as-is for educational and commercial use. Please ensure compliance with OpenAI's usage policies and applicable financial regulations in your jurisdiction.

## 🙏 Acknowledgments

- Built with [CrewAI](https://github.com/joaomdmoura/crewAI) framework
- Powered by OpenAI's GPT-4
- PDF generation using ReportLab
- Community contributions and feedback

## 📞 Support

For issues, questions, or contributions, please:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting guide above
- Consult the [QUICKSTART.md](QUICKSTART.md) for quick solutions

## 📚 Additional Resources

- [QUICKSTART.md](QUICKSTART.md) - Get started in 5 minutes
- [CHANGELOG.md](CHANGELOG.md) - Version history and updates
- [example_usage.py](example_usage.py) - Interactive demo script
- [setup.py](setup.py) - Automated setup and validation

---

**⚠️ Disclaimer**: This system is designed for demonstration and educational purposes. In production use, ensure compliance with all applicable financial regulations, data protection laws, and ethical guidelines for automated negotiation systems. Always consult with legal and financial professionals before deploying in a production environment.

**🔐 Privacy Note**: This system processes sensitive financial information. Ensure proper data handling, encryption, and compliance with relevant privacy laws (GDPR, CCPA, etc.) when handling real customer data.ature`)
5. Open a Pull Request

## 📄 License

This project is provided as-is for educational and commercial use. Please ensure compliance with OpenAI's usage policies and applicable financial regulations in your jurisdiction.

## 🙏 Acknowledgments

- Built with [CrewAI](https://github.com/joaomdmoura/crewAI) framework
- Powered by OpenAI's GPT-4
- PDF generation using ReportLab

## 📞 Support

For issues, questions, or contributions, please:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting guide above

---

**Note**: This system is designed for demonstration and educational purposes. In production use, ensure compliance with all applicable financial regulations and data protection laws.