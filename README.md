# Mental Health A2A Agent Ecosystem

A comprehensive mental health support system leveraging the Agent-to-Agent (A2A) protocol to create a collaborative ecosystem of specialized AI agents.

## 🌟 Overview

This innovative project addresses the critical shortage of mental health professionals by creating five distinct but interconnected agents that communicate seamlessly to deliver personalized therapeutic interventions:

- **Primary Screening Agent**: Conducts intake assessments using validated clinical tools (PHQ-9, GAD-7)
- **Crisis Detection Agent**: Monitors for high-risk content and provides emergency intervention
- **Therapeutic Intervention Agent**: Delivers evidence-based mental health treatments (CBT, ACT)
- **Care Coordination Agent**: Bridges AI system with traditional healthcare infrastructure
- **Progress Analytics Agent**: Monitors treatment effectiveness and predicts outcomes

## ✨ Key Features

- **Multi-modal Input Support**: Text, voice, documents, images, and sensor data
- **Evidence-based Assessments**: PHQ-9, GAD-7, and other validated clinical tools
- **Crisis Intervention**: Real-time risk detection and emergency protocols
- **HIPAA Compliance**: End-to-end encryption and comprehensive data protection
- **A2A Protocol**: Seamless agent-to-agent communication and collaboration
- **Real-time Monitoring**: Continuous assessment and intervention capabilities

## 🏗️ Architecture

The system is built on the A2A protocol, enabling agents to:
- Discover each other's capabilities through Agent Cards
- Negotiate interaction methods and data formats
- Collaborate securely on complex cases
- Maintain privacy while sharing relevant context
- Handle crisis situations with immediate response protocols

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd mental-health-a2a-ecosystem
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
# Add your OpenAI API key
echo "OPENAI_API_KEY=your-api-key-here" >> .env
```

4. **Start the system:**
```bash
python start.py
```

5. **Access the application:**
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Admin Interface: http://localhost:8000/redoc

## 📁 Project Structure

```
mental-health-a2a-ecosystem/
├── agents/                          # Individual agent implementations
│   ├── primary-screening-agent/     # Initial assessment and screening
│   ├── crisis-detection-agent/      # Crisis monitoring and intervention
│   ├── therapeutic-intervention-agent/ # Therapy delivery and treatment
│   ├── care-coordination-agent/     # Healthcare system integration
│   └── progress-analytics-agent/    # Treatment monitoring and analytics
├── shared/                          # Common utilities and A2A protocol
│   └── a2a-protocol/               # A2A protocol implementation
│       ├── communication_layer.py  # Agent communication
│       ├── agent_discovery.py      # Agent discovery and capabilities
│       ├── task_management.py      # Task lifecycle management
│       └── security.py             # Security and authentication
├── frontend/                        # User interfaces
│   └── index.html                  # Web interface
├── main.py                         # FastAPI application
├── start.py                        # Startup script
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Database Configuration
DATABASE_URL=sqlite:///./mental_health.db

# Security
SECRET_KEY=your-secret-key-here

# A2A Protocol
A2A_BASE_URL=http://localhost:8000
```

### Agent Configuration

Each agent can be configured through the `config.py` file:

- **Crisis Detection Thresholds**: Adjust sensitivity for crisis detection
- **Assessment Thresholds**: Customize PHQ-9 and GAD-7 scoring
- **Response Timeouts**: Set maximum response times for different operations
- **Security Settings**: Configure encryption and access controls

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/security/
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=agents --cov=shared --cov-report=html
```

## 🚨 Safety and Ethics

This system incorporates comprehensive safety measures:

- **Human Oversight**: All crisis situations require human intervention
- **Crisis Intervention**: Immediate response protocols for high-risk situations
- **Bias Detection**: Regular monitoring for algorithmic bias
- **Clinical Validation**: Evidence-based assessment tools and interventions
- **Transparent Decision-Making**: Clear audit trails for all decisions
- **HIPAA Compliance**: End-to-end encryption and data protection

## 📊 API Usage

### Start a Mental Health Screening

```bash
curl -X POST "http://localhost:8000/screening/start" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-token" \
  -d '{
    "user_id": "user-001",
    "input_data": {
      "text": "I have been feeling really down lately and having trouble sleeping",
      "input_type": "text"
    }
  }'
```

### Analyze for Crisis Indicators

```bash
curl -X POST "http://localhost:8000/crisis/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-token" \
  -d '{
    "user_id": "user-001",
    "session_id": "session-123",
    "interaction_data": {
      "text": "I dont want to live anymore"
    }
  }'
```

## 🔒 Security Considerations

- **Data Encryption**: All sensitive data is encrypted at rest and in transit
- **Access Control**: Role-based access control for different user types
- **Audit Logging**: Comprehensive logging of all system activities
- **Privacy Protection**: Minimal data collection and automatic data purging
- **Secure Communication**: All agent-to-agent communication is encrypted

## 🤝 Contributing

We welcome contributions! Please ensure:

1. All changes maintain HIPAA compliance
2. Clinical safety standards are upheld
3. Code follows the established patterns
4. Tests are included for new features
5. Documentation is updated accordingly

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run linting
black .
flake8 .
mypy .
```

## 📈 Roadmap

- [ ] Additional assessment tools (PCL-5, AUDIT, etc.)
- [ ] Integration with Electronic Health Records (EHR)
- [ ] Mobile application interface
- [ ] Advanced analytics and reporting
- [ ] Multi-language support
- [ ] Integration with wearable devices

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- Create an issue in the GitHub repository
- Contact the development team
- Review the documentation at `/docs`

## ⚠️ Disclaimer

This system is designed for research and educational purposes. It should not replace professional mental health care. Always consult with qualified mental health professionals for clinical decisions.

---

**Built with ❤️ for better mental health care**