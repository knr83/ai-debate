# AI Debate System

A sophisticated AI-powered debate system that orchestrates debates between multiple AI models, with intelligent judgment and cost tracking capabilities.

## 🚀 Features

- **Multi-Model Debates**: Conduct debates between different AI models (GPT-4, GPT-5, etc.)
- **Intelligent Judging**: Automated judgment system to evaluate debate quality
- **Cost Tracking**: Real-time cost estimation and tracking for API usage
- **Configurable Rounds**: Adjustable debate rounds (1-3) for different complexity levels
- **Token Management**: Configurable token limits (100-4000) for response control
- **Modern Web Interface**: Built with Gradio for an intuitive user experience
- **Environment Configuration**: Secure API key management via .env files

## 🛠️ Technology Stack

- **Python 3.11+**: Core programming language
- **OpenAI API**: Integration with various GPT models
- **Gradio**: Modern web interface framework
- **Python-dotenv**: Environment variable management
- **Logging**: Comprehensive application logging

## 📋 Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Virtual environment (recommended)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-debate
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\Activate.ps1
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🎯 Usage

1. **Start the application**
   ```bash
   python main.py
   ```

2. **Access the web interface**
   Open your browser and navigate to `http://localhost:7860`

3. **Configure debate settings**
   - Enter your debate question
   - Select AI models for debaters and judge
   - Adjust token limits and debate rounds
   - Set cost limits if desired

4. **Launch the debate**
   Click "Start Debate" to begin the AI-powered discussion

## 🔧 Configuration

### Available Models
- **GPT-5 Models**: `gpt-5-mini`, `gpt-5-nano`
- **GPT-4 Models**: `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`, `gpt-4`
- **GPT-3.5 Models**: `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`

### Cost Structure
The system tracks costs based on OpenAI's pricing:
- Input tokens: $0.0015 - $0.30 per 1M tokens
- Output tokens: $0.002 - $2.00 per 1M tokens

### Debate Settings
- **Rounds**: 1-3 debate rounds
- **Token Limits**: 100-4000 tokens per response
- **Cost Limits**: Optional budget constraints

## 📁 Project Structure

```
ai-debate/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── .env                # Environment variables (create this)
└── venv/               # Virtual environment
```

## 🔍 Core Functions

- **`enhanced_debate_function`**: Main debate orchestration
- **`validate_inputs`**: Input validation and sanitization
- **`estimate_cost_for_question`**: Cost estimation before debate
- **`call_openai_model`**: OpenAI API integration
- **`create_interface`**: Gradio web interface setup

## 🚨 Error Handling

The system includes comprehensive error handling for:
- Invalid API keys
- Network connectivity issues
- Input validation errors
- Cost limit exceeded scenarios
- Model availability issues

## 📊 Logging

Comprehensive logging is implemented for:
- API requests and responses
- Cost tracking
- Error conditions
- User interactions
- System performance metrics

## 🔒 Security

- API keys are stored in environment variables
- Input sanitization prevents injection attacks
- Rate limiting considerations for API usage
- Secure token handling

## 🧪 Testing

To test the system:
1. Ensure all dependencies are installed
2. Verify your OpenAI API key is valid
3. Start with simple questions to test functionality
4. Monitor the logs for any issues

## 🚀 Deployment

The application can be deployed using:
- **Local Development**: Direct Python execution
- **Docker**: Containerized deployment (Dockerfile not included)
- **Cloud Platforms**: Deploy to Heroku, AWS, or similar platforms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the logs for error messages
2. Verify your API key and configuration
3. Ensure all dependencies are properly installed
4. Check OpenAI API status and quotas

## 🔮 Future Enhancements

- **Multi-language Support**: Support for debates in different languages
- **Advanced Analytics**: Detailed debate analysis and insights
- **Custom Model Integration**: Support for other AI providers
- **Debate Templates**: Pre-configured debate scenarios
- **Export Functionality**: Save debates in various formats
- **Real-time Collaboration**: Multi-user debate participation

---

**Note**: This system requires an active OpenAI API key and internet connectivity to function properly.
