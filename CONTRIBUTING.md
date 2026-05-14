# 🤝 Contributing to Real-Time Indic Sign Language to Speech Translator

Thank you for your interest in contributing to the **Real-Time Indic Sign Language to Speech Translator**! This project aims to bridge communication gaps for the Deaf and Hard-of-Hearing community by translating Indian Sign Language (ISL) gestures into spoken sentences in real-time.

We welcome contributions from developers, researchers, and the community. Whether it's fixing bugs, adding features, improving documentation, or expanding language support, your help is valuable.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Community](#community)

## 📜 Code of Conduct

This project adheres to a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other community members

## 🚀 How to Contribute

### Types of Contributions

- **🐛 Bug Fixes**: Identify and fix issues in the codebase
- **✨ New Features**: Implement new functionality or improve existing ones
- **📚 Documentation**: Improve README, add examples, or create tutorials
- **🧪 Testing**: Add or improve test cases
- **🌐 Localization**: Add support for more Indian languages
- **🎨 UI/UX**: Enhance the web interface or desktop scripts
- **🔧 Tooling**: Improve build scripts, CI/CD, or development tools

### Getting Started

1. **Fork the Repository**: Click the "Fork" button on GitHub
2. **Clone Your Fork**: `git clone https://github.com/your-username/Real-Time-Indic-Sign-Language-to-Speech-Translator.git`
3. **Create a Branch**: `git checkout -b feature/your-feature-name`
4. **Make Changes**: Implement your contribution
5. **Test Locally**: Ensure everything works as expected
6. **Commit Changes**: `git commit -m "Add your descriptive commit message"`
7. **Push to Your Fork**: `git push origin feature/your-feature-name`
8. **Create a Pull Request**: Submit a PR with a clear description

## 🛠 Development Setup

### Prerequisites

- Python 3.12 or higher
- Git
- A webcam for testing
- Windows/macOS/Linux

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Real-Time-Indic-Sign-Language-to-Speech-Translator.git
   cd Real-Time-Indic-Sign-Language-to-Speech-Translator
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (optional for basic functionality)
   ```

5. **Run the application**:
   ```bash
   python app.py
   # Visit http://localhost:5000
   ```

### Development Scripts

- `scripts/personal_collector.py`: Record training data
- `scripts/train_personal_only.py`: Train model on personal data
- `scripts/word_tester.py`: Test sign recognition
- `scripts/evaluation_report.py`: Evaluate model performance

## 💻 Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused on single responsibilities

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (e.g., "Add", "Fix", "Update")
- Reference issue numbers when applicable: `Fix #123: Handle edge case in landmark extraction`

### Code Structure

- Keep the main application logic in `app.py`
- Place utility scripts in the `scripts/` directory
- Store trained models in `models/`
- Use the `data/` directory for datasets and features

## 🧪 Testing

### Running Tests

```bash
# Run basic functionality tests
python -m pytest tests/  # If test files exist

# Test specific components
python scripts/word_tester.py
python scripts/evaluation_report.py
```

### Testing Guidelines

- Test your changes thoroughly before submitting
- Ensure the web app works with different browsers
- Test with various webcam setups if possible
- Verify that existing functionality isn't broken

## 📤 Submitting Changes

### Pull Request Process

1. **Ensure your branch is up-to-date**:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Create a Pull Request**:
   - Provide a clear title and description
   - Reference any related issues
   - Include screenshots for UI changes
   - Explain the problem solved and approach taken

3. **Review Process**:
   - Maintainers will review your PR
   - Address any feedback or requested changes
   - Once approved, your changes will be merged

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes and why they're needed

## 🐛 Reporting Issues

Found a bug? We'd love to hear about it!

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear title describing the problem
   - Steps to reproduce
   - Expected vs. actual behavior
   - Screenshots if applicable
   - Your environment (OS, Python version, etc.)

## 💡 Feature Requests

Have an idea to improve the project?

1. **Check existing issues** for similar requests
2. **Create a feature request** with:
   - Clear description of the proposed feature
   - Use case or problem it solves
   - Any implementation ideas you have

## 🌍 Community

- **Discussions**: Use GitHub Discussions for questions and ideas
- **Issues**: Report bugs and request features
- **Pull Requests**: Submit code changes
- **Email**: Contact maintainers for sensitive matters

## 🙏 Recognition

Contributors will be acknowledged in the README and project documentation. Significant contributions may lead to co-authorship on related publications.

Thank you for contributing to making communication more accessible! 🇮🇳🤟