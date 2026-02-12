# AI/ML Programmers Handbook

Documentation site for Artificial Intelligence and Machine Learning resources.

## Local Development

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MorariuT/MorariuT.github.io.git
   cd MorariuT.github.io/AI_Docs
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the development server**
   ```bash
   mkdocs serve
   ```

5. **Visit the site**
   Open your browser and go to `http://localhost:8000`

## Building for Production

To build the static site:

```bash
mkdocs build
```

This will create a `site/` folder with the static HTML files.

## Project Structure

```
AI_Docs/
├── docs/                    # Documentation markdown files
│   ├── index.md            # Home page
│   ├── getting-started.md  # Getting started guide
│   ├── Basics/             # Fundamental concepts
│   ├── linear_models/      # Linear regression, logistic regression
│   ├── NeuralNetworks/     # Neural network concepts
│   ├── ComputerVision/     # CV techniques
│   ├── NLP/                # Natural language processing
│   ├── ensemble_models/    # Ensemble methods
│   └── DataAnalisys/       # Data analysis techniques
├── code/                   # Jupyter notebooks and code examples
│   └── LossFunctions/
├── _config.yml            # MkDocs configuration
├── requirements.txt       # Python dependencies
└── .gitignore            # Git ignore file
```

## Deployment

The site is automatically deployed to GitHub Pages when you push to the `main` branch. The workflow:

1. GitHub Actions detects a push to the `main` branch
2. It builds the documentation using MkDocs
3. The generated `site/` folder is deployed to GitHub Pages

**Live site:** https://MorariuT.github.io/AI_Docs/

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Test locally with `mkdocs serve`
5. Submit a pull request

## Documentation Guidelines

- Use Markdown formatting
- Include LaTeX equations using `$...$` for inline and `$$...$$` for blocks
- Add code examples where applicable
- Include links to relevant papers or resources
- Follow the existing structure and style

## License

[Add your license here]

## Contact

For questions or suggestions, please open an issue on GitHub: [MorariuT/MorariuT.github.io](https://github.com/MorariuT/MorariuT.github.io/issues)
