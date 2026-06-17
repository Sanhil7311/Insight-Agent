# 📊 Insight-Agent: Autonomous Data Analytics Consultant

**Insight-Agent** is an enterprise-grade, locally hosted AI data analytics platform that acts as an autonomous data science consultant. Built with **Python**, **Streamlit**, and **CrewAI**, it automates the entire data analysis lifecycle—from raw data ingestion and deterministic cleaning to machine learning (AutoML), business insight generation, and dynamic report exporting.

Because it relies on **Ollama** for local Large Language Model (LLM) inference, your data never leaves your machine, ensuring **100% privacy and security**.

---

## ✨ Key Features

- **Multi-Agent AI Orchestration:** Powered by CrewAI, the system uses a team of specialized, autonomous AI agents (Cleaner, Analyst, ML Engineer, Storyteller, and Slides Formatter) to collaboratively analyze your data.
- **Secure & Flexible Data Ingestion:** Drag-and-drop CSV/Excel files or connect directly to a PostgreSQL database via the built-in SQL connection dashboard.
- **Automated Data Cleaning & EDA:** Automatically profiles data, handles missing values, clips outliers, detects datetime columns, and generates descriptive statistics and correlations.
- **AutoML (Predictive Modelling):** Automatically trains baseline `RandomForest` classification or regression models, providing performance metrics and extracting top feature importances.
- **Interactive Data Chat:** Ask questions directly to your data using a LangChain-powered conversational interface (`qwen2.5-coder`).
- **Enterprise-Ready Exports:** Instantly generate comprehensive **PDF reports** or download automatically formatted **PowerPoint (.pptx)** slide decks complete with embedded Plotly charts and executive bullet points.

---

## 🧠 The Agentic Workflow

Instead of a single prompt, Insight-Agent orchestrates a sequential pipeline of specialized AI personas:

1. **🧹 Data Cleaner**
   - Analyzes the dataset profile and prescribes deterministic cleaning strategies.

2. **📊 ML Analyst**
   - Interprets descriptive statistics, distributions, and correlations to recommend specific business charts.

3. **🤖 Model Engineer (Optional)**
   - Interprets the AutoML results, explaining model performance and feature drivers in plain English.

4. **✍️ Business Storyteller**
   - Synthesizes the findings into a polished, board-level executive Markdown report.

5. **🖼️ Presentation Formatter**
   - Distills the lengthy report into a structured JSON array to generate the PowerPoint slide deck.

---

## 🛠️ Technology Stack

| Category | Technologies |
|-----------|-------------|
| Application UI & Logic | Python, Streamlit |
| Agentic AI Framework | CrewAI, LangChain (`langchain-experimental`) |
| Local LLM Provider | Ollama (`qwen2.5-coder:3b`) |
| Data Processing & ML | Pandas, NumPy, Scikit-Learn |
| Visualizations | Plotly Express |
| Export Engines | `python-pptx`, `reportlab` |
| Database Integration | SQLAlchemy, psycopg2 |

---

## 🚀 Installation & Setup

### Prerequisites

Before starting, ensure you have:

- **Python 3.10+**
- **Ollama** installed and running globally on your system

Download Ollama:

https://ollama.com

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/insight-agent.git
cd insight-agent
```

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
```

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn plotly crewai \
langchain langchain-community langchain-experimental \
python-pptx reportlab sqlalchemy psycopg2-binary openpyxl
```

---

### 4️⃣ Setup the Local LLM (Ollama)

Pull the base model:

```bash
ollama pull qwen2.5-coder:3b
```

Create the custom model:

```bash
ollama create insight-agent -f Modelfile
```

Make sure the Ollama service is running in the background before launching the application.

---

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## 🖥️ How to Use

### 1. Authenticate

Log in through the sidebar.

> Note: The current development version accepts any username/password combination.

### 2. Ingest Data

Use the **Data Ingestion Dashboard** to:

- Upload `.csv` or `.xlsx` files
- Connect to a PostgreSQL database
- Query and extract datasets

### 3. Configure AutoML (Optional)

After loading data:

- Select a **Target Column**
- Choose:
  - Classification
  - Regression

The system will train a baseline predictive model automatically.

### 4. Run the Pipeline

Click:

```text
🚀 Generate Business Report
```

The CrewAI workflow will execute sequentially.

You can monitor agent reasoning and outputs inside the **Agent Workspace** panel.

### 5. Review & Export

Once complete, you can:

- Explore interactive Plotly visualizations
- Read the executive Markdown report
- Chat with your dataset
- Export findings as:
  - PDF
  - PowerPoint (.pptx)

---

## 📂 Project Structure

```text
insight-agent/
│
├── app.py                   # Main Streamlit application entry point
├── Modelfile                # Ollama model configuration
├── .gitignore               # Git ignore rules
│
├── ui/                      # Frontend / Streamlit UI Components
│   ├── auth.py              # Sidebar authentication
│   ├── dashboard.py         # File upload & SQL connection dashboard
│   └── data_chat.py         # LangChain conversational dataframe UI
│
└── logic/                   # Backend / Business Logic
    ├── agent_factory.py     # LLM initialization & data context preparation
    ├── analytics_engine.py  # Data cleaning, EDA & AutoML
    ├── crew_pipeline.py     # CrewAI agents & orchestration pipeline
    └── export_engine.py     # PDF and PowerPoint generation
```

---

## 🔄 End-to-End Workflow

```text
Data Upload / SQL Connection
              │
              ▼
      Data Profiling
              │
              ▼
       Data Cleaning
              │
              ▼
 Exploratory Data Analysis
              │
              ▼
      AutoML (Optional)
              │
              ▼
    Multi-Agent Analysis
              │
              ▼
 Business Report Creation
              │
              ▼
 PDF / PPTX Export
```

---

## 🔒 Privacy & Security

Insight-Agent is designed for privacy-first analytics.

- All LLM inference runs locally through Ollama.
- No data is transmitted to external APIs.
- Database credentials remain local to the application.
- Reports and generated outputs are stored on your machine.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

To contribute:

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/amazing-feature
```

3. Commit your changes

```bash
git commit -m "Add amazing feature"
```

4. Push to GitHub

```bash
git push origin feature/amazing-feature
```

5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License**.

Feel free to use, modify, and distribute it in accordance with the license terms.

---

## 👨‍💻 Author

**Sanhil**

GitHub: https://github.com/Sanhil7311

---

⭐ If you found this project useful, consider giving it a star on GitHub!
