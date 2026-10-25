# Exploratory Data Analysis - Task 2: India Crime Data Analysis

An end-to-end Exploratory Data Analysis (EDA) project analyzing crime statistics across Indian States and Union Territories.

---

## 📌 Project Overview
This repository contains comprehensive exploratory data analysis, data cleaning pipelines, statistical diagnostics, and visualization workflows for Indian crime records across multiple states, Union Territories, and historical years.

### Key Objectives:
- Investigate temporal crime trends across years and regions.
- Evaluate crime distribution across major categories (Violent Crime, Property Crime, Economic Offenses, etc.).
- Identify high-incidence states and analyze chargesheet / conviction efficiency.
- Generate interactive and publication-quality visualizations using Pandas, Matplotlib, and Seaborn.

---

## 📂 Repository Structure

```text
├── 02_india_crime_eda.ipynb         # Interactive Jupyter Notebook with complete EDA, charts & insights
├── cleaned_india_crime_data.csv     # Granular incident-level cleaned dataset
├── cleaned_crime_state_year.csv     # Aggregated State-Year panel dataset for trend analysis
├── generate_crime_eda_notebook.py   # Script to programmatically generate/reproduce the EDA notebook
├── inspect_crime_data.py            # Summary diagnostics and validation script
├── .gitignore                       # Git ignore file for Python/Jupyter artifacts
└── README.md                        # Project documentation
```

---

## 📊 Key Highlights & Findings
- **Data Coverage**: Multi-year crime statistics across Indian States & Union Territories.
- **Top Metrics Analyzed**:
  - Total cases reported vs. arrests and chargesheet rates.
  - Property stolen vs. recovered valuation (INR).
  - State-wise comparative rankings and yearly growth trajectories.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with the required libraries:

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### Running the Project

1. **Run diagnostics**:
   ```bash
   python inspect_crime_data.py
   ```

2. **Launch the EDA Notebook**:
   ```bash
   jupyter notebook 02_india_crime_eda.ipynb
   ```

---

## 👤 Author
- **GitHub**: [@Sreenu142](https://github.com/Sreenu142)
