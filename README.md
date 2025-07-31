# fNIRS Hyperscanning Analysis Tool

A desktop application for analyzing fNIRS hyperscanning data using graph theory measures. Built as a capstone project for a B.Sc. in Software Engineering by Yonatan Sherer.

## 🔍 Overview

This tool enables neuroscientists and researchers to:

- Organize raw `.mat` files from hyperscanning experiments
- Generate correlation matrices between fNIRS channels
- Apply thresholding techniques to build connectivity graphs
- Compute global and local graph metrics
- Compare metrics across dyads, conditions, and groups
- Export results as CSV files, images, and PDF reports with insights

## 🧠 Key Features

- 📂 Automatic folder-based experiment import
- 📊 Correlation matrix generation and heatmap visualization
- ⚙️ Thresholding options: Fixed, Median, and Top Percentile
- 🌐 Global graph metrics: Global Efficiency, Modularity, Clustering Coefficient, etc.
- 🔬 Local metrics: Node Strength, Local Efficiency
- 📈 Visual comparisons across experimental conditions
- 📝 Auto-generated textual insights and exportable reports

## 📁 Folder Structure Example

```
ExperimentName/
├── GroupA/
│   └── Condition1/
│       ├── dyad1_Mom.mat
│       ├── dyad1_Baby.mat
├── GroupB/
│   └── Condition2/
│       ├── dyad2_Mom.mat
│       ├── dyad2_Baby.mat
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Recommended: Create a virtual environment

### Installation

```bash
git clone https://github.com/your-username/fnirs-hyperscanning-tool.git
cd fnirs-hyperscanning-tool
pip install -r requirements.txt
```

### Running the App

```bash
python opening_screen.py
```

### Building an Executable (Windows)

```bash
pyinstaller --name=fnirs-tool --windowed --onefile --icon=favicon3.ico opening_screen.py
```

> Make sure to include `.ui` and other resource files in the `dist/` folder if needed.

## 📦 Dependencies (requirements.txt)

- PyQt6  
- numpy  
- scipy  
- pandas  
- seaborn  
- matplotlib  
- networkx  
- scikit-learn  
- reportlab  
- fpdf  

## 📄 Documentation Included

- 📘 **Project Book** – System overview, architecture, development process  
- 👨‍💻 **User Guide** – Step-by-step usage instructions  
- 🔧 **Maintenance Guide** – Installation, environment setup, and deployment notes  

## 🧪 Sample Data

Use synthetic `.mat` files named by dyad and role (e.g., `dyad1_Mom.mat`) for testing. Each file should contain a time × channels matrix.
Alternatively, load real time-series data by clicking on "Select Folder...".
Sample folder can be found in test directory on this repository.

## 📤 Export Formats

- Heatmaps (.png)  
- Comparison plots (.png)  
- Graph metrics (.csv)  
- PDF reports with auto-generated interpretation  

## 📘 License

MIT License

## 👤 Author

**Yonatan Sherer**  
Capstone Project, B.Sc. Software Engineering  
2025  
---

### 🧠 Acknowledgments

Thanks to the neuroscience community and academic mentors for supporting this project.
