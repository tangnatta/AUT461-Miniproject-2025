# AUT461-Miniproject-2025

## Dataset Setup

### Dataset Requirements

The dataset for this project should be placed in the `data/` directory. The expected structure is as follows:

```
data/
├── Comprehensive_Global_COVID-19_Dataset.csv
├── Covid19-TestingRecord.csv
├── Covid19-VariantsFound.csv
├── Vaccinations_ByCountry_ByManufacturer.csv
└── Vaccinations_ByCountry.csv
```

Please ensure all CSV files maintain their original format to prevent data loading issues.

## Installation Guide

### Prerequisites

- Python 3.13.2
- pip package manager

### Setup Instructions

1. Clone this repository

   ```bash
   git clone https://github.com/tangnatta/AUT461-Miniproject-2025.git
   cd AUT461-Miniproject-2025
   ```

2. Create a virtual environment (optional but recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from LEB2 and extract to the `data/` directory

5. Run the project using jupyter notebook `main.ipynb`
