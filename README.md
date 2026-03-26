# 🍽️ Smart Mess Waste Optimization System

> **Predict, analyze, and reduce food waste in institutional mess facilities using Machine Learning.**

An system built for Jaiswal Mess at IIT Gandhinagar that analyzes daily mess food waste patterns (March 2026), provides detailed exploratory data analysis, and predicts future waste based on menu items — helping mess managers optimize food preparation and minimize waste.

---

## ✨ Features

### 🔮 Waste Prediction
- Enter the day's menu (breakfast, lunch, snacks, dinner) and get an instant waste prediction
- Dynamic insights based on statistical thresholds (25th/75th percentiles)
- Best model auto-selected from multiple candidates

### 🏆 Multi-Model Comparison
- **3 ML models** evaluated: Random Forest, Gradient Boosting, Ridge Regression
- Cross-validation with R², MAE, RMSE metrics
- Best model automatically selected based on CV performance

### 📊 Model Visualization
- **Actual vs Predicted** scatter plot
- **Residual analysis** plot with ±2σ bounds
- **Model comparison** bar chart (Train R² vs CV R²)
- **Cross-validation score distribution** box plot
- **Per-sample prediction error** chart

### 📈 Comprehensive EDA (13 Analyses)
| # | Analysis | Description |
|---|----------|-------------|
| 1 | Waste Trend | Total waste over time with fill |
| 2 | Meal Distribution | Average waste per meal (bar chart) |
| 3 | Waste Histogram | Distribution with mean line |
| 4 | Correlation Heatmap | Annotated correlation matrix |
| 5 | Top Waste Days | Highest waste days with breakdown |
| 6 | Day-of-Week Pattern | Waste by weekday (color-coded weekends) |
| 7 | Stacked Meal Area | Meal-wise waste stacked over time |
| 8 | Rolling Average | 7-day moving average trend |
| 9 | Meal Pie Chart | Proportional waste by meal |
| 10 | Meal Box Plot | Waste spread per meal |
| 11 | Cumulative Waste | Running total over time |
| 12 | Summary Statistics | Mean, std, min, max, percentiles |
| 13 | Weekday vs Weekend | Comparative bar chart |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **ML Framework** | scikit-learn |
| **Feature Engineering** | TF-IDF Vectorization |
| **Models** | Random Forest, Gradient Boosting, Ridge |
| **Visualization** | Matplotlib |
| **Web UI** | Gradio |
| **Data Processing** | Pandas, NumPy |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MessWastage
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser** at [http://localhost:7860](http://localhost:7860)

---

## 📁 Project Structure

```
MessWastage/
├── app.py               # Gradio web application (6 tabs)
├── model.py             # Multi-model ML pipeline + visualization
├── eda.py               # 13 EDA analyses with styled plots
├── utility.py           # Data loading, cleaning & processing
├── clean_menu.csv       # Weekly mess menu data
├── food-wastage.csv     # Daily food waste records (25 days)
├── menu.xlsx            # Raw menu data (Excel)
├── food-wastage.xlsx    # Raw waste data (Excel)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 📊 Data Description

### `food-wastage.csv`
| Column | Description |
|--------|-------------|
| Date | Date of record (DD-MM-YY) |
| Breakfast_waste | Breakfast food waste (kg) |
| Lunch_waste | Lunch food waste (kg) |
| Snacks_waste | Snacks food waste (kg) |
| Dinner_waste | Dinner food waste (kg) |
| Total_waste | Sum of all meals (kg) |

### `clean_menu.csv`
| Column | Description |
|--------|-------------|
| Day | Day of the week |
| Meal_Type | Meal category |
| Item_Type | Day identifier |
| Food_Item | Name of food item |

---

## 🔮 How It Works

1. **Data Pipeline**: Raw menu and waste data are cleaned, merged by date, and text features are created from menu items
2. **Feature Engineering**: Menu items are converted to TF-IDF vectors capturing food item importance
3. **Model Training**: Three models are trained and evaluated via 3-fold cross-validation
4. **Best Model Selection**: The model with the highest CV R² is auto-selected for predictions
5. **Prediction**: Users enter a menu → TF-IDF transform → Best model predicts waste → Dynamic insight generated

---

## 🔭 Future Scope

- 📱 Mobile-friendly responsive UI
- 📊 Real-time data collection integration
- 🤖 Deep learning models (LSTM/Transformer) for time-series forecasting
- 📧 Automated alerts for predicted high-waste days
- 🗄️ Database integration for persistent data storage
- 📈 Weekly/monthly waste reports generation
- 🍽️ Menu recommendation system to minimize waste

---

## 👨‍💻 Author

**Yash Kiritkumar Parmar**
IIT Gandhinagar

---

## 📄 License

This project is for academic purposes at IIT Gandhinagar.
