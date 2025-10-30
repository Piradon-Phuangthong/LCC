# 🧱 Low-Carbon Concrete Mix Designer (LCC)

A data-driven Python tool that designs **low-carbon concrete mixes** with realistic strength, density, and embodied-carbon profiles. Built to support sustainable material design through machine learning and engineering computation.

## 🚀 Key Features
- Predicts **water-to-binder ratio** and binder composition using **KNN regression**
- Optimises **SCM blends** (GGBFS, Fly Ash) for target 3-day and 28-day strengths  
- Performs **aggregate scaling and density validation** to match real mix behaviour  
- Estimates **embodied carbon** through both empirical and factor-based pathways  
- Includes **spreadsheet-style validation** for comparison with manual calculations  

## 🧩 Tech Stack
- **Python** (NumPy, pandas, scikit-learn)  
- **Excel-style computation logic** for mass balance and absolute volume  
- Command-line interface for user-driven mix design and testing  

## 📁 Project Structure
LCC/
├── lcc_mix_designer.py # Core mix design and KNN logic
├── data/ # Experimental dataset (45 mixes)
├── results/ # Validation outputs
└── README.md

## 📊 Example Output
=== Strength-Driven Mix Designer ===
Target 28-day strength: 45 MPa
Predicted w/b ratio: 0.47
Binder split:
Cement 35.1%
GGBFS 64.9%
Fresh density: 2405 kg/m³ | Embodied carbon: 290 kg CO₂/m³


## 👤 Author
**Piradon Phuangthong**  

📄 *MIT License © 2025 Piradon Phuangthong*
