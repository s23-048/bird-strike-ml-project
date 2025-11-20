✈️ Bird Strike Damage Prediction Using Machine Learning

This project predicts the severity of aircraft damage caused by bird strikes using machine learning.
It was developed as part of the Machine Learning Techniques (MLT) course.

📌 Problem Overview

Bird strikes pose a significant threat to aviation.
They can cause:

Mechanical damage

Flight delays

Emergency landings

In rare cases, serious accidents

Traditional monitoring methods are manual and reactive.
Our goal is to build a predictive ML model that can assess the severity of damage based on incident conditions.

🧠 Project Workflow

Download Data
We use a publicly available FAA-like bird strike dataset.

Preprocessing

Extract key features (Aircraft, Altitude, Species, Time of Day, Phase of Flight)

Clean missing values

Convert damage text to numeric severity

Model Training

Use Random Forest Classifier

Train a pipeline with one-hot encoding + classifier

Save model to model/bird_strike_pipeline.pkl

User Interface (UI)

Built with Streamlit

User selects:

Aircraft type

Phase of flight

Bird species

Time of day

Height (altitude)

Model predicts severity:

0 = No Damage

1 = Minor Damage

2 = Severe Damage

📁 Project Structure
bird-strike-ml-project/
│
├── app.py                    # Streamlit UI
├── train.py                  # Train the ML model
├── preprocess.py             # Load and clean dataset
├── get_dataset.py            # Download + prepare dataset
├── requirements.txt          # Dependencies
├── model/
│   └── bird_strike_pipeline.pkl
├── data/
│   └── faa_bird_strike_sample_10k.csv
└── README.md

✔️ Features Used for ML
Feature Name	Description
AIRCRAFT	Type of aircraft involved
PHASE_OF_FLIGHT	Takeoff, landing, cruise, etc.
SPECIES	Bird species
TIME_OF_DAY	Day, night, dawn, dusk
HEIGHT	Altitude in feet
DAMAGE_SEVERITY	Target variable (0, 1, 2)
🤖 Model Used

Random Forest Classifier

Handles non-linear relationships

Works well with categorical + numeric features

Robust for imbalanced datasets

🚀 How to Run the Project (Windows)
1️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate

2️⃣ Install requirements
pip install -r requirements.txt

3️⃣ Generate dataset
python get_dataset.py

4️⃣ Train the model
python train.py

5️⃣ Run the UI
streamlit run app.py


The app opens at:

http://localhost:8501

🧪 Sample Output

The model predicts:

Severity: Minor
Probability:
- No Damage: 60%
- Minor: 30%
- Severe: 10%

📈 Future Improvements

Add more feature engineering

Use XGBoost for higher accuracy

Add ability to upload CSV for bulk prediction

Improve Streamlit UI styling

👨‍💻 Author

Sharanabasava S (USN: 1SI23CI048)
B.Tech CSE (AI & ML)
SIT, Tumkur
