# Import libraries
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
import time
import pickle

# Load prepared csv as dataframe to get min & max value of each attributes
df = pd.read_csv("app/csv/cleveland.csv")

# Load test set
df_test = pd.read_csv("app/csv/test_resampled_scaled_cleveland.csv")
X_test = df_test.drop("target", axis=1)
y_test = df_test['target']

# Load scaler
with open("app/scalers/scaler_minmax.pkl", 'rb') as file:
  scaler = pickle.load(file)

# Load trained models
models = {
  'title': [],
  'file': [],
  'data': [],
  'accuracy': [],
  'title_acc': []
}

with open("app/models/models.txt", "r") as file:
  models_read = file.readlines()

for i in range(len(models_read)):
  models['title'].append(models_read[i].strip().split(',')[0])
  models['file'].append(models_read[i].strip().split(',')[1])

for item in models['file']:
  with open(f"app/models/{item}", 'rb') as file:
    models['data'].append(pickle.load(file))

for i in range(len(models['data'])):
  accuracy = round((accuracy_score(y_test, models['data'][i].predict(X_test)) * 100), 2)
  models['accuracy'].append(accuracy)
  models['title_acc'].append(f"{models['title'][i]} - {accuracy}%")

best_model_index = models['accuracy'].index(max(models['accuracy']))

# ========================================================================================================================================================================================

# STREAMLIT
st.set_page_config(
  page_title = "Heart Disease Classification (UCI Cleveland)",
  page_icon = ":sparkling_heart:"
)

st.title(":red[Heart Disease Classification :sparkling_heart:]")
st.subheader("Dataset Source: [UCI Heart Disease (Cleveland) - 1988](https://archive.ics.uci.edu/dataset/45/heart+disease)")
st.write("by [Noval Althoff](https://www.linkedin.com/in/novalalthoff/) (Feb 2024)")
st.write("")

model_sb = st.selectbox(label=":orange[**Model used**]", options=models['title_acc'], index=best_model_index)
st.write("")
st.write("")
model_index = models["title_acc"].index(model_sb)
model = models['data'][model_index]

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1:
  st.sidebar.header("**User Inputs** Sidebar")

  age = st.sidebar.number_input(label=":violet[**Age**]", min_value=df['age'].min(), max_value=df['age'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df['age'].min()}**], :red[Max] value: :red[**{df['age'].max()}**]")
  st.sidebar.write("")

  sex_sb = st.sidebar.selectbox(label=":violet[**Sex**]", options=["Male", "Female"])
  st.sidebar.write("")
  st.sidebar.write("")
  if sex_sb == "Male":
    sex = 1
  elif sex_sb == "Female":
    sex = 0
  # -- Value 0: Female
  # -- Value 1: Male

  cp_sb = st.sidebar.selectbox(label=":violet[**Chest pain type**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
  st.sidebar.write("")
  st.sidebar.write("")
  if cp_sb == "Typical angina":
    cp = 1
  elif cp_sb == "Atypical angina":
    cp = 2
  elif cp_sb == "Non-anginal pain":
    cp = 3
  elif cp_sb == "Asymptomatic":
    cp = 4
  # -- Value 1: typical angina
  # -- Value 2: atypical angina
  # -- Value 3: non-anginal pain
  # -- Value 4: asymptomatic

  trestbps = st.sidebar.number_input(label=":violet[**Resting blood pressure** (in mm Hg on admission to the hospital)]", min_value=df['trestbps'].min(), max_value=df['trestbps'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df['trestbps'].min()}**], :red[Max] value: :red[**{df['trestbps'].max()}**]")
  st.sidebar.write("")

  chol = st.sidebar.number_input(label=":violet[**Serum cholestoral** (in mg/dl)]", min_value=df['chol'].min(), max_value=df['chol'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df['chol'].min()}**], :red[Max] value: :red[**{df['chol'].max()}**]")
  st.sidebar.write("")

  fbs_sb = st.sidebar.selectbox(label=":violet[**Fasting blood sugar > 120 mg/dl?**]", options=["False", "True"])
  st.sidebar.write("")
  st.sidebar.write("")
  if fbs_sb == "False":
    fbs = 0
  elif fbs_sb == "True":
    fbs = 1
  # -- Value 0: false
  # -- Value 1: true

  restecg_sb = st.sidebar.selectbox(label=":violet[**Resting electrocardiographic results**]", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
  st.sidebar.write("")
  st.sidebar.write("")
  if restecg_sb == "Normal":
    restecg = 0
  elif restecg_sb == "Having ST-T wave abnormality":
    restecg = 1
  elif restecg_sb == "Showing left ventricular hypertrophy":
    restecg = 2
  # -- Value 0: normal
  # -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV)
  # -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

  thalach = st.sidebar.number_input(label=":violet[**Maximum heart rate achieved**]", min_value=df['thalach'].min(), max_value=df['thalach'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df['thalach'].min()}**], :red[Max] value: :red[**{df['thalach'].max()}**]")
  st.sidebar.write("")

  exang_sb = st.sidebar.selectbox(label=":violet[**Exercise induced angina?**]", options=["No", "Yes"])
  st.sidebar.write("")
  st.sidebar.write("")
  if exang_sb == "No":
    exang = 0
  elif exang_sb == "Yes":
    exang = 1
  # -- Value 0: No
  # -- Value 1: Yes

  oldpeak = st.sidebar.number_input(label=":violet[**ST depression induced by exercise relative to rest**]", min_value=df['oldpeak'].min(), max_value=df['oldpeak'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df['oldpeak'].min()}**], :red[Max] value: :red[**{df['oldpeak'].max()}**]")
  st.sidebar.write("")

  slope_sb = st.sidebar.selectbox(label=":violet[**The slope of the peak exercise ST segment**]", options=["Upsloping", "Flat", "Downsloping"])
  st.sidebar.write("")
  st.sidebar.write("")
  if slope_sb == "Upsloping":
    slope = 1
  elif slope_sb == "Flat":
    slope = 2
  elif slope_sb == "Downsloping":
    slope = 3
  # -- Value 1: Upsloping
  # -- Value 2: Flat
  # -- Value 3: Downsloping

  ca = st.sidebar.slider(label=":violet[**Number of major vessels colored by flourosopy**]", min_value=0, max_value=3)
  st.sidebar.write("")

  thal_sb = st.sidebar.selectbox(label=":violet[**Defect**]", options=["Normal", "Fixed defect", "Reversable defect"])
  st.sidebar.write("")
  st.sidebar.write("")
  if thal_sb == "Normal":
    thal = 3
  elif thal_sb == "Fixed defect":
    thal = 6
  elif thal_sb == "Reversable defect":
    thal = 7
  # -- Value 3: Normal
  # -- Value 6: Fixed defect
  # -- Value 7: Reversable defect

  data = {
    'attributes': [
      "Age", "Sex", "Chest pain type", "Resting Blood Pressure",
      "Serum Cholestoral", "FBS > 120 mg/dl?", "Resting ECG",
      "Maximum heart rate", "Exercise induced angina?", "ST depression",
      "Peak exercise slope", "Colored major vessels", "Defect"
    ],
    'inputs': [
      age, sex_sb, cp_sb, f"{trestbps} mm Hg", f"{chol} mg/dl", fbs_sb,
      restecg_sb, thalach, exang_sb, oldpeak, slope_sb, ca, thal_sb
    ]
  }

  st.header("User Inputs")
  st.write("")

  col1, col2 = st.columns([1, 2])

  with col1:
    for item in data['attributes']:
      st.write(f":violet[**{item}**]")

  with col2:
    for item in data['inputs']:
      st.write(str(item))

  st.write("")
  st.write("")

  result = ":violet[-]"

  predict_btn = st.button("**Predict**", type="primary")

  st.write("")
  if predict_btn:
    inputs = scaler.transform([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(inputs)[0]

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    if prediction == 0:
      result = ":green[**Healthy**]"
    elif prediction == 1:
      result = ":orange[**Stage I Heart Disease**]"
    elif prediction == 2:
      result = ":orange[**Stage II Heart Disease**]"
    elif prediction == 3:
      result = ":red[**Stage III Heart Disease**]"
    elif prediction == 4:
      result = ":red[**Stage IV Heart Disease**]"

  st.write("")
  st.write("")
  st.subheader("Prediction:")
  st.subheader(result)

with tab2:
  st.header("Predict multiple rows of data")

  sample_csv = df.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

  st.write("")
  st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

  st.write("")
  st.write("")
  file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

  if file_uploaded:
    df_uploaded = pd.read_csv(file_uploaded)
    df_transformed = scaler.transform(df_uploaded)
    prediction_arr = model.predict(df_transformed)

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    result_arr = []

    for prediction in prediction_arr:
      if prediction == 0:
        result = "Healthy"
      elif prediction == 1:
        result = "Stage I Heart Disease"
      elif prediction == 2:
        result = "Stage II Heart Disease"
      elif prediction == 3:
        result = "Stage III Heart Disease"
      elif prediction == 4:
        result = "Stage IV Heart Disease"
      result_arr.append(result)

    df_results = pd.DataFrame({'Prediction Result': result_arr})

    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    col1, col2 = st.columns([1, 2])

    with col1:
      st.dataframe(df_results)
    with col2:
      st.dataframe(df_uploaded)
