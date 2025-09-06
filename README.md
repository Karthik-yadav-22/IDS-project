# (Introduction To Data Science) IDS-project
this was very basic project which deals with Logistic Regression 

# Cardiovascular Diseases Analysis And Prediction 

Cardiovascular diseases (CVDs) are the leading cause of mortality worldwide, responsible for millions of deaths each year. Rapid urbanization, sedentary lifestyles, unhealthy diets, stress, and increased life expectancy have significantly raised the prevalence of heart-related illnesses. Early diagnosis is critical because many CVDs develop silently over time, showing symptoms only when the condition becomes severe.
->With accurate and timely diagnosis:
->Patients can adopt preventive lifestyle changes.
->Doctors can recommend proper medications or interventions before complications arise.
->Healthcare systems can reduce the economic burden of advanced treatments and hospitalizations.
Thus, building efficient, data-driven tools for early detection of CVD risk has become a global necessity.

I've used one of the most widely used statistical machine learning model called the "Logistic Regression"
the reason behind the selection of this model is mainly due to its binary nature of predictions i.e 1/0 (or) having a problem/perfectly fit.
It is statistically robust, interpretable, and widely used in medical research, making it ideal for datasets where explainability is as important as accuracy.

# Dataset Description

For this study, I used a cardiovascular disease dataset consisting of 70,000 patient records with 12 attributes. Each record corresponds to an individual patient’s medical and lifestyle details, along with a target variable indicating the presence of cardiovascular disease.

The attributes are:
id → Unique identifier for each patient record.
age → Age of the patient (measured in days).
gender → Biological sex of the patient (1 = female, 2 = male).
height → Patient’s height in centimeters.
weight → Patient’s weight in kilograms.
ap_hi → Systolic blood pressure (higher value in a blood pressure reading).
ap_lo → Diastolic blood pressure (lower value in a blood pressure reading).
cholesterol → Cholesterol level of the patient (1 = normal, 2 = above normal, 3 = well above normal).
gluc → Glucose level in the blood (1 = normal, 2 = above normal, 3 = well above normal).
smoke → Whether the patient smokes (0 = no, 1 = yes).
alco → Whether the patient consumes alcohol (0 = no, 1 = yes).
active → Whether the patient engages in physical activity (0 = no, 1 = yes).
cardio → Target variable indicating the presence of cardiovascular disease (0 = no disease, 1 = disease).

# Scope for further enhancement 

To further enhance the diagnosis, deep learning methods such as Convolutional Neural Networks(CNN) can be applied to medical imaging data like X-rays, CT scans, or MRI scans. CNNs excel at automatically learning complex patterns from images, which are often difficult for traditional algorithms to capture. By integrating structured patient data with imaging analysis, predictions can become more accurate and holistic. This approach can help doctors not only detect cardiovascular disease risk but also visualize affected regions for better clinical decision making.

