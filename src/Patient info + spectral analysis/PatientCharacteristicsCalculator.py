"""
This file is used to calculate patient characteristics per age category.
"""

import os
import pandas as pd
import numpy as np

patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PICU_PatientData.xlsx'

patientData = pd.read_excel(os.path.join(patient_data_file_path, patient_data_file_name))

#%% Select age category
age_category = 'all'# '0-2 months'
#patientData_sub = patientData[patientData["Age category"] == age_category]
patientData_sub = patientData
#patientData_sub = patientData[patientData["Age category"].isin(['2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years', '9-13 years', '13-18 years'])]
#patientData_sub = patientData[patientData["Age category"].isin(['6-12 months', '1-3 years', '3-5 years', '5-9 years', '9-13 years', '13-18 years'])]

# Mean age calculation
ages_num = []
for row in patientData_sub.iterrows():
    year = float(row[1]['Age'].split()[0])
    month = float(row[1]['Age'].split()[2])
    days = float(row[1]['Age'].split()[4])
    age_num = year + month/12 + days/365
    ages_num.append(age_num)

mean_age_num = np.mean(ages_num) # Age in decimals
median_age_num = np.median(ages_num)

# Percentage of males
num_males = np.sum(patientData_sub['Gender'] == 'm')
num_females = np.sum(patientData_sub['Gender'] == 'f')

# Percentage of cognitive impairment
cognitive_impairment_prc = np.sum(patientData_sub['Potential neurophysiological EEG interference by cognitive impairment']
                                  == 'yes')/len(patientData_sub)

print('Age category:', age_category,
      '\n Total number: ', len(patientData_sub),
      '\n Mean age:', mean_age_num,
      '\n Males:' ,num_males,
      '\n Females:', num_females,
      '\n Cognitive impairment:', cognitive_impairment_prc)

### Information for all
# PSG group
airway_obstruction = np.sum(patientData_sub['PSG group'] == 'Airway obstruction')/len(patientData_sub)
neuromuscular_disease = np.sum(patientData_sub['PSG group'] == 'Neuromuscular disease')/len(patientData_sub)
pulmonary_disease = np.sum(patientData_sub['PSG group'] == 'Pulmonary disease')/len(patientData_sub)
autonomic_dysfunction = np.sum(patientData_sub['PSG group'] == 'Central sleep apnea')/len(patientData_sub)
unknown = np.sum(patientData_sub['PSG group'] == 'Unknown')/len(patientData_sub)

# Percentage of epilepsy
epilepsy_prc = np.sum(patientData['Epilepsy'] == 'yes')/len(patientData_sub)

print('\n Epilepsy:', epilepsy_prc,
'\n Airway obstruction:', airway_obstruction*100,
'\n Neuromuscular disease:', neuromuscular_disease*100,
'\n Pulmonary disease:', pulmonary_disease*100,
'\n Autonomic dysfunction:', autonomic_dysfunction*100,
'\n Unknown:', unknown*100)

### Syndromal information
crouzon = np.sum(patientData_sub['Syndrome'] == 'Crouzon syndrome')
down = np.sum(patientData_sub['Syndrome'] == 'Down syndrome')
treacher_collins = np.sum(patientData_sub['Syndrome'] == 'Treacher Collins syndrome')
prader_wili = np.sum(patientData_sub['Syndrome'] == 'Prader-Willi syndrome')
charge = np.sum(patientData_sub['Syndrome'] == 'CHARGE syndrome')

print('Crouzon:', crouzon,
      '\n Down:', down,
      '\n Treacher Collins:', treacher_collins,
      '\n Prader-Wili:', prader_wili,
      '\n charge:', charge)

print('Median age = ', median_age_num)