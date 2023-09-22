from flask import Flask
import os

import tensorflow as tf
import tensorflow_hub as hub
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(model_url)
print('Model Loaded')

import pymongo
import pandas as pd

# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://ao-uat-ai:2d9wzIRiMs6xF97r@actingoffice.sp1um.mongodb.net/ActingOfficeUATAI?retryWrites=true&w=majority")
db = client["ActingOfficeUATAI"]
collection = db["TrainData"]


# Query and retrieve data with specific columns
projection = {
    "Description": 1,
    "Debit": 1,
    "Credit": 1,
    "SIC Code": 1,
    "A'Heads": 1,
    "_id": 0  # Exclude the default "_id" field
}
data = collection.find({}, projection)

# Convert data to a list of dictionaries
data_list = list(data)

# Create a Pandas DataFrame with specific columns
final_df = pd.DataFrame(data_list)

# final_df = pd.read_excel('Corrected EXL.xlsx')
final_df= final_df.fillna(0)

collection = db["SICs"]
projection1 = {
    "SIC Code (Defined)": 1,
    "_id": 0  # Exclude the default "_id" field
}

data1 = collection.find({}, projection1)

# Convert data to a list of dictionaries
data_list1 = list(data1)

# Create a Pandas DataFrame with specific columns
dfs = pd.DataFrame(data_list1)

# dfs = pd.read_excel("all_sic_codes (1).xlsx")
sample_suggestions = dfs["SIC Code (Defined)"].tolist()

def embed(texts):
    return model(texts)
def preprocess_text(text, lowercase=True, remove_special_chars=True, apply_stemming=False, remove_stopwords=False,remove_extra_spaces=True):
    if not isinstance(text, str):

        return ""
    if remove_special_chars:

        text = re.sub(r'[^a-zA-Z\s]', '', text)
    if lowercase:
        text = text.lower()
    
        
    if remove_extra_spaces:
      
        text = re.sub(r'\s+', ' ', text)    

    return text

from flask import Flask, redirect, render_template, request, send_file, send_from_directory, jsonify, url_for
from flask_mail import Message, Mail
import pandas as pd
import io
import joblib

from flask import Flask
from flask_mail import Mail

app = Flask(__name__)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'rathi.1@iitj.ac.in'  # Replace with your Gmail address
app.config['MAIL_PASSWORD'] = 'oyil vhjs ftvu fugj'  # Replace with your Gmail password or an app password
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)


# def email_sent_callback(message, app):
#     def callback(sender, message, **extra):
#         app.logger.info(f"Email sent to {', '.join(message.recipients)}")
#     return callback
def determine_dc(row):
    if row["Debit"] != 0 and row["Credit"] ==0 :
        return 1
    elif row["Credit"] != 0 and row["Debit"] ==0 :
        return -1
    else:
        return 0
    
final_df['Description'] = final_df['Description'].apply(preprocess_text)
final_df['SIC Code'] = final_df['SIC Code'].apply(preprocess_text)

final_df["DC"] = final_df.apply(determine_dc, axis=1)

#Model 1
Desc = np.array(final_df['Description'])
Desc_embeddings = embed(Desc)
DC_emb = np.array(final_df['DC']).reshape(-1, 1)
combined_emb = np.concatenate((Desc_embeddings, DC_emb), axis=1)
nn = NearestNeighbors(n_neighbors=50, metric = 'manhattan')  
nn.fit(combined_emb)

#Model 2
Desc = np.array(final_df['Description'])
Desc_embeddings = embed(Desc)
Sic = np.array(final_df['SIC Code'])
Sic_embeddings = embed(Sic)
DC_emb = np.array(final_df['DC']).reshape(-1, 1)
combined_emb1 = np.concatenate((Desc_embeddings,Sic_embeddings, DC_emb), axis=1)
nn1 = NearestNeighbors(n_neighbors=50, metric = 'manhattan')
nn1.fit(combined_emb1)


df = None
output_file_name = None
file = None
@app.route('/')
def index():
    sic_codes = sorted(sample_suggestions)
    return render_template('upload.html',sic_codes=sic_codes)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global df
    global output_file_name
    global file
    try:
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if file and nn and nn1:

            df = pd.read_excel(file)
            sic_input = request.form['SIC']
        
            df["sic code"] = sic_input

            def determine_dcinput(row):
                if row["debit_amount"] != 0 and row["credit_amount"] ==0 :
                    return 1
                elif row["credit_amount"] != 0 and row["debit_amount"] ==0 :
                    return -1
                else:
                    return 0
            df['description'] = df['description'].apply(preprocess_text)
            df['debit_amount'] = df['debit_amount'].astype(float)
            df['credit_amount'] = df['credit_amount'].astype(float)
            df['description'] = df['description'].astype(str)
            df['sic code'] = df['sic code'].astype(str)
            df["dc"] = df.apply(determine_dcinput, axis=1)
        
            def recommend(desc, dc):
                processed_desc = preprocess_text(desc)
                desc_emb = embed([processed_desc])
                dc_emb = np.array([[dc]]) 
                combined_embinp = np.concatenate((desc_emb, dc_emb), axis=1)

                neighbors = nn.kneighbors(combined_embinp, return_distance=False)[0]
                df_neighbors = final_df.iloc[neighbors]
                neighbor_names = df_neighbors["A'Heads"].tolist()
                if "Square Off" in neighbor_names and "refund" not in processed_desc:
                    neighbor_names.remove("Square Off")
                    
                start_index = next((i for i, name in enumerate(neighbor_names) if name != "Sundry Expenses"), None)
                
                if start_index is not None:
                    neighbor_names = neighbor_names[start_index:]
                    
                unique_neighbor_names = set() 
                updated_neighbor_names = []
                
                for name in neighbor_names:
                    if name not in unique_neighbor_names:
                        updated_neighbor_names.append(name)
                        unique_neighbor_names.add(name)
                
                return updated_neighbor_names

            def recommend1(desc,sic, dc):
                processed_desc = preprocess_text(desc)
                desc_emb = embed([processed_desc])
                processed_sic = preprocess_text(sic)
                sic_emb = embed([processed_sic])
                dc_emb = np.array([[dc]]) 
                combined_embinp = np.concatenate((desc_emb, sic_emb, dc_emb), axis=1)

                neighbors = nn1.kneighbors(combined_embinp, return_distance=False)[0]
                df_neighbors = final_df.iloc[neighbors]
                neighbor_names = df_neighbors["A'Heads"].tolist()
                if "Square Off" in neighbor_names and "refund" not in processed_desc:
                    neighbor_names.remove("Square Off")
                    
                start_index = next((i for i, name in enumerate(neighbor_names) if name != "Sundry"), None)
                
                if start_index is not None:
                    neighbor_names = neighbor_names[start_index:]
                    
                unique_neighbor_names1 = set()  
                updated_neighbor_names1 = []    
                
                for name in neighbor_names:
                    if name not in unique_neighbor_names1:
                        updated_neighbor_names1.append(name)
                        unique_neighbor_names1.add(name)
                
                return updated_neighbor_names1

            def combine_recommendations(desc, sic, dc):
                recommendations1 = recommend(desc, dc)
                recommendations2 = recommend1(desc, sic, dc)
                pointer1, pointer2 = 0, 0
                unique_combined_recommendations = []

                while pointer1 < len(recommendations1) or pointer2 < len(recommendations2):
                    if pointer1 < len(recommendations1):
                        recommendation1 = recommendations1[pointer1]
                        if recommendation1 not in unique_combined_recommendations:
                            unique_combined_recommendations.append(recommendation1)
                        pointer1 += 1

                    if pointer2 < len(recommendations2):
                        recommendation2 = recommendations2[pointer2]
                        if recommendation2 not in unique_combined_recommendations:
                            unique_combined_recommendations.append(recommendation2)
                        pointer2 += 1
                
                return unique_combined_recommendations
            
            df["Predicted Heads"] = df.apply(lambda row: combine_recommendations(row["description"], row["sic code"], row["dc"]), axis=1) 
            extracted_data = df[['description', 'debit_amount', 'credit_amount', 'sic code','Predicted Heads']]  
            remaining_ones = final_df["A'Heads"].unique().tolist()
            return render_template('correction_form.html', excel_data=df.to_dict('records'),remaining_ones=remaining_ones)

        else:
            return "Model not found or uploaded file is invalid."
    except Exception as e:

        return f"An error occurred: {str(e)}"



@app.route('/process_corrections', methods=['POST'])
def process_corrections():
    global df
    global output_file_name
    global file
    try:
        if 'predicted_heads' not in request.form:
            return "No predicted heads data received."
        
        predicted_heads = request.form.getlist('predicted_heads')
        others_text = request.form.getlist('others_text')
        updated_predicted_heads = []

        for i, selection in enumerate(predicted_heads):
            if selection == "Others":
        # If "Others" is selected, use the manually entered text
                updated_predicted_heads.append(others_text[i])
            else:
                updated_predicted_heads.append(selection)

        # Update the 'Predicted Heads' column directly
        df['Predicted Heads'] = updated_predicted_heads

        input_filename = os.path.splitext(file.filename)[0]
        output_file_name = f'Corrected_{input_filename}.xlsx'  # New file name
        output_file_path = os.path.join('output', output_file_name)
        df.to_excel(output_file_path, index=False)

        # Send an email with the corrected Excel file attached
        msg = Message('Correction Report', sender='rathi.1@iitj.ac.in', recipients=['ajayrathi48@gmail.com'])
        msg.body = 'Here is the copy of the corrected Excel file.'
        with app.open_resource(output_file_path) as attachment:
            msg.attach(output_file_name, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', attachment.read())
        mail.send(msg)

        # Check if the data has already been inserted
        if not hasattr(process_corrections, 'data_inserted'):
            # Read the data from the saved Excel file
            collection = db[input_filename]
            tested_data = pd.read_excel(output_file_path, usecols=["description", "debit_amount", "credit_amount", "sic code", "Predicted Heads"])
            
            # Insert the data into the 'Tested' collection in MongoDB
            collection.insert_many(tested_data.to_dict(orient='records'))
            
            # Set the flag to indicate that data has been inserted
            process_corrections.data_inserted = True

        return redirect(url_for('download_file', filename=output_file_name))
    except Exception as e:
        # app.logger.error(f"An error occurred while sending the email: {str(e)}")
        return f"An error occurred: {str(e)}"


    
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('output', filename), as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)

    
