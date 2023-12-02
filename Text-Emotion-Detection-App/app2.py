import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import pickle
import joblib
import requests
import pathlib
import os

from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

#enter credentials
account_name = 'emotiondetectionblob'
account_key = '3MeAhHc/2K8HSxUa8YsElqgGLfYVq6QDmeO3PtvSMA1GQhAU3Mav6RcnSyPnxgRnf/8bVN2tdF2Y+AStsgul2Q=='
container_name = 'textemotionpickle'

# create a client to interact with blob storage
connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

f = open(pathlib.Path(__file__).parent/'emotion_model.pkl', 'rb')
pipe_lr = pickle.load(f)

# def download_pickle_from_url(blob_url, save_path):
#     response = requests.get(blob_url)
#     if response.status_code == 200:
#         with open(save_path, 'wb') as file:
#             file.write(response.content)
#     else:
#         # Handle the error, raise an exception, or return None as needed
#         raise Exception(f"Failed to download pickle file. Status code: {response.status_code}")
#
# # Example usage:
# blob_url = 'https://emotiondetectionblob.blob.core.windows.net/textemotionpickle/text_emotion.pkl?sp=r&st=2023-12-02T05:06:54Z&se=2023-12-02T13:06:54Z&spr=https&sv=2022-11-02&sr=b&sig=NkDFqJwM%2Ba%2BCqtx0%2BfFZ0TxOiNic4av7pDuRI%2BVoB%2FQ%3D'
# # save_path = 'C:/Users/sathy/OneDrive/Desktop/Fall-2023/Cloud Computing/TextEmotionDetection/Text-Emotion-Detection-App/text_emotion.pkl'
# save_path = 'https://github.com/Sathyalahari/TextEmotionDetection/blob/main/Text-Emotion-Detection-App/text_emotion.pkl'
# try:
#     download_pickle_from_url(blob_url, save_path)
#     pipe_lr = joblib.load(open(save_path, 'rb'))
#     # Now you can use the loaded model in your application
# except Exception as e:
#     print(f"Error: {str(e)}")

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            #st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)






if __name__ == '__main__':
    main()
