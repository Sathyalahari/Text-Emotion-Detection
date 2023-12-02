import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import pickle
import joblib

from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

#enter credentials
account_name = 'emotiondetectionblob'
account_key = '3MeAhHc/2K8HSxUa8YsElqgGLfYVq6QDmeO3PtvSMA1GQhAU3Mav6RcnSyPnxgRnf/8bVN2tdF2Y+AStsgul2Q=='
container_name = 'textemotionpickle'

# create a client to interact with blob storage
connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# use the client to connect to the container
container_client = blob_service_client.get_container_client(container_name)

# get a list of all blob files in the container
blob_list = []
for blob_i in container_client.list_blobs():
    blob_list.append(blob_i.name)

sas_url_list = []
# generate a shared access signature for files and load them into Python
for blob_i in blob_list:
    # generate a shared access signature for each blob file
    if blob_i == 'emotion_model.pkl':
        sas_i = generate_blob_sas(account_name=account_name,
                                  container_name=container_name,
                                  blob_name=blob_i,
                                  account_key=account_key,
                                  permission=BlobSasPermissions(read=True),
                                  expiry=datetime.utcnow() + timedelta(hours=1))

        sas_url = 'https://' + account_name + '.blob.core.windows.net/' + container_name + '/' + blob_i + '?' + sas_i
        sas_url_list.append(sas_url)
# import joblib

sas_url_job = 'https://emotiondetectionblob.blob.core.windows.net/textemotionpickle/text_emotion.pkl?sp=r&st=2023-12-02T05:06:54Z&se=2023-12-02T13:06:54Z&spr=https&sv=2022-11-02&sr=b&sig=NkDFqJwM%2Ba%2BCqtx0%2BfFZ0TxOiNic4av7pDuRI%2BVoB%2FQ%3D'
pipe_lr = joblib.load(open(sas_url_job, 'rb'))

sas_url_direct = 'https://emotiondetectionblob.blob.core.windows.net/textemotionpickle/emotion_model.pkl?sp=r&st=2023-12-02T05:02:34Z&se=2023-12-02T13:02:34Z&spr=https&sv=2022-11-02&sr=b&sig=30upLVZcQpy5RSvDJ%2BMQSwBcYYLTiyOb0Xfrwc3FQvo%3D'
# pipe_lr = pickle.load(open(sas_url_direct, 'rb'))

# pipe_lr = pickle.load(open(emotion_model.pkl, 'rb'))

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
