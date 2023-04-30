from __future__ import division
import streamlit as st
import mysql.connector
import os
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

def connect_to_db():
    connection = mysql.connector.connect(
        host=st.secrets["mysql_host"],
        user=st.secrets["mysql_user"],
        password=st.secrets["mysql_password"],
        database="train",
    )
    return connection

def insert_data(connection, data_type, title, content):
    cursor = connection.cursor()
    insert_query = """
        INSERT INTO content (data_type, title, content)
        VALUES (%s, %s, %s)
    """
    cursor.execute(insert_query, (data_type, title, content))
    connection.commit()
    cursor.close()

def main():
    st.title("Upload Data")
    data_type = st.selectbox("Select data type:", ("Question & Answer", "Blog", "Forum"))

    title_text = ""

    if data_type == "Question & Answer":
        title_text = "Enter the Question & Answer"
    elif data_type == "Blog":
        title_text = "Enter the Blog Title and Content"
    else:
        title_text = "Enter the Forum Post Content"
    
    title = st.text_input(title_text)
    content = st.text_area("Enter the content:")

    uploaded_file = st.file_uploader("Or choose a file to upload:")
    if uploaded_file is not None:
        content = uploaded_file.read().decode()

    if st.button("Submit"):
        if title and content:
            connection = connect_to_db()
            insert_data(connection, data_type, title, content)
            st.success("Data uploaded successfully!")
            connection.close()
        else:
            st.error("Please provide both title and content.")

if __name__ == "__main__":
    main()
