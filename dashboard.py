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

    num_fields = st.number_input("Enter the number of entries to add:", min_value=1, value=1, step=1)
    titles = []
    contents = []
    

    for i in range(int(num_fields)):
        with st.container():
            st.write(f"Entry {i + 1}")
            titles.append(st.text_input(f"Enter title {i + 1}:"))
            contents.append(st.text_area(f"Enter content {i + 1}:"))
            


    if st.button("Submit"):
        connection = connect_to_db()
        for title, content in zip(titles, contents):
            if title and content:
                insert_data(connection, data_type, title, content)
        st.success("Data uploaded successfully!")
        connection.close()
if __name__ == "__main__":
    main()
