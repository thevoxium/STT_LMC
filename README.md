# Speech to Text LMC

### Steps to Setup locally

First, clone the repo locally.

```python
pip install -r requirements.txt
```

If you are on linux, install the packages below if not already installed.
```python
sudo apt install portaudio19-dev
sudo apt install python3-all-dev
```
You need a Google Service Account to run this script. Visit the link below to know how to set it up. Then download a credential json file.
[Google Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts)
[Get Credentials JSON File](https://developers.google.com/workspace/guides/create-credentials)

Create a `.streamlit` folder and make a `secrets.toml` file in that folder. The downloaded json file will have all the details required to set up the `secrets.toml` file. 

Content of `secrets.toml` need to be like shown below:

```yml
# .streamlit/secrets.toml
[gcp_service_account]
type = "service_account"
project_id = "xxx"
private_key_id = "xxx"
private_key = "xxx"
client_email = "xxx"
client_id = "xxx"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "xxx"
```

Run the streamlit file using:
```python 
streamlit run test.py
```
