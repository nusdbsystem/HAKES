# %%
from hakesclient import Client

client = Client("localhost", 8080)
if client.connect("root", "root"):
    print("Connected to Hakes server")
else:
    print("Failed to connect to Hakes server")

# %%