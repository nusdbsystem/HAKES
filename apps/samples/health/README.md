# Sample application with HealthGPT

In this sample application, we connect a finetuned healthcare large model with external domain information via HAKES.

We will use PubMedQA dataset as the external domain information, follow the `prepare_PubMedQA.py` to prepare the dataset and initialize the learned hakes index parameters.

Follow the `ingestion.py` to ingest the dataset into the hakes service.

`server.py` provides a application server implementation that performs healthcare QA service.
