
# Impact AI React Dashboard

This is the frontend for the Impact AI prototype. It is used in my thesis to show how API change impact is analysed end to end. The React app talks only to the Spring Boot service. The Spring Boot service then contacts the Python AI Core. This setup keeps things simple for running everything locally.

The dashboard loads curated dataset files, run the impact analysis on different API versions, and visualise the outputs such as predicted risk, ACE details, backend and frontend impacts, calibration, confusion matrix and other metadata.

It also provides export buttons for JSON, CSV and NDJSON


The app sends all HTTP requests to
[http://localhost:8080/analysis](http://localhost:8080/analysis)
unless overridden by the environment variable REACT_APP_BACKEND_URL

The Spring Boot service exposes these endpoints for the UI. They are forwarded to the Python AI Core.

* GET /analysis/datasets
* GET /analysis/datasets/{dataset}
* GET /analysis/report
* GET /analysis/graph
* GET /analysis/consumers
* GET /analysis/versioning
* POST /analysis/train

These map to the AI Core endpoints like /datasets, /files, /report, /graph, /api/v1/consumers, /versioning and /train.

UI dashboard role in thesis:

1. Lets the user select old and new OpenAPI files from the curated dataset.
2. Calls the AI Core through the Spring proxy and displays the results in a clean dashboard.
3. Shows ACE details with explanations and can view the raw JSON .
4. Visualises top backend and frontend impacts, predicted risk and confidence.
5. Provides export options so all results can be saved as evidence.
6. Allows batch runs to show reproducibility and continuous analysis.
7. Demonstrates how a CI check could trigger this analysis before code is merged.

Requirements

* Node.js 18 or newer
* npm or yarn
* Spring Boot backend running on port 8080
* AI Core server running on port 8000
* Curated dataset files inside datasets/curated

If the curated dataset folder is empty, run curated_datasets.py first in the AI Core.

Setup to run locally

1. Start the AI Core
   python3 server.py
   Check health using
   curl [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

2. Start the Spring Boot service
   mvn spring-boot:run
   Check health using
   curl [http://127.0.0.1:8080/analysis/health](http://127.0.0.1:8080/analysis/health)

3. Start the React app
   cd react
   npm install
   npm start
   It runs on [http://localhost:3000](http://localhost:3000)

If the Spring Boot URL is different, set
export REACT_APP_BACKEND_URL=[http://your-url/analysis](http://your-url/analysis)

File structure (high level)
src/App.jsx holds the main screen with charts, ACE viewer and analysis logic
src/services/api.js has all HTTP calls to Spring
src/components contains small UI helpers
public/ holds static assets

Typical workflow when using the UI

1. Pick a dataset from the dropdown
2. Select old and new API files
3. Press Run Analysis
4. Inspect risk, impacts and ACEs
5. Export the report if needed
6. Optionally run a batch of reports for testing

Notes on continuous development

* npm start gives hot reload
* react fetches from Spring which reduces CORS issues
* when the backend changes, just refresh the page to reflect new behaviour
* commit only the curated dataset folder, not raw data

Troubleshooting
If datasets do not load:

* Check Spring is running
* Check AI Core /datasets is reachable
* Check REACT_APP_BACKEND_URL

If analysis fails:

* Look at Spring logs
* Look at server.py logs
* Check if the curated files exist in datasets/curated/canonical and datasets/curated/ndjson

