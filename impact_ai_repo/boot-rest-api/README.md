Impact Predictions for API version changes - Spring Backend

Purpose
This small Spring Boot service forwards requests from the React frontend and CI demo
to the AI Core. It acts as a thin proxy and provides a few convenience behaviors:
- forwards /analysis/report to AI Core /report and returns raw JSON
- forwards dataset and graph requests
- provides a train passthrough
- avoids automatic deserialization issues by forwarding raw bytes for /report

Prerequisites
- Java 17 or newer
- Maven (or Gradle if your project uses it)
- The AI Core service running and reachable (default at http://localhost:8000)

Configuration
The controller reads the AI core base URL from the AI_CORE_URL environment variable.
If not set, it defaults to http://localhost:8000.

Example:
export AI_CORE_URL=http://localhost:8000

Build and run (Maven)
1. Build
   cd spring-backend
   ./mvnw clean package -DskipTests

2. Run
   java -jar target/*.jar

Or run from the IDE using the main SpringBootApplication class.

Endpoints exposed by this service
All endpoints are prefixed with /analysis.

GET /analysis/health
  Forwards to AI Core /health
  Returns health JSON, e.g. {"ok": true}

GET /analysis/datasets
  Forwards to AI Core /datasets
  Returns dataset list available to AI Core

GET /analysis/datasets/{dataset}
  Forwards to AI Core /files?dataset={dataset}
  Returns sample files for a dataset

GET /analysis/files?dataset=...
  (Same as above, older route may be used by UI code)

GET /analysis/graph
  Forwards to AI Core /graph
  Returns the curated dependency graph used for scoring

GET /analysis/report?old=FILE1&new=FILE2&dataset=openapi[&pairId=pair-...]
  Forwards to AI Core /report and streams the response bytes back to client
  Use this for the dashboard and CI PR annotation demo

GET /analysis/consumers?service=svc:orders[&path=/v1/foo]
  Forwards to AI Core /api/v1/consumers
  Returns sample backend and frontend consumers for a service

GET /analysis/versioning?pairId=pair-abc123
  Forwards to AI Core /versioning?pair_id=pair-abc123
  Returns per-pair version metadata

POST /analysis/train
  Forwards JSON body to AI Core /train
  Saves samples in AI Core models/last_train.json

Notes for examiners and developers
- The controller is intentionally thin. It does minimal transformation and returns
  the AI Core responses as-is. This keeps the demo simple and reproducible.
- If AI Core returns a String that contains JSON text, the controller tries to
  parse that string to a JSON object before returning it for better client handling.
- Timeouts and resilience: RestTemplate defaults are used. For production level
  robustness add timeouts, retry logic and proper logging.

Testing manually with curl
- Health
  curl http://localhost:8080/analysis/health

- List datasets
  curl http://localhost:8080/analysis/datasets

- Report (example)
  curl "http://localhost:8080/analysis/report?dataset=openapi&old=openapi--svc-a--v1.canonical.json&new=openapi--svc-a--v2.canonical.json"

Environment used in the thesis demo
- The frontend calls the Spring backend.
- The Spring backend forwards to the AI Core.
- The AI Core reads files under datasets/curated produced by curated_datasets.py.


