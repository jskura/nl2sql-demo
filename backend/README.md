# NL2SQL Agent Demo - Backend

This is the backend for the NL2SQL Agent Demo application. It provides API endpoints for converting natural language questions to SQL queries using Gemini and BigQuery.

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

The application uses environment variables for configuration. You can set these in a `.env` file in the backend directory.

### BigQuery Configuration

To use real BigQuery data:

1. Make sure you have the Google Cloud SDK installed and configured.
2. Set up authentication:
   ```
   gcloud auth application-default login
   ```
3. Configure the following environment variables in `.env`:
   ```
   USE_MOCK_DATA=false
   GOOGLE_CLOUD_PROJECT=your-project-id
   BIGQUERY_DATASET=your-dataset-id
   ```

If you don't have BigQuery access, you can use the public datasets:
```
USE_MOCK_DATA=false
GOOGLE_CLOUD_PROJECT=bigquery-public-data
BIGQUERY_DATASET=samples
```

Or use the mock data:
```
USE_MOCK_DATA=true
```

### Vertex AI Configuration

To use real Gemini model:

1. Make sure you have Vertex AI API enabled in your Google Cloud project.
2. Configure the following environment variables in `.env`:
   ```
   USE_MOCK_RESPONSES=false
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_REGION=us-central1
   ```

If you don't have Vertex AI access, you can use the mock responses:
```
USE_MOCK_RESPONSES=true
```

## Running the Server

Start the FastAPI server:
```
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000.

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Available Endpoints

- `GET /api/tables` - List available BigQuery tables
- `GET /api/tables/{table_name}/schema` - Get schema for a specific table
- `GET /api/tables/{table_name}/sample` - Get a sample of data from a table
- `POST /api/generate/basic` - Generate SQL using only the table schema
- `POST /api/generate/enhanced` - Generate SQL with additional context
- `POST /api/generate/agent` - Generate SQL using an agent with tools
- `POST /api/execute` - Execute a SQL query and return results
- `GET /api/history` - Get the query history 