# Vertex AI Configuration for NL2SQL Demo

This document explains how to configure Vertex AI for the NL2SQL demo application.

## Overview

The NL2SQL demo application can use Google's Vertex AI Gemini models to generate SQL queries from natural language questions. This provides more accurate and sophisticated SQL generation compared to the mock responses.

## Prerequisites

Before using Vertex AI with this application, you need to:

1. Have a Google Cloud account with billing enabled
2. Enable the Vertex AI API in your Google Cloud project
3. Set up authentication credentials

## Setting Up Authentication

1. Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install

2. Log in to your Google Cloud account:
   ```
   gcloud auth login
   ```

3. Set up application default credentials:
   ```
   gcloud auth application-default login
   ```

4. Set your project ID:
   ```
   gcloud config set project YOUR_PROJECT_ID
   ```

## Configuration

The Vertex AI integration is configured through environment variables in the `.env` file:

```
# Google Cloud Project ID (required for Vertex AI)
GOOGLE_CLOUD_PROJECT=your-project-id

# Set to "false" to use the actual Gemini model
USE_MOCK_RESPONSES=false

# Google Cloud region for Vertex AI
# Make sure this region supports Gemini models
GOOGLE_CLOUD_REGION=us-central1

# Vertex AI model configuration
VERTEX_AI_MODEL=gemini-1.5-pro
VERTEX_AI_MAX_OUTPUT_TOKENS=1024
VERTEX_AI_TEMPERATURE=0.2
```

### Configuration Options

- `USE_MOCK_RESPONSES`: Set to "false" to use the actual Gemini model, or "true" to use mock responses.
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID.
- `GOOGLE_CLOUD_REGION`: The Google Cloud region to use for Vertex AI. Make sure the region supports Gemini models.
- `VERTEX_AI_MODEL`: The Gemini model to use. Options include:
  - `gemini-1.5-pro`: The most capable model for complex tasks
  - `gemini-1.5-flash`: A faster, more cost-effective model
  - `gemini-1.0-pro`: The previous generation model
- `VERTEX_AI_MAX_OUTPUT_TOKENS`: Maximum number of tokens in the generated response (1-8192).
- `VERTEX_AI_TEMPERATURE`: Controls randomness in the output (0.0-1.0). Lower values make the output more deterministic.

## Troubleshooting

If you encounter issues with Vertex AI:

1. Check that you have enabled the Vertex AI API in your Google Cloud project.
2. Verify that your authentication credentials are set up correctly.
3. Make sure the specified region supports the Gemini model you're trying to use.
4. Check the application logs for specific error messages.

If issues persist, the application will automatically fall back to using mock responses.

## Costs

Using Vertex AI Gemini models incurs costs based on the number of input and output tokens. For the latest pricing information, see the [Vertex AI pricing page](https://cloud.google.com/vertex-ai/pricing).

For development and testing, you can use the free tier, which includes a limited number of tokens per month.

## Fallback to Mock Responses

If you don't want to use Vertex AI or encounter issues, you can set:

```
USE_MOCK_RESPONSES=true
```

This will use pre-defined mock responses instead of calling the Gemini API. 