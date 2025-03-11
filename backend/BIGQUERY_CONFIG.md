# BigQuery Configuration for NL2SQL Demo

This document explains how to configure BigQuery datasets for the NL2SQL demo application.

## Overview

The NL2SQL demo application can be configured to use different BigQuery datasets for each demo phase:

1. **Basic Demo**: Simple SQL generation using table schema
2. **Enhanced Demo**: SQL generation with additional context and metadata
3. **Agent Demo**: Advanced SQL generation using agent tools

Each demo phase can use a different dataset, allowing you to showcase different types of data and complexity levels.

## Configuration

The configuration is done through environment variables in the `.env` file:

```
# Default BigQuery dataset
BIGQUERY_DATASET=car_crashes

# Dataset configuration for each demo phase
BASIC_DEMO_DATASET=car_crashes
ENHANCED_DEMO_DATASET=car_crashes
AGENT_DEMO_DATASET=car_crashes
```

You can set different datasets for each phase:

```
BASIC_DEMO_DATASET=simple_dataset
ENHANCED_DEMO_DATASET=medium_dataset
AGENT_DEMO_DATASET=complex_dataset
```

## Recommended Datasets

Here are some recommended public datasets for each demo phase:

### Basic Demo

Simple datasets with straightforward schemas:
- `bigquery-public-data.samples.natality`: Birth statistics
- `bigquery-public-data.samples.shakespeare`: Shakespeare's works
- `bigquery-public-data.samples.gsod`: Global weather data

### Enhanced Demo

Datasets with more complex schemas and relationships:
- `bigquery-public-data.austin_311.311_service_requests`: Austin 311 service requests
- `bigquery-public-data.chicago_crime.crime`: Chicago crime data
- `bigquery-public-data.new_york_trees.tree_census_2015`: NYC tree census

### Agent Demo

Complex datasets with multiple tables and relationships:
- `bigquery-public-data.google_analytics_sample.ga_sessions_*`: Google Analytics sample data
- `bigquery-public-data.stackoverflow`: Stack Overflow public dataset
- `bigquery-public-data.covid19_open_data`: COVID-19 open data

## Using Your Own Datasets

To use your own datasets:

1. Make sure your Google Cloud account has access to the datasets
2. Set the `GOOGLE_CLOUD_PROJECT` to your project ID
3. Set the dataset names for each demo phase

Example:
```
GOOGLE_CLOUD_PROJECT=my-project-id
BASIC_DEMO_DATASET=my_simple_dataset
ENHANCED_DEMO_DATASET=my_medium_dataset
AGENT_DEMO_DATASET=my_complex_dataset
```

## Fallback to Mock Data

If the application cannot connect to BigQuery or the specified datasets, it will fall back to using mock data. To force using mock data, set:

```
USE_MOCK_DATA=true
```

This is useful for development or when you don't have access to BigQuery. 