import os
import json
from typing import Dict, List, Any, Optional
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
import pandas as pd
import time
import logging

from app.models.models import TableSchema, ColumnInfo, DataSample
from app.core.config import (
    USE_MOCK_DATA, 
    GOOGLE_CLOUD_PROJECT, 
    BIGQUERY_DATASET,
    BASIC_DEMO_DATASET,
    ENHANCED_DEMO_DATASET,
    AGENT_DEMO_DATASET
)
from app.utils.performance import log_execution_time, PerformanceTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
BIGQUERY_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "bigquery-public-data.samples")

# Sample mock data for demo purposes (used as fallback)
MOCK_SCHEMAS = {
    "sales": TableSchema(
        table_name="sales",
        columns=[
            ColumnInfo(name="sale_id", data_type="INTEGER", description="Unique identifier for each sale"),
            ColumnInfo(name="product_id", data_type="INTEGER", description="Foreign key to products table"),
            ColumnInfo(name="customer_id", data_type="INTEGER", description="Foreign key to customers table"),
            ColumnInfo(name="sale_date", data_type="DATE", description="Date when the sale occurred"),
            ColumnInfo(name="quantity", data_type="INTEGER", description="Number of units sold"),
            ColumnInfo(name="unit_price", data_type="FLOAT", description="Price per unit in USD"),
            ColumnInfo(name="total_amount", data_type="FLOAT", description="Total sale amount in USD")
        ],
        description="Contains records of all sales transactions"
    ),
    "customers": TableSchema(
        table_name="customers",
        columns=[
            ColumnInfo(name="customer_id", data_type="INTEGER", description="Unique identifier for each customer"),
            ColumnInfo(name="name", data_type="STRING", description="Customer's full name"),
            ColumnInfo(name="email", data_type="STRING", description="Customer's email address"),
            ColumnInfo(name="signup_date", data_type="DATE", description="Date when customer signed up"),
            ColumnInfo(name="last_purchase_date", data_type="DATE", description="Date of customer's most recent purchase"),
            ColumnInfo(name="total_spent", data_type="FLOAT", description="Total amount spent by customer in USD"),
            ColumnInfo(name="loyalty_tier", data_type="STRING", description="Customer loyalty tier (Bronze, Silver, Gold, Platinum)")
        ],
        description="Contains customer information and purchase history summary"
    ),
    "products": TableSchema(
        table_name="products",
        columns=[
            ColumnInfo(name="product_id", data_type="INTEGER", description="Unique identifier for each product"),
            ColumnInfo(name="name", data_type="STRING", description="Product name"),
            ColumnInfo(name="category", data_type="STRING", description="Product category"),
            ColumnInfo(name="price", data_type="FLOAT", description="Current product price in USD"),
            ColumnInfo(name="cost", data_type="FLOAT", description="Product cost in USD"),
            ColumnInfo(name="inventory", data_type="INTEGER", description="Current inventory level"),
            ColumnInfo(name="reorder_level", data_type="INTEGER", description="Inventory level that triggers reordering")
        ],
        description="Contains product catalog information"
    ),
    "orders": TableSchema(
        table_name="orders",
        columns=[
            ColumnInfo(name="order_id", data_type="INTEGER", description="Unique identifier for each order"),
            ColumnInfo(name="customer_id", data_type="INTEGER", description="Foreign key to customers table"),
            ColumnInfo(name="order_date", data_type="DATE", description="Date when the order was placed"),
            ColumnInfo(name="status", data_type="STRING", description="Order status (Pending, Shipped, Delivered, Cancelled)"),
            ColumnInfo(name="shipping_address", data_type="STRING", description="Shipping address"),
            ColumnInfo(name="shipping_cost", data_type="FLOAT", description="Shipping cost in USD"),
            ColumnInfo(name="total_amount", data_type="FLOAT", description="Total order amount including shipping in USD")
        ],
        description="Contains order header information"
    )
}

MOCK_DATA_SAMPLES = {
    "sales": DataSample(
        table_name="sales",
        columns=["sale_id", "product_id", "customer_id", "sale_date", "quantity", "unit_price", "total_amount"],
        data=[
            {"sale_id": 1001, "product_id": 5, "customer_id": 42, "sale_date": "2023-01-15", "quantity": 2, "unit_price": 29.99, "total_amount": 59.98},
            {"sale_id": 1002, "product_id": 8, "customer_id": 17, "sale_date": "2023-01-15", "quantity": 1, "unit_price": 149.99, "total_amount": 149.99},
            {"sale_id": 1003, "product_id": 12, "customer_id": 28, "sale_date": "2023-01-16", "quantity": 3, "unit_price": 12.50, "total_amount": 37.50},
            {"sale_id": 1004, "product_id": 3, "customer_id": 42, "sale_date": "2023-01-18", "quantity": 1, "unit_price": 499.99, "total_amount": 499.99},
            {"sale_id": 1005, "product_id": 15, "customer_id": 53, "sale_date": "2023-01-20", "quantity": 2, "unit_price": 24.99, "total_amount": 49.98}
        ]
    ),
    "customers": DataSample(
        table_name="customers",
        columns=["customer_id", "name", "email", "signup_date", "last_purchase_date", "total_spent", "loyalty_tier"],
        data=[
            {"customer_id": 17, "name": "Jane Smith", "email": "jane.smith@example.com", "signup_date": "2022-03-15", "last_purchase_date": "2023-01-15", "total_spent": 1249.87, "loyalty_tier": "Silver"},
            {"customer_id": 28, "name": "John Doe", "email": "john.doe@example.com", "signup_date": "2021-11-20", "last_purchase_date": "2023-01-16", "total_spent": 567.50, "loyalty_tier": "Bronze"},
            {"customer_id": 42, "name": "Alice Johnson", "email": "alice.j@example.com", "signup_date": "2020-05-10", "last_purchase_date": "2023-01-18", "total_spent": 3752.43, "loyalty_tier": "Gold"},
            {"customer_id": 53, "name": "Bob Williams", "email": "bob.w@example.com", "signup_date": "2022-09-05", "last_purchase_date": "2023-01-20", "total_spent": 842.65, "loyalty_tier": "Bronze"},
            {"customer_id": 67, "name": "Carol Brown", "email": "carol.b@example.com", "signup_date": "2019-12-01", "last_purchase_date": "2023-01-10", "total_spent": 5241.32, "loyalty_tier": "Platinum"}
        ]
    ),
    "products": DataSample(
        table_name="products",
        columns=["product_id", "name", "category", "price", "cost", "inventory", "reorder_level"],
        data=[
            {"product_id": 3, "name": "Premium Laptop", "category": "Electronics", "price": 499.99, "cost": 350.00, "inventory": 15, "reorder_level": 5},
            {"product_id": 5, "name": "Wireless Earbuds", "category": "Electronics", "price": 29.99, "cost": 12.50, "inventory": 42, "reorder_level": 10},
            {"product_id": 8, "name": "Smart Watch", "category": "Electronics", "price": 149.99, "cost": 75.00, "inventory": 23, "reorder_level": 8},
            {"product_id": 12, "name": "Coffee Mug", "category": "Kitchen", "price": 12.50, "cost": 3.25, "inventory": 78, "reorder_level": 20},
            {"product_id": 15, "name": "Yoga Mat", "category": "Fitness", "price": 24.99, "cost": 8.75, "inventory": 35, "reorder_level": 12}
        ]
    ),
    "orders": DataSample(
        table_name="orders",
        columns=["order_id", "customer_id", "order_date", "status", "shipping_address", "shipping_cost", "total_amount"],
        data=[
            {"order_id": 5001, "customer_id": 42, "order_date": "2023-01-18", "status": "Shipped", "shipping_address": "123 Main St, Anytown, USA", "shipping_cost": 8.99, "total_amount": 508.98},
            {"order_id": 5002, "customer_id": 17, "order_date": "2023-01-15", "status": "Delivered", "shipping_address": "456 Oak Ave, Somewhere, USA", "shipping_cost": 0.00, "total_amount": 149.99},
            {"order_id": 5003, "customer_id": 28, "order_date": "2023-01-16", "status": "Processing", "shipping_address": "789 Pine Rd, Nowhere, USA", "shipping_cost": 4.99, "total_amount": 42.49},
            {"order_id": 5004, "customer_id": 53, "order_date": "2023-01-20", "status": "Pending", "shipping_address": "321 Elm St, Elsewhere, USA", "shipping_cost": 5.99, "total_amount": 55.97},
            {"order_id": 5005, "customer_id": 67, "order_date": "2023-01-10", "status": "Delivered", "shipping_address": "654 Maple Dr, Anywhere, USA", "shipping_cost": 12.99, "total_amount": 212.98}
        ]
    )
}

# Initialize BigQuery client if not using mock data
client = None
if not USE_MOCK_DATA:
    try:
        client = bigquery.Client(project=BIGQUERY_PROJECT)
        logger.info(f"Successfully initialized BigQuery client for project: {BIGQUERY_PROJECT}")
    except Exception as e:
        logger.error(f"Failed to initialize BigQuery client: {str(e)}")
        logger.info("Falling back to mock data")
        USE_MOCK_DATA = True
else:
    logger.info("Using mock data for BigQuery operations")

def get_dataset_for_demo_phase(demo_phase: str) -> str:
    """
    Get the appropriate dataset for the specified demo phase.
    
    Args:
        demo_phase: The demo phase (basic, enhanced, agent)
        
    Returns:
        The dataset ID for the specified demo phase
    """
    logger.debug(f"Getting dataset for demo phase: {demo_phase}")
    
    if demo_phase == "basic":
        return BASIC_DEMO_DATASET
    elif demo_phase == "enhanced":
        return ENHANCED_DEMO_DATASET
    elif demo_phase == "agent":
        return AGENT_DEMO_DATASET
    else:
        logger.warning(f"Unknown demo phase: {demo_phase}, using default dataset")
        return BIGQUERY_DATASET

@log_execution_time
def list_tables(demo_phase: str = "basic") -> List[str]:
    """
    List all tables in the dataset for the specified demo phase.
    
    Args:
        demo_phase: The demo phase (basic, enhanced, agent)
        
    Returns:
        List of table names
    """
    logger.info(f"Listing tables for demo phase: {demo_phase}")
    
    if USE_MOCK_DATA:
        logger.debug("Using mock data for table listing")
        return list(MOCK_SCHEMAS.keys())
    
    try:
        with PerformanceTracker("bigquery_list_tables"):
            dataset_id = get_dataset_for_demo_phase(demo_phase)
            dataset_ref = client.dataset(dataset_id)
            tables = list(client.list_tables(dataset_ref))
            table_names = [table.table_id for table in tables]
            
        logger.info(f"Found {len(table_names)} tables in dataset {dataset_id}")
        return table_names
    except Exception as e:
        logger.exception(f"Error listing tables: {str(e)}")
        return list(MOCK_SCHEMAS.keys())

@log_execution_time
def get_table_schema(table_name: str, demo_phase: str = "basic") -> TableSchema:
    """
    Get the schema for a table.
    
    Args:
        table_name: The name of the table
        demo_phase: The demo phase (basic, enhanced, agent)
        
    Returns:
        TableSchema object with table information
    """
    logger.info(f"Getting schema for table {table_name} in demo phase: {demo_phase}")
    
    if USE_MOCK_DATA:
        logger.debug(f"Using mock schema for table: {table_name}")
        return MOCK_SCHEMAS.get(table_name, MOCK_SCHEMAS["sales"])
    
    try:
        with PerformanceTracker("bigquery_get_table_schema"):
            dataset_id = get_dataset_for_demo_phase(demo_phase)
            table_ref = client.dataset(dataset_id).table(table_name)
            table = client.get_table(table_ref)
            
            columns = []
            for field in table.schema:
                columns.append(ColumnInfo(
                    name=field.name,
                    data_type=field.field_type,
                    description=field.description or f"Field {field.name}"
                ))
            
        logger.info(f"Retrieved schema for table {table_name} with {len(columns)} columns")
        return TableSchema(
            table_name=table_name,
            columns=columns,
            description=table.description or f"Table {table_name}"
        )
    except Exception as e:
        logger.exception(f"Error getting table schema: {str(e)}")
        return MOCK_SCHEMAS.get(table_name, MOCK_SCHEMAS["sales"])

@log_execution_time
def get_data_sample(table_name: str, sample_size: int = 10, demo_phase: str = "basic") -> DataSample:
    """
    Get a sample of data from a table.
    
    Args:
        table_name: The name of the table
        sample_size: The number of rows to sample
        demo_phase: The demo phase (basic, enhanced, agent)
        
    Returns:
        DataSample object with sample data
    """
    logger.info(f"Getting data sample for table {table_name} (size: {sample_size}) in demo phase: {demo_phase}")
    
    if USE_MOCK_DATA:
        logger.debug(f"Using mock data sample for table: {table_name}")
        # Return mock data sample
        mock_data = []
        schema = MOCK_SCHEMAS.get(table_name, MOCK_SCHEMAS["sales"])
        
        for i in range(sample_size):
            row = {}
            for col in schema.columns:
                if col.data_type == "INTEGER":
                    row[col.name] = i + 1000
                elif col.data_type == "FLOAT":
                    row[col.name] = (i + 1) * 10.5
                elif col.data_type == "DATE":
                    row[col.name] = f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
                else:
                    row[col.name] = f"Sample {col.name} {i+1}"
            mock_data.append(row)
            
        return DataSample(
            table_name=table_name,
            columns=[col.name for col in schema.columns],
            data=mock_data
        )
    
    try:
        with PerformanceTracker("bigquery_get_data_sample"):
            dataset_id = get_dataset_for_demo_phase(demo_phase)
            query = f"SELECT * FROM `{dataset_id}.{table_name}` LIMIT {sample_size}"
            
            query_job = client.query(query)
            results = query_job.result()
            
            # Convert to list of dicts
            data = []
            columns = [field.name for field in results.schema]
            
            for row in results:
                row_dict = {}
                for col in columns:
                    value = row[col]
                    # Convert non-serializable types
                    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                        value = str(value)
                    row_dict[col] = value
                data.append(row_dict)
        
        logger.info(f"Retrieved {len(data)} sample rows from table {table_name}")
        return DataSample(
            table_name=table_name,
            columns=columns,
            data=data
        )
    except Exception as e:
        logger.exception(f"Error getting data sample: {str(e)}")
        # Fall back to mock data
        return get_data_sample(table_name, sample_size, demo_phase)

@log_execution_time
def execute_sql(sql: str, demo_phase: str = "basic") -> Dict[str, Any]:
    """
    Execute a SQL query and return the results.
    
    Args:
        sql: The SQL query to execute
        demo_phase: The demo phase (basic, enhanced, agent)
        
    Returns:
        Dictionary with query results
    """
    logger.info(f"Executing SQL query in demo phase: {demo_phase}")
    logger.debug(f"SQL query: {sql}")
    
    if USE_MOCK_DATA:
        logger.debug("Using mock data for SQL execution")
        # Return mock results
        return {
            "columns": ["column1", "column2", "column3"],
            "data": [
                {"column1": "value1", "column2": 100, "column3": "2023-01-01"},
                {"column1": "value2", "column2": 200, "column3": "2023-01-02"},
                {"column1": "value3", "column2": 300, "column3": "2023-01-03"}
            ],
            "row_count": 3,
            "execution_time": 0.5
        }
    
    try:
        start_time = time.time()
        
        with PerformanceTracker("bigquery_execute_sql"):
            dataset_id = get_dataset_for_demo_phase(demo_phase)
            
            # Qualify table names if they're not already qualified
            # This allows queries to work without explicit dataset qualification
            if "." not in sql and "`" not in sql:
                # Simple replacement for unqualified table names
                # Note: This is a basic implementation and may not handle all SQL variations
                tables = list_tables(demo_phase)
                for table in tables:
                    # Replace table name with fully qualified name
                    # Only replace if it's a whole word (not part of another word)
                    sql = sql.replace(f" {table} ", f" `{dataset_id}.{table}` ")
                    sql = sql.replace(f" {table},", f" `{dataset_id}.{table}`,")
                    sql = sql.replace(f" {table}\n", f" `{dataset_id}.{table}`\n")
                    sql = sql.replace(f" {table};", f" `{dataset_id}.{table}`;")
                    sql = sql.replace(f"FROM {table}", f"FROM `{dataset_id}.{table}`")
                    sql = sql.replace(f"JOIN {table}", f"JOIN `{dataset_id}.{table}`")
            
            logger.debug(f"Executing query with qualified table names: {sql}")
            
            query_job = client.query(sql)
            results = query_job.result()
            
            # Convert to list of dicts
            data = []
            columns = [field.name for field in results.schema]
            
            for row in results:
                row_dict = {}
                for col in columns:
                    value = row[col]
                    # Convert non-serializable types
                    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                        value = str(value)
                    row_dict[col] = value
                data.append(row_dict)
        
        execution_time = time.time() - start_time
        logger.info(f"Query executed successfully in {execution_time:.2f}s, returned {len(data)} rows")
        
        return {
            "columns": columns,
            "data": data,
            "row_count": len(data),
            "execution_time": execution_time
        }
    except Exception as e:
        logger.exception(f"Error executing SQL query: {str(e)}")
        return {
            "columns": ["error"],
            "data": [{"error": str(e)}],
            "row_count": 1,
            "execution_time": 0,
            "error": str(e)
        } 