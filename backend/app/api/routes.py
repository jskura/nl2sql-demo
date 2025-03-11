from fastapi import APIRouter, Depends, HTTPException, Body, Query, Request
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
import uuid

from app.services.nl2sql_service import (
    generate_basic_sql,
    generate_enhanced_sql,
    generate_agent_sql,
    suggest_tables_for_question
)
from app.services.bigquery_service import (
    list_tables,
    get_table_schema,
    get_data_sample,
    execute_sql
)
from app.services.agent_tools import get_agent_tools
from app.models.models import (
    NL2SQLRequest,
    NL2SQLResponse,
    TableSchema,
    DataSample,
    QueryHistoryItem
)
from app.utils.performance import async_log_execution_time, PerformanceTracker

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Get agent tools
agent_tools = get_agent_tools()

# In-memory query history for demo purposes
query_history: List[QueryHistoryItem] = []

def add_to_query_history(question: str, table_name: str, sql: str, demo_phase: str, context: Optional[Dict[str, Any]] = None):
    """Add a query to the history"""
    query_item = QueryHistoryItem(
        question=question,
        table_name=table_name,
        sql=sql,
        demo_phase=demo_phase,
        context=context
    )
    query_history.append(query_item)
    logger.info(f"Added query to history: {question[:30]}...")
    return query_item

def get_history(table_name: Optional[str] = None, limit: int = 10) -> List[QueryHistoryItem]:
    """Get query history, optionally filtered by table name"""
    if table_name:
        filtered_history = [item for item in query_history if item.table_name == table_name]
        return filtered_history[-limit:]
    else:
        return query_history[-limit:]

@router.get("/tables", response_model=List[str])
@async_log_execution_time
async def get_tables(
    request: Request,
    demo_phase: str = Query("basic", description="Demo phase (basic, enhanced, agent)")
):
    """List available BigQuery tables"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"Listing tables for demo phase: {demo_phase}")
    
    try:
        tables = list_tables(demo_phase)
        logger.info(f"Found {len(tables)} tables", extra={"table_count": len(tables)})
        return tables
    except Exception as e:
        logger.exception(f"Failed to list tables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")

@router.get("/tables/{table_name}/schema", response_model=TableSchema)
@async_log_execution_time
async def get_schema(
    request: Request,
    table_name: str, 
    demo_phase: str = Query("basic", description="Demo phase (basic, enhanced, agent)")
):
    """Get schema for a specific table"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"Getting schema for table: {table_name}, demo phase: {demo_phase}", 
                extra={"table_name": table_name})
    
    try:
        schema = get_table_schema(table_name, demo_phase)
        logger.info(f"Schema retrieved with {len(schema.columns)} columns", 
                    extra={"column_count": len(schema.columns)})
        return schema
    except Exception as e:
        logger.exception(f"Failed to get schema for table {table_name}: {str(e)}", 
                         extra={"table_name": table_name})
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")

@router.get("/tables/{table_name}/sample", response_model=DataSample)
@async_log_execution_time
async def get_sample(
    request: Request,
    table_name: str, 
    sample_size: int = Query(10, description="Number of rows to sample"),
    demo_phase: str = Query("basic", description="Demo phase (basic, enhanced, agent)")
):
    """Get a sample of data from a table"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"Getting sample for table: {table_name}, sample size: {sample_size}, demo phase: {demo_phase}", 
                extra={"table_name": table_name, "sample_size": sample_size})
    
    try:
        sample = get_data_sample(table_name, sample_size, demo_phase)
        logger.info(f"Sample retrieved with {len(sample.data)} rows", 
                    extra={"row_count": len(sample.data)})
        return sample
    except Exception as e:
        logger.exception(f"Failed to get sample for table {table_name}: {str(e)}", 
                         extra={"table_name": table_name})
        raise HTTPException(status_code=500, detail=f"Failed to get sample: {str(e)}")

@router.post("/execute", response_model=Dict[str, Any])
@async_log_execution_time
async def execute_sql_query(
    request: Request,
    sql: str = Body(..., embed=True),
    demo_phase: str = Query("basic", description="Demo phase (basic, enhanced, agent)")
):
    """Execute a SQL query and return results"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"Executing SQL query in demo phase: {demo_phase}")
    logger.debug(f"SQL query: {sql}", extra={"sql": sql})
    
    try:
        # Execute the query
        results = execute_sql(sql, demo_phase)
        
        # Add to query history if successful
        if not results.get('error'):
            logger.info(f"Query executed successfully, returned {results.get('row_count', 0)} rows", 
                        extra={"row_count": results.get('row_count', 0)})
        
        return results
    except Exception as e:
        logger.exception(f"Failed to execute SQL query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute query: {str(e)}")

@router.post("/nl2sql/basic", response_model=NL2SQLResponse)
@async_log_execution_time
async def basic_nl2sql(
    request: Request,
    nl2sql_request: NL2SQLRequest,
    demo_phase: str = Query("basic", description="Demo phase (basic, enhanced, agent)")
):
    """Generate SQL from natural language using only the table schema"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    question = nl2sql_request.question
    table_name = nl2sql_request.table_name
    additional_tables = nl2sql_request.additional_tables or []
    
    logger.info(f"Basic NL2SQL request for primary table: {table_name}, additional tables: {', '.join(additional_tables)}", 
                extra={"table_name": table_name, "additional_tables": additional_tables})
    logger.debug(f"Question: {question}", extra={"question": question})
    
    try:
        # Get the primary table schema
        primary_schema = get_table_schema(table_name, demo_phase)
        
        # Get schemas for additional tables
        additional_schemas = {}
        for additional_table in additional_tables:
            additional_schemas[additional_table] = get_table_schema(additional_table, demo_phase)
        
        # Get a sample of data from the primary table
        sample = get_data_sample(table_name, 5, demo_phase)
        
        # Generate SQL from natural language
        response = await generate_basic_sql(
            question=question,
            schema=primary_schema,
            additional_schemas=additional_schemas
        )
        
        # Add to query history
        context = {
            "additional_tables": additional_tables
        }
        add_to_query_history(question, table_name, response.sql, demo_phase, context)
        
        logger.info(f"Generated SQL query with length {len(response.sql)}")
        return response
    except Exception as e:
        logger.exception(f"Failed to generate SQL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {str(e)}")

@router.post("/nl2sql/enhanced", response_model=NL2SQLResponse)
@async_log_execution_time
async def enhanced_nl2sql(
    request: Request,
    nl2sql_request: NL2SQLRequest,
    demo_phase: str = Query("enhanced", description="Demo phase (basic, enhanced, agent)")
):
    """Generate SQL from natural language with additional context"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    question = nl2sql_request.question
    table_name = nl2sql_request.table_name
    additional_tables = nl2sql_request.additional_tables or []
    
    logger.info(f"Enhanced NL2SQL request for primary table: {table_name}, additional tables: {', '.join(additional_tables)}", 
                extra={"table_name": table_name, "additional_tables": additional_tables})
    logger.debug(f"Question: {question}", extra={"question": question})
    
    try:
        # Get the primary table schema
        primary_schema = get_table_schema(table_name, demo_phase)
        
        # Get schemas for additional tables
        additional_schemas = {}
        for additional_table in additional_tables:
            additional_schemas[additional_table] = get_table_schema(additional_table, demo_phase)
        
        # Get a sample of data from the primary table
        sample = get_data_sample(table_name, 5, demo_phase)
        
        # Get previous queries for context
        previous_queries = get_history(table_name=table_name, limit=3)
        
        # Generate SQL from natural language with enhanced context
        response = await generate_enhanced_sql(
            question=question,
            schema=primary_schema,
            additional_schemas=additional_schemas,
            table_description=primary_schema.description,
            column_descriptions={col.name: col.description for col in primary_schema.columns if col.description}
        )
        
        # Add to query history
        context = {
            "additional_tables": additional_tables
        }
        add_to_query_history(question, table_name, response.sql, demo_phase, context)
        
        logger.info(f"Generated SQL query with length {len(response.sql)}")
        return response
    except Exception as e:
        logger.exception(f"Failed to generate SQL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {str(e)}")

@router.post("/nl2sql/agent", response_model=NL2SQLResponse)
@async_log_execution_time
async def agent_nl2sql(
    request: Request,
    nl2sql_request: NL2SQLRequest,
    demo_phase: str = Query("agent", description="Demo phase (basic, enhanced, agent)")
):
    """Generate SQL from natural language using an agent approach"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    question = nl2sql_request.question
    table_name = nl2sql_request.table_name
    
    logger.info(f"Agent NL2SQL request for table: {table_name}", 
                extra={"table_name": table_name})
    logger.debug(f"Question: {question}", extra={"question": question})
    
    try:
        # Get the table schema
        schema = get_table_schema(table_name, demo_phase)
        
        # Get a sample of data
        sample = get_data_sample(table_name, 5, demo_phase)
        
        # Get previous queries for context
        previous_queries = get_history(table_name=table_name, limit=3)
        
        # Generate SQL using the agent approach
        response = await generate_agent_sql(
            question=question,
            schema=schema,
            table_description=schema.description,
            column_descriptions={col.name: col.description for col in schema.columns if col.description},
            data_sample=sample.data,
            previous_queries=previous_queries
        )
        
        # Add to query history
        add_to_query_history(
            question=question,
            table_name=table_name,
            sql=response.sql,
            demo_phase=demo_phase,
            context={
                "agent_reasoning": response.explanation
            }
        )
        
        logger.info(f"Generated SQL query with length {len(response.sql)}", 
                    extra={"sql_length": len(response.sql)})
        
        return response
    except Exception as e:
        logger.exception(f"Failed to generate SQL from question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {str(e)}")

@router.get("/history", response_model=List[QueryHistoryItem])
@async_log_execution_time
async def get_query_history(
    request: Request,
    table_name: Optional[str] = None,
    limit: int = Query(10, description="Maximum number of history items to return")
):
    """Get query history, optionally filtered by table name"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"Getting query history")
    
    try:
        history = get_history(table_name, limit)
        logger.info(f"Retrieved {len(history)} history items", 
                    extra={"table_name": table_name if table_name else "all"})
        return history
    except Exception as e:
        logger.exception(f"Failed to get query history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get query history: {str(e)}")

@router.post("/tools/analyze-query-history", response_model=Dict[str, Any])
async def analyze_query_history(
    question: str = Body(...),
    table_name: Optional[str] = Body(None)
):
    """Analyze query history to find patterns and make recommendations"""
    try:
        # Filter history by table name if provided
        filtered_history = query_history
        if table_name:
            filtered_history = [h for h in query_history if h.table_name == table_name]
        
        # Use the QueryAnalyzer tool
        analysis_results = agent_tools["query_analyzer"].analyze_query_history(filtered_history, question)
        return analysis_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze query history: {str(e)}")

@router.post("/tools/profile-data-sample", response_model=Dict[str, Any])
async def profile_data_sample(
    table_name: str,
    demo_phase: str = Body("basic", description="Demo phase (basic, enhanced, agent)")
):
    """Profile a data sample to extract insights"""
    try:
        # Get data sample
        data_sample = await get_data_sample(table_name, 5, demo_phase)
        
        # Use the DataProfiler tool
        profiler_results = agent_tools["data_profiler"].profile_data_sample({
            "rows": data_sample.rows,
            "columns": data_sample.columns
        })
        return profiler_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to profile data sample: {str(e)}")

@router.post("/tools/extract-sql-patterns", response_model=Dict[str, Any])
async def extract_sql_patterns(sql: str = Body(...)):
    """Extract patterns from a SQL query."""
    try:
        patterns = agent_tools["query_analyzer"].extract_sql_patterns(sql)
        return {"patterns": patterns}
    except Exception as e:
        logger.error(f"Error extracting SQL patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting SQL patterns: {str(e)}")

@router.post("/tools/suggest-tables", response_model=Dict[str, List[str]])
async def suggest_tables(
    request: Request,
    question: str = Body(...),
    all_table_schemas: Dict[str, TableSchema] = Body(...),
    table_descriptions: Dict[str, str] = Body({}),
    demo_phase: str = Query("basic", description="Demo phase (basic, enhanced, agent)")
):
    """Suggest tables that might be relevant for answering a natural language question."""
    try:
        logger.info(f"Suggesting tables for question: {question}")
        
        suggested_tables = await suggest_tables_for_question(
            question=question,
            all_table_schemas=all_table_schemas,
            table_descriptions=table_descriptions,
            demo_phase=demo_phase
        )
        
        logger.info(f"Suggested {len(suggested_tables)} tables for the question")
        return {"suggested_tables": suggested_tables}
    except Exception as e:
        logger.error(f"Error suggesting tables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error suggesting tables: {str(e)}") 