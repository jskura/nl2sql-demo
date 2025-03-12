import os
import time
import json
from typing import Dict, List, Any, Optional, TypedDict
import asyncio
import logging
import re

# Import Vertex AI SDK
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, SafetySetting, HarmCategory, HarmBlockThreshold

from app.models.models import TableSchema, NL2SQLResponse, QueryHistoryItem, DataSample
from app.core.config import (
    USE_MOCK_RESPONSES, 
    GOOGLE_CLOUD_PROJECT, 
    GOOGLE_CLOUD_REGION,
    VERTEX_AI_MODEL,
    VERTEX_AI_MAX_OUTPUT_TOKENS,
    VERTEX_AI_TEMPERATURE,
    VERTEX_AI_SAFETY_THRESHOLD
)
from app.services.agent_tools import get_agent_tools
from app.utils.performance import async_log_execution_time, PerformanceTracker
from app.services.query_rewriter import QueryRewriter
from app.services.langgraph_agent import LangGraphAgent

# Configure logging
logger = logging.getLogger(__name__)

# Initialize agent tools
agent_tools = get_agent_tools()

# Initialize Vertex AI
try:
    vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)
    model = GenerativeModel(VERTEX_AI_MODEL)
    logger.info(f"Initialized Vertex AI with model: {VERTEX_AI_MODEL}")
except Exception as e:
    logger.warning(f"Failed to initialize Vertex AI: {str(e)}")
    logger.warning("Using mock responses instead")
    model = None

# Constants
VERTEX_AI_MAX_OUTPUT_TOKENS = 1024
VERTEX_AI_TEMPERATURE = 0.2
USE_MOCK_RESPONSES = model is None

# Helper function for text generation
async def generate_text(prompt: str, max_output_tokens: int = VERTEX_AI_MAX_OUTPUT_TOKENS, temperature: float = VERTEX_AI_TEMPERATURE) -> str:
    """
    Generate text using Vertex AI.
    
    Args:
        prompt: The prompt to send to the model
        max_output_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        
    Returns:
        Generated text as string
    """
    with PerformanceTracker("gemini_api_call"):
        generation_config = GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )
        
        # Map the string threshold to the appropriate enum value
        threshold_map = {
            "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
            "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
        }
        
        # Get the appropriate threshold enum value
        threshold = threshold_map.get(VERTEX_AI_SAFETY_THRESHOLD, HarmBlockThreshold.BLOCK_ONLY_HIGH)
        
        # Configure safety settings with proper SafetySetting objects
        safety_settings = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=threshold
            )
        ]
        
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        response_text = response.text
        
    logger.debug(f"Raw response from Vertex AI: {response_text}")
    return response_text

def format_schema_for_prompt(schema: TableSchema) -> str:
    """Format table schema for inclusion in a prompt"""
    result = f"Table Name: {schema.table_name}\n"
    if schema.description:
        result += f"Description: {schema.description}\n"
    result += "Columns:\n"
    
    for col in schema.columns:
        col_desc = f" - {col.description}" if col.description else ""
        result += f"- {col.name} ({col.data_type}){col_desc}\n"
    
    return result

def format_data_sample_for_prompt(data_sample: DataSample) -> str:
    """Format a data sample for inclusion in a prompt"""
    sample_str = f"Sample data from {data_sample.table_name} (showing {len(data_sample.rows)} rows):\n\n"
    
    # Format as a simple table
    # Header
    sample_str += " | ".join(data_sample.columns) + "\n"
    sample_str += "-|-".join(["-" * len(col) for col in data_sample.columns]) + "\n"
    
    # Rows
    for row in data_sample.rows:
        row_str = " | ".join([str(row[col]) for col in data_sample.columns])
        sample_str += row_str + "\n"
    
    return sample_str

def format_query_history_for_prompt(history: List[QueryHistoryItem]) -> str:
    """Format query history for inclusion in a prompt"""
    if not history:
        return ""
    
    history_str = "Previous related queries:\n\n"
    for i, item in enumerate(history, 1):
        history_str += f"Query {i}:\n"
        history_str += f"Question: {item.question}\n"
        history_str += f"SQL: {item.sql}\n\n"
    
    return history_str

def format_data_insights_for_prompt(data_sample: DataSample) -> str:
    """Format data insights for inclusion in a prompt"""
    if not data_sample:
        return ""
    
    # Use the DataProfiler tool to get insights
    profiler_results = agent_tools["data_profiler"].profile_data_sample({
        "rows": data_sample.rows,
        "columns": data_sample.columns
    })
    
    insights = profiler_results["insights"]
    column_stats = profiler_results["column_stats"]
    
    insights_str = "Data insights:\n\n"
    
    # Add general insights
    for insight in insights:
        insights_str += f"- {insight}\n"
    
    insights_str += "\nColumn statistics:\n"
    
    # Add column statistics (simplified for prompt)
    for col_name, stats in column_stats.items():
        insights_str += f"\n{col_name} ({stats['type']}):\n"
        
        if stats["type"] == "numeric" and stats["numeric_stats"]["min"] is not None:
            insights_str += f"- Range: {stats['numeric_stats']['min']} to {stats['numeric_stats']['max']}\n"
            if stats["numeric_stats"]["avg"] is not None:
                insights_str += f"- Average: {stats['numeric_stats']['avg']:.2f}\n"
        
        if stats["unique_count"] > 0:
            insights_str += f"- Unique values: {stats['unique_count']} ({(stats['unique_count'] / (stats['non_null_count'] + stats['null_count']) * 100):.1f}% of total)\n"
            
        if stats["null_count"] > 0:
            insights_str += f"- Contains nulls: {stats['null_count']} ({(stats['null_count'] / (stats['non_null_count'] + stats['null_count']) * 100):.1f}% of total)\n"
            
        if stats["date_format"]:
            insights_str += f"- Date format: {stats['date_format']}\n"
    
    return insights_str

def format_query_analysis_for_prompt(history: List[QueryHistoryItem], question: str) -> str:
    """Format query analysis for inclusion in a prompt"""
    if not history:
        return ""
    
    # Use the QueryAnalyzer tool to get recommendations
    analysis_results = agent_tools["query_analyzer"].analyze_query_history(history, question)
    
    recommendations = analysis_results["recommendations"]
    similar_questions = analysis_results["similar_questions"]
    
    analysis_str = "Query history analysis:\n\n"
    
    # Add recommendations
    analysis_str += "Recommendations based on query history:\n"
    for recommendation in recommendations:
        analysis_str += f"- {recommendation}\n"
    
    # Add similar questions
    if similar_questions:
        analysis_str += "\nSimilar questions from history:\n"
        for i, item in enumerate(similar_questions, 1):
            analysis_str += f"{i}. Question: {item['question']}\n"
            analysis_str += f"   SQL: {item['sql']}\n\n"
    
    return analysis_str

@async_log_execution_time
async def generate_basic_sql(
    question: str, 
    schema: TableSchema, 
    additional_schemas: Optional[Dict[str, TableSchema]] = None
) -> NL2SQLResponse:
    """
    Generate SQL from natural language using the table schema.
    This is a simple one-shot generation approach.
    
    Args:
        question: The natural language question
        schema: The primary table schema
        additional_schemas: Optional dictionary of additional table schemas
        
    Returns:
        NL2SQLResponse with generated SQL and explanation
    """
    logger.info(f"Generating basic SQL for question: {question}")
    
    if USE_MOCK_RESPONSES:
        logger.debug("Using mock response for basic SQL generation")
        # Return mock response
        await asyncio.sleep(1)  # Simulate API delay
        return NL2SQLResponse(
            sql="SELECT COUNT(*) FROM accident",
            explanation="This is a mock response for demonstration purposes."
        )
    
    try:
        with PerformanceTracker("format_schema_for_prompt"):
            primary_schema_text = format_schema_for_prompt(schema)
            
            additional_schemas_text = ""
            if additional_schemas:
                for table_name, table_schema in additional_schemas.items():
                    additional_schemas_text += f"\nAdditional Table Schema:\n{format_schema_for_prompt(table_schema)}\n"
        
        prompt = f"""
        You are an expert SQL query generator. Your task is to convert natural language questions into SQL queries.
        
        Primary Table Schema:
        {primary_schema_text}
        {additional_schemas_text}
        
        Question: {question}
        
        Generate a SQL query that answers this question. The query should be valid SQL that can be executed against the provided schema.
        If you need to join tables, make sure to use the correct join conditions based on the table schemas.
        
        SQL Query:
        """
        
        logger.debug(f"Basic SQL generation prompt: {prompt[:500]}...")
        
        with PerformanceTracker("gemini_api_call"):
            response_text = await generate_text(prompt, max_output_tokens=VERTEX_AI_MAX_OUTPUT_TOKENS)
        
        logger.debug(f"Basic SQL generation response: {response_text}")
        
        # Extract SQL and explanation from response
        sql = ""
        explanation = ""
        
        if "SQL:" in response_text:
            parts = response_text.split("SQL:", 1)
            if len(parts) > 1:
                sql_and_explanation = parts[1].strip()
                if "Explanation:" in sql_and_explanation:
                    sql_parts = sql_and_explanation.split("Explanation:", 1)
                    sql = sql_parts[0].strip()
                    explanation = sql_parts[1].strip()
                else:
                    sql = sql_and_explanation
        
        # If SQL is still empty, try to extract from code blocks with triple backticks
        if not sql and "```sql" in response_text.lower():
            # Extract SQL from code blocks
            pattern = r"```sql\s*(.*?)\s*```"
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                sql = matches[0].strip()
                
                # Try to extract explanation after the code block
                if "Explanation:" in response_text:
                    explanation_parts = response_text.split("Explanation:", 1)
                    explanation = explanation_parts[1].strip()
                elif "explanation:" in response_text.lower():
                    explanation_parts = re.split(r"explanation:", response_text, flags=re.IGNORECASE, maxsplit=1)
                    if len(explanation_parts) > 1:
                        explanation = explanation_parts[1].strip()
        
        # If SQL is still empty, try to extract from code blocks without sql tag
        if not sql and "```" in response_text:
            pattern = r"```\s*(.*?)\s*```"
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                # Use the first code block as SQL
                sql = matches[0].strip()
                
                # Try to extract explanation after the code block
                if "Explanation:" in response_text:
                    explanation_parts = response_text.split("Explanation:", 1)
                    explanation = explanation_parts[1].strip()
                elif "explanation:" in response_text.lower():
                    explanation_parts = re.split(r"explanation:", response_text, flags=re.IGNORECASE, maxsplit=1)
                    if len(explanation_parts) > 1:
                        explanation = explanation_parts[1].strip()
        
        if not sql:
            logger.warning("Failed to extract SQL from response, using fallback")
            logger.debug(f"Full response that failed SQL extraction: {response_text}")
            sql = f"SELECT * FROM {schema.table_name} LIMIT 10"
            explanation = "Failed to generate SQL. This is a fallback query."
        
        return NL2SQLResponse(
            sql=sql,
            explanation=explanation
        )
    except Exception as e:
        logger.exception(f"Error generating basic SQL: {str(e)}")
        return NL2SQLResponse(
            sql=f"SELECT * FROM {schema.table_name} LIMIT 10",
            explanation=f"Error generating SQL: {str(e)}"
        )

@async_log_execution_time
async def generate_enhanced_sql(
    question: str, 
    schema: TableSchema, 
    additional_schemas: Optional[Dict[str, TableSchema]] = None,
    table_description: Optional[str] = None,
    column_descriptions: Optional[Dict[str, str]] = None,
    examples: Optional[List[Dict[str, str]]] = None
) -> NL2SQLResponse:
    """
    Generate SQL from natural language using a LangGraph agent.
    This agent takes the schema, generates SQL, evaluates it, and corrects it if needed.
    
    Args:
        question: The natural language question
        schema: The primary table schema
        additional_schemas: Optional dictionary of additional table schemas
        table_description: Optional description of the primary table
        column_descriptions: Optional descriptions for columns
        examples: Optional example queries and results
        
    Returns:
        NL2SQLResponse with generated SQL and explanation
    """
    logger.info(f"Generating enhanced SQL for question: {question}")
    
    if USE_MOCK_RESPONSES:
        logger.debug("Using mock response for enhanced SQL generation")
        # Return mock response
        await asyncio.sleep(1.5)  # Simulate API delay
        return NL2SQLResponse(
            sql="SELECT COUNT(*) FROM accident WHERE severity_code = 1",
            explanation="This is a mock response for demonstration purposes. In enhanced mode, I would use a LangGraph agent to generate, evaluate, and correct SQL."
        )
    
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.prebuilt import ToolNode
        import operator
        
        # Define the agent state
        class AgentState(TypedDict):
            question: str
            schema: str
            additional_schemas: str
            sql: str
            evaluation: str
            explanation: str
            iterations: int
            final_sql: Optional[str]
        
        # Format the schema for the prompt
        with PerformanceTracker("format_schema_for_prompt"):
            primary_schema_text = format_schema_for_prompt(schema)
            
            # Format additional schemas if provided
            additional_schemas_text = ""
            if additional_schemas:
                for table_name, table_schema in additional_schemas.items():
                    additional_schemas_text += f"\nAdditional Table Schema:\n{format_schema_for_prompt(table_schema)}\n"
        
        # Add table and column descriptions to the schema text if provided
        if table_description:
            primary_schema_text = f"Table Description: {table_description}\n\n{primary_schema_text}"
        
        if column_descriptions:
            column_desc_text = "Column Descriptions:\n"
            for col_name, desc in column_descriptions.items():
                column_desc_text += f"- {col_name}: {desc}\n"
            primary_schema_text = f"{primary_schema_text}\n\n{column_desc_text}"
        
        # Initialize state
        initial_state = {
            "question": question,
            "schema": primary_schema_text,
            "additional_schemas": additional_schemas_text,
            "sql": "",
            "evaluation": "",
            "explanation": "",
            "iterations": 0,
            "final_sql": None
        }
        
        # Define nodes
        async def generate_sql(state: AgentState) -> AgentState:
            """Generate SQL based on the question and schema"""
            try:
                prompt = f"""
                You are an expert SQL query generator. Your task is to convert natural language questions into SQL queries.
                
                Table Schema:
                {state['schema']}
                
                {state['additional_schemas']}
                
                User Question: {state['question']}
                
                Generate a SQL query that answers the user's question. The response should be in the following format:
                
                SQL: <your SQL query>
                Explanation: <explanation of how the SQL query works>
                
                IMPORTANT: Do NOT use markdown code blocks with triple backticks. Instead, use the exact format shown above with "SQL:" prefix followed by your query.
                """
                
                response_text = await generate_text(prompt, max_output_tokens=VERTEX_AI_MAX_OUTPUT_TOKENS)
                
                # Extract SQL and explanation
                sql = ""
                explanation = ""
                
                # Log the raw response for debugging
                logger.debug(f"Raw response in generate_sql: {response_text}")
                
                # First try to extract using the "SQL:" prefix format
                if "SQL:" in response_text:
                    parts = response_text.split("SQL:", 1)
                    if len(parts) > 1:
                        sql_and_explanation = parts[1].strip()
                        if "Explanation:" in sql_and_explanation:
                            sql_parts = sql_and_explanation.split("Explanation:", 1)
                            sql = sql_parts[0].strip()
                            explanation = sql_parts[1].strip()
                        else:
                            sql = sql_and_explanation
                
                # If SQL is still empty, try to extract from code blocks with triple backticks
                if not sql and "```sql" in response_text.lower():
                    # Extract SQL from code blocks
                    pattern = r"```sql\s*(.*?)\s*```"
                    matches = re.findall(pattern, response_text, re.DOTALL)
                    if matches:
                        sql = matches[0].strip()
                        
                        # Try to extract explanation after the code block
                        if "Explanation:" in response_text:
                            explanation_parts = response_text.split("Explanation:", 1)
                            explanation = explanation_parts[1].strip()
                        elif "explanation:" in response_text.lower():
                            explanation_parts = re.split(r"explanation:", response_text, flags=re.IGNORECASE, maxsplit=1)
                            if len(explanation_parts) > 1:
                                explanation = explanation_parts[1].strip()
                
                # If SQL is still empty, try to extract from code blocks without sql tag
                if not sql and "```" in response_text:
                    pattern = r"```\s*(.*?)\s*```"
                    matches = re.findall(pattern, response_text, re.DOTALL)
                    if matches:
                        # Use the first code block as SQL
                        sql = matches[0].strip()
                        
                        # Try to extract explanation after the code block
                        if "Explanation:" in response_text:
                            explanation_parts = response_text.split("Explanation:", 1)
                            explanation = explanation_parts[1].strip()
                        elif "explanation:" in response_text.lower():
                            explanation_parts = re.split(r"explanation:", response_text, flags=re.IGNORECASE, maxsplit=1)
                            if len(explanation_parts) > 1:
                                explanation = explanation_parts[1].strip()
                
                if not sql:
                    logger.warning("Failed to extract SQL from model response")
                    logger.debug(f"Full response that failed SQL extraction: {response_text}")
                    explanation = "The model failed to generate a valid SQL query. Please try rephrasing your question."
                
                return {
                    **state,
                    "sql": sql,
                    "explanation": explanation,
                    "iterations": state["iterations"] + 1
                }
            except Exception as e:
                logger.exception(f"Error in generate_sql node: {str(e)}")
                return {
                    **state,
                    "sql": "",
                    "explanation": f"Error generating SQL: {str(e)}",
                    "iterations": state["iterations"] + 1
                }
        
        async def evaluate_sql(state: AgentState) -> AgentState:
            """Evaluate the generated SQL for correctness"""
            prompt = f"""
            You are an expert SQL reviewer. Your task is to evaluate if the SQL query correctly answers the user's question.
            
            Table Schema:
            {state['schema']}
            
            {state['additional_schemas']}
            
            User Question: {state['question']}
            
            Generated SQL:
            {state['sql']}
            
            Evaluate if the SQL query correctly answers the user's question. Consider:
            1. Does it use the correct tables and columns?
            2. Does it handle the filtering conditions correctly?
            3. Does it perform the right aggregations or calculations?
            4. Are there any syntax errors or logical issues?
            
            Evaluation:
            """
            
            response_text = await generate_text(prompt, max_output_tokens=VERTEX_AI_MAX_OUTPUT_TOKENS)
            
            return {
                **state,
                "evaluation": response_text
            }
        
        async def correct_sql(state: AgentState) -> AgentState:
            """Correct the SQL based on evaluation"""
            prompt = f"""
            You are an expert SQL query generator. Your task is to correct a SQL query based on evaluation feedback.
            
            Table Schema:
            {state['schema']}
            
            {state['additional_schemas']}
            
            User Question: {state['question']}
            
            Original SQL:
            {state['sql']}
            
            Evaluation:
            {state['evaluation']}
            
            Please correct the SQL query to address the issues identified in the evaluation.
            
            Corrected SQL: 
            
            IMPORTANT: Do NOT use markdown code blocks with triple backticks. Instead, provide your corrected SQL directly after "Corrected SQL:" without any formatting.
            """
            
            response_text = await generate_text(prompt, max_output_tokens=VERTEX_AI_MAX_OUTPUT_TOKENS)
            
            # Log the raw response for debugging
            logger.debug(f"Raw response in correct_sql: {response_text}")
            
            # Extract SQL and explanation
            sql = ""
            explanation = ""
            
            # First try to extract using the "SQL:" prefix format
            if "SQL:" in response_text:
                parts = response_text.split("SQL:", 1)
                if len(parts) > 1:
                    sql_and_explanation = parts[1].strip()
                    if "Explanation:" in sql_and_explanation:
                        sql_parts = sql_and_explanation.split("Explanation:", 1)
                        sql = sql_parts[0].strip()
                        explanation = sql_parts[1].strip()
                    else:
                        sql = sql_and_explanation
            
            # If SQL is still empty, try to extract from code blocks with triple backticks
            if not sql and "```sql" in response_text.lower():
                # Extract SQL from code blocks
                pattern = r"```sql\s*(.*?)\s*```"
                matches = re.findall(pattern, response_text, re.DOTALL)
                if matches:
                    sql = matches[0].strip()
                    
                    # Try to extract explanation after the code block
                    if "Explanation:" in response_text:
                        explanation_parts = response_text.split("Explanation:", 1)
                        explanation = explanation_parts[1].strip()
                    elif "explanation:" in response_text.lower():
                        explanation_parts = re.split(r"explanation:", response_text, flags=re.IGNORECASE, maxsplit=1)
                        if len(explanation_parts) > 1:
                            explanation = explanation_parts[1].strip()
            
            # If SQL is still empty, try to extract from code blocks without sql tag
            if not sql and "```" in response_text:
                pattern = r"```\s*(.*?)\s*```"
                matches = re.findall(pattern, response_text, re.DOTALL)
                if matches:
                    # Use the first code block as SQL
                    sql = matches[0].strip()
                    
                    # Try to extract explanation after the code block
                    if "Explanation:" in response_text:
                        explanation_parts = response_text.split("Explanation:", 1)
                        explanation = explanation_parts[1].strip()
                    elif "explanation:" in response_text.lower():
                        explanation_parts = re.split(r"explanation:", response_text, flags=re.IGNORECASE, maxsplit=1)
                        if len(explanation_parts) > 1:
                            explanation = explanation_parts[1].strip()
            
            if not sql:
                logger.warning("Failed to extract corrected SQL from model response")
                logger.debug(f"Full response that failed SQL extraction: {response_text}")
                # Keep the original SQL if we couldn't extract a corrected version
                sql = state["sql"]
                explanation = "The model failed to generate a corrected SQL query."
            
            return {
                **state,
                "sql": sql,
                "explanation": explanation,
                "iterations": state["iterations"] + 1
            }
        
        async def finalize(state: AgentState) -> AgentState:
            """Finalize the SQL query"""
            return {
                **state,
                "final_sql": state["sql"]
            }
        
        # Define edges
        def should_correct(state: AgentState) -> str:
            """Determine if SQL needs correction based on evaluation"""
            if "Correct: No" in state["evaluation"] or "correct: no" in state["evaluation"].lower():
                if state["iterations"] < 3:  # Limit iterations to prevent infinite loops
                    return "correct"
            return "finalize"
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("generate", generate_sql)
        workflow.add_node("evaluate", evaluate_sql)
        workflow.add_node("correct", correct_sql)
        workflow.add_node("finalize", finalize)
        
        # Add edges
        workflow.add_edge("generate", "evaluate")
        workflow.add_conditional_edges(
            "evaluate",
            should_correct,
            {
                "correct": "correct",
                "finalize": "finalize"
            }
        )
        workflow.add_edge("correct", "evaluate")
        workflow.add_edge("finalize", END)
        
        # Set entry point
        workflow.set_entry_point("generate")
        
        # Compile the graph
        app = workflow.compile()
        
        # Run the graph
        logger.info("Running LangGraph agent for enhanced SQL generation")
        try:
            result = await app.ainvoke(initial_state)
            
            # Extract final SQL and explanation
            final_sql = result.get("final_sql", "")
            explanation = result.get("explanation", "")
            
            if not final_sql:
                logger.warning("Failed to generate SQL with LangGraph agent, using fallback")
                final_sql = f"SELECT * FROM {schema.table_name} LIMIT 10"
                explanation = "Failed to generate SQL. This is a fallback query."
                
                # Add more detailed error information if available
                if "error" in result:
                    explanation += f"\n\nError details: {result.get('error')}"
            
            # Add information about the agent process to the explanation
            agent_process = f"\n\nEnhanced SQL Generation Process:\n"
            agent_process += f"- Initial SQL generation\n"
            agent_process += f"- Evaluation of SQL correctness\n"
            if result.get("iterations", 0) > 1:
                agent_process += f"- Correction based on evaluation feedback ({result.get('iterations', 0) - 1} iterations)\n"
            agent_process += f"- Final SQL query generation"
            
            enhanced_explanation = explanation + agent_process
            
            return NL2SQLResponse(
                sql=final_sql,
                explanation=enhanced_explanation
            )
        except Exception as e:
            logger.exception(f"Error running LangGraph agent: {str(e)}")
            return NL2SQLResponse(
                sql=f"SELECT * FROM {schema.table_name} LIMIT 10",
                explanation=f"Error generating SQL: {str(e)}"
            )
    except Exception as e:
        logger.exception(f"Error generating enhanced SQL: {str(e)}")
        return NL2SQLResponse(
            sql=f"SELECT * FROM {schema.table_name} LIMIT 10",
            explanation=f"Error generating SQL: {str(e)}"
        )

@async_log_execution_time
async def generate_agent_sql(
    question: str, 
    schema: TableSchema, 
    table_description: Optional[str] = None,
    column_descriptions: Optional[Dict[str, str]] = None,
    data_sample: Optional[List[Dict[str, Any]]] = None,
    previous_queries: Optional[List[QueryHistoryItem]] = None,
    use_langgraph: bool = False,
    num_candidates: Optional[int] = None,
    temperatures: Optional[List[float]] = None
) -> NL2SQLResponse:
    """
    Generate SQL from natural language using an agent-based approach.
    
    Args:
        question: The natural language question
        schema: The table schema
        table_description: Optional description of the table
        column_descriptions: Optional descriptions for columns
        data_sample: Optional sample data from the table
        previous_queries: List of previously run queries on this table
        use_langgraph: Whether to use the LangGraph agent
        num_candidates: Number of candidate queries to generate (for LangGraph agent)
        temperatures: List of temperatures to use for each candidate (for LangGraph agent)
        
    Returns:
        NL2SQLResponse with generated SQL and explanation
    """
    logger.info(f"Generating SQL with agent for question: {question}")
    
    # If use_langgraph is True, use the LangGraph agent
    if use_langgraph:
        logger.info("Using LangGraph agent for SQL generation")
        
        # Use default values from config if not provided
        if num_candidates is None:
            from app.core.config import LANGGRAPH_NUM_CANDIDATES
            num_candidates = LANGGRAPH_NUM_CANDIDATES
            
        if temperatures is None:
            from app.core.config import LANGGRAPH_TEMPERATURES
            temperatures = LANGGRAPH_TEMPERATURES
            
        logger.info(f"Using {num_candidates} candidates with temperatures: {temperatures}")
        
        return await LangGraphAgent.generate_sql(
            question=question,
            schema=schema,
            additional_schemas=None,
            table_description=table_description,
            column_descriptions=column_descriptions,
            data_sample=data_sample,
            previous_queries=previous_queries,
            generate_text_func=generate_text,
            num_candidates=num_candidates,
            temperatures=temperatures
        )
    
    try:
        # Format schema for prompt
        with PerformanceTracker("format_schema_for_prompt"):
            schema_text = format_schema_for_prompt(schema)
        
        # Step 0: Rewrite the question to better match table metadata
        logger.info("Rewriting question to better match table metadata")
        rewritten_question = await QueryRewriter.rewrite_query(
            question=question,
            schema=schema,
            generate_text_func=generate_text
        )
        logger.info(f"Rewritten question: {rewritten_question}")
        
        # Step 1: Analyze previous queries and suggest the best one
        best_previous_query = None
        previous_query_analysis = ""
        
        if previous_queries and len(previous_queries) > 0:
            logger.info(f"Analyzing {len(previous_queries)} previous queries")
            
            # Format previous queries for the prompt
            previous_queries_text = "Previously Run Queries:\n"
            for i, query in enumerate(previous_queries):
                previous_queries_text += f"Query {i+1}:\n"
                previous_queries_text += f"Question: {query.question}\n"
                previous_queries_text += f"SQL: {query.sql}\n"
                if hasattr(query, 'result') and query.result:
                    previous_queries_text += f"Result: {query.result}\n"
                previous_queries_text += "\n"
            
            # Generate prompt to analyze previous queries
            prompt = f"""
            You are an expert SQL analyst. Your task is to analyze previously run queries and suggest the best one to adapt for a new question.
            
            Table Schema:
            {schema_text}
            
            {previous_queries_text}
            
            New Question: {rewritten_question}
            
            Analyze the previous queries and determine which one would be the best starting point to answer the new question.
            Provide your response in the following format:
            
            Best Query: [Query number]
            Reasoning: [Explain why this query is the best starting point]
            Suggested Modifications: [Describe how the query should be modified to answer the new question]
            """
            
            analysis_response = await generate_text(prompt, max_output_tokens=VERTEX_AI_MAX_OUTPUT_TOKENS)
            previous_query_analysis = analysis_response
            
            # Extract the best query number
            best_query_num = None
            if "Best Query:" in analysis_response:
                best_query_line = analysis_response.split("Best Query:", 1)[1].split("\n", 1)[0].strip()
                try:
                    # Extract the query number (assuming format like "Query 3" or just "3")
                    if "Query" in best_query_line:
                        best_query_num = int(best_query_line.replace("Query", "").strip()) - 1
                    else:
                        best_query_num = int(best_query_line) - 1
                    
                    if 0 <= best_query_num < len(previous_queries):
                        best_previous_query = previous_queries[best_query_num]
                        logger.info(f"Selected query {best_query_num + 1} as the best starting point")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to extract best query number: {str(e)}")
        
        # Step 2: Run multiple approaches in parallel
        tasks = []
        
        # Approach 1: Basic generation with schema only
        tasks.append(generate_basic_sql(rewritten_question, schema))
        
        # Approach 2: Enhanced generation with schema and metadata
        tasks.append(generate_enhanced_sql(
            rewritten_question, 
            schema, 
            table_description=table_description,
            column_descriptions=column_descriptions
        ))
        
        # Approach 3: If we have a best previous query, generate a modified version
        if best_previous_query:
            # Create a prompt to modify the best previous query
            prompt = f"""
            You are an expert SQL query generator. Your task is to modify an existing SQL query to answer a new question.
            
            Table Schema:
            {schema_text}
            
            Previous Question: {best_previous_query.question}
            Previous SQL: {best_previous_query.sql}
            
            New Question: {rewritten_question}
            
            Modify the previous SQL query to answer the new question. The response should be in the following format:
            
            SQL: <your modified SQL query>
            Explanation: <explanation of the modifications and how the SQL query works>
            """
            
            # Generate the modified query
            async def generate_modified_query():
                response_text = await generate_text(prompt, max_output_tokens=VERTEX_AI_MAX_OUTPUT_TOKENS)
                
                # Extract SQL and explanation
                sql = ""
                explanation = ""
                
                if "SQL:" in response_text:
                    parts = response_text.split("SQL:", 1)
                    if len(parts) > 1:
                        sql_and_explanation = parts[1].strip()
                        if "Explanation:" in sql_and_explanation:
                            sql_parts = sql_and_explanation.split("Explanation:", 1)
                            sql = sql_parts[0].strip()
                            explanation = sql_parts[1].strip()
                        else:
                            sql = sql_and_explanation
                
                return NL2SQLResponse(
                    sql=sql,
                    explanation=explanation
                )
            
            tasks.append(generate_modified_query())
        
        # Approach 4: If we have data samples, use them for generation
        if data_sample and len(data_sample) > 0:
            # Format data sample for the prompt
            sample_text = "Data Sample:\n"
            # Convert to list of dictionaries if it's not already
            sample_dicts = data_sample
            
            # Format as a table with headers and values
            if len(sample_dicts) > 0:
                # Get column names
                columns = list(sample_dicts[0].keys())
                
                # Add headers
                sample_text += "| " + " | ".join(columns) + " |\n"
                sample_text += "| " + " | ".join(["---" for _ in columns]) + " |\n"
                
                # Add rows (limit to 5 rows to keep prompt size reasonable)
                for row in sample_dicts[:5]:
                    sample_text += "| " + " | ".join([str(row.get(col, "")) for col in columns]) + " |\n"
            
            # Create a prompt that includes the data sample
            prompt = f"""
            You are an expert SQL query generator. Your task is to convert natural language questions into SQL queries.
            
            Table Schema:
            {schema_text}
            
            {sample_text}
            
            User Question: {rewritten_question}
            
            Generate a SQL query that answers the user's question. The response should be in the following format:
            
            SQL: <your SQL query>
            Explanation: <explanation of how the SQL query works>
            """
            
            # Generate the query with data sample
            async def generate_with_sample():
                response_text = await generate_text(prompt, max_output_tokens=VERTEX_AI_MAX_OUTPUT_TOKENS)
                
                # Extract SQL and explanation
                sql = ""
                explanation = ""
                
                if "SQL:" in response_text:
                    parts = response_text.split("SQL:", 1)
                    if len(parts) > 1:
                        sql_and_explanation = parts[1].strip()
                        if "Explanation:" in sql_and_explanation:
                            sql_parts = sql_and_explanation.split("Explanation:", 1)
                            sql = sql_parts[0].strip()
                            explanation = sql_parts[1].strip()
                        else:
                            sql = sql_and_explanation
                
                return NL2SQLResponse(
                    sql=sql,
                    explanation=explanation
                )
            
            tasks.append(generate_with_sample())
        
        # Execute all tasks in parallel
        logger.info(f"Running {len(tasks)} parallel approaches for SQL generation")
        results = await asyncio.gather(*tasks)
        
        # Step 3: Select the best result
        # Create a prompt to evaluate and select the best SQL query
        sql_options = []
        for i, result in enumerate(results):
            if result and result.sql:
                sql_options.append({
                    "index": i,
                    "sql": result.sql,
                    "explanation": result.explanation
                })
        
        if not sql_options:
            logger.warning("No valid SQL queries generated")
            return NL2SQLResponse(
                sql=f"SELECT * FROM {schema.table_name} LIMIT 10",
                explanation="Failed to generate SQL. This is a fallback query."
            )
        
        # If we only have one valid result, return it
        if len(sql_options) == 1:
            logger.info("Only one valid SQL query generated, returning it")
            return NL2SQLResponse(
                sql=sql_options[0]["sql"],
                explanation=sql_options[0]["explanation"]
            )
        
        # Create a prompt to evaluate and select the best SQL query
        evaluation_prompt = f"""
        You are an expert SQL reviewer. Your task is to evaluate multiple SQL queries and select the best one to answer a question.
        
        Table Schema:
        {schema_text}
        
        User Question: {rewritten_question}
        
        SQL Options:
        """
        
        for i, option in enumerate(sql_options):
            evaluation_prompt += f"\nOption {i+1}:\nSQL: {option['sql']}\nExplanation: {option['explanation']}\n"
        
        evaluation_prompt += """
        Evaluate each SQL query based on the following criteria:
        1. Correctness: Does it correctly answer the question?
        2. Efficiency: Is it optimized and efficient?
        3. Readability: Is it clear and easy to understand?
        4. Completeness: Does it include all necessary information?
        
        Provide your evaluation in the following format:
        
        Best Option: [Option number]
        Reasoning: [Explain why this option is the best]
        """
        
        # Generate the evaluation
        evaluation_response = await generate_text(evaluation_prompt, max_output_tokens=VERTEX_AI_MAX_OUTPUT_TOKENS)
        
        # Extract the best option
        best_option_index = 0  # Default to the first option
        if "Best Option:" in evaluation_response:
            best_option_line = evaluation_response.split("Best Option:", 1)[1].split("\n", 1)[0].strip()
            try:
                # Extract the option number (assuming format like "Option 2" or just "2")
                if "Option" in best_option_line:
                    best_option_num = int(best_option_line.replace("Option", "").strip())
                else:
                    best_option_num = int(best_option_line)
                
                best_option_index = best_option_num - 1
                if best_option_index < 0 or best_option_index >= len(sql_options):
                    best_option_index = 0
            except ValueError:
                logger.warning(f"Failed to extract best option number, defaulting to first option")
        
        # Get the best SQL query
        best_option = sql_options[best_option_index]
        
        # Extract reasoning from evaluation
        reasoning = ""
        if "Reasoning:" in evaluation_response:
            reasoning = evaluation_response.split("Reasoning:", 1)[1].strip()
        
        # Combine the explanation with the evaluation reasoning
        combined_explanation = best_option["explanation"]
        
        # Add information about the agent process
        agent_process = "\n\nAdvanced SQL Generation Process:\n"
        agent_process += f"- Rewritten the original question to better match table metadata\n"
        agent_process += f"- Generated {len(sql_options)} SQL queries using different approaches\n"
        if best_previous_query:
            agent_process += f"- Analyzed {len(previous_queries)} previous queries and selected the best one as a starting point\n"
        agent_process += f"- Evaluated all generated queries and selected the best one (Option {best_option_index + 1})\n"
        if reasoning:
            agent_process += f"\nSelection Reasoning:\n{reasoning}"
        
        combined_explanation += agent_process
        
        return NL2SQLResponse(
            sql=best_option["sql"],
            explanation=combined_explanation,
            rewritten_question=rewritten_question
        )
    except Exception as e:
        logger.exception(f"Error generating agent SQL: {str(e)}")
        return NL2SQLResponse(
            sql=f"SELECT * FROM {schema.table_name} LIMIT 10",
            explanation=f"Error generating SQL: {str(e)}"
        )

@async_log_execution_time
async def suggest_tables_for_question(
    question: str,
    all_table_schemas: Dict[str, TableSchema],
    table_descriptions: Dict[str, str] = {},
    demo_phase: str = "basic"
) -> List[str]:
    """
    Suggest tables that might be relevant for answering a natural language question.
    
    Args:
        question: The natural language question
        all_table_schemas: Dictionary of all available table schemas (table_name -> TableSchema)
        table_descriptions: Dictionary of table descriptions (table_name -> description)
        demo_phase: The demo phase (basic, enhanced, agent)
        
    Returns:
        List of suggested table names
    """
    logger.info(f"Suggesting tables for question: {question}")
    
    # Format all schemas for the prompt
    formatted_schemas = []
    for table_name, schema in all_table_schemas.items():
        schema_str = f"Table: {table_name}\n"
        
        # Add description if available and in enhanced or agent mode
        if (demo_phase in ["enhanced", "agent"]) and table_name in table_descriptions and table_descriptions[table_name]:
            schema_str += f"Description: {table_descriptions[table_name]}\n"
            
        schema_str += "Columns:\n"
        for column in schema.columns:
            schema_str += f"- {column.name} ({column.data_type})"
            # Include column descriptions in enhanced or agent mode
            if (demo_phase in ["enhanced", "agent"]) and column.description:
                schema_str += f": {column.description}"
            schema_str += "\n"
        
        formatted_schemas.append(schema_str)
    
    all_schemas_str = "\n".join(formatted_schemas)
    
    # Create the prompt - enhanced for better table selection
    if demo_phase in ["enhanced", "agent"]:
        prompt = f"""You are a database expert. Given a question and a list of available tables with their schemas and descriptions, 
suggest which tables would be most relevant to answer the question.

QUESTION: {question}

AVAILABLE TABLES WITH DESCRIPTIONS:
{all_schemas_str}

Based on the question and the available tables, analyze which tables would be most relevant for answering this question.
Consider the following:
1. The semantic meaning of the question and how it relates to table and column descriptions
2. Which tables contain the data needed to answer the question
3. If multiple tables might need to be joined to provide a complete answer

Return your answer as a comma-separated list of table names, without any additional text or explanation.
For example: "table1,table2,table3"
"""
    else:
        # Basic mode prompt without emphasizing descriptions
        prompt = f"""You are a database expert. Given a question and a list of available tables with their schemas, 
suggest which tables would be most relevant to answer the question.

QUESTION: {question}

AVAILABLE TABLES:
{all_schemas_str}

Based on the question and the available tables, list the names of the tables that would be most relevant 
for answering this question. Only include tables that are directly relevant to the question.
Return your answer as a comma-separated list of table names, without any additional text or explanation.
For example: "table1,table2,table3"
"""

    # Call the LLM
    try:
        response = await generate_text(prompt)
        
        # Parse the response to get table names
        # The response should be a comma-separated list of table names
        suggested_tables = [table.strip() for table in response.split(',')]
        
        # Filter out any tables that don't exist in our schema
        valid_tables = [table for table in suggested_tables if table in all_table_schemas]
        
        logger.info(f"Suggested tables: {valid_tables}")
        return valid_tables
    except Exception as e:
        logger.error(f"Error suggesting tables: {str(e)}")
        # Return an empty list in case of error
        return [] 