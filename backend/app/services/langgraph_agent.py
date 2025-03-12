import logging
import asyncio
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence, Tuple, Union, cast
from enum import Enum
import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.models.models import TableSchema, NL2SQLResponse
from app.services.query_rewriter import QueryRewriter
from app.core.config import VERTEX_AI_TEMPERATURE

# Configure logging
logger = logging.getLogger(__name__)

# Define the state for our LangGraph agent
class AgentState(TypedDict):
    """State for the SQL generation agent"""
    question: str  # Original question
    rewritten_question: str  # Rewritten question
    schema: TableSchema  # Table schema
    additional_schemas: Optional[Dict[str, TableSchema]]  # Additional table schemas
    table_description: Optional[str]  # Table description
    column_descriptions: Optional[Dict[str, str]]  # Column descriptions
    data_sample: Optional[List[Dict[str, Any]]]  # Sample data
    previous_queries: Optional[List[Dict[str, Any]]]  # Previous queries
    
    # Metrics understanding
    metrics_analysis: Optional[str]  # Analysis of metrics to calculate
    
    # Join understanding
    join_analysis: Optional[str]  # Analysis of tables to join
    
    # Filter understanding
    filter_analysis: Optional[str]  # Analysis of filters to apply
    
    # Grouping understanding
    grouping_analysis: Optional[str]  # Analysis of grouping to apply
    
    # SQL generation
    candidate_queries: List[Dict[str, Any]]  # List of candidate SQL queries
    
    # Evaluation
    evaluation_results: Optional[Dict[str, Any]]  # Evaluation results
    
    # Final result
    final_sql: Optional[str]  # Final SQL query
    explanation: Optional[str]  # Explanation of the SQL query
    
    # Messages for the agent
    messages: List[BaseMessage]  # Messages for the agent

# Define the nodes for our LangGraph agent
async def rewrite_question(state: AgentState, generate_text_func) -> AgentState:
    """Rewrite the question to better match table metadata"""
    logger.info(f"Rewriting question: {state['question']}")
    
    rewritten_question = await QueryRewriter.rewrite_query(
        question=state["question"],
        schema=state["schema"],
        additional_schemas=state.get("additional_schemas"),
        generate_text_func=generate_text_func
    )
    
    logger.info(f"Rewritten question: {rewritten_question}")
    
    # Update the state
    state["rewritten_question"] = rewritten_question
    state["messages"].append(HumanMessage(content=f"I need to analyze this question: {rewritten_question}"))
    
    return state

async def analyze_metrics(state: AgentState, generate_text_func) -> AgentState:
    """Analyze what metrics we're trying to calculate and suggest how they should be calculated"""
    logger.info(f"Analyzing metrics for question: {state['rewritten_question']}")
    
    # Format schema for prompt
    schema_text = format_schema_for_prompt(state["schema"])
    
    # Create the prompt
    prompt = f"""
    You are an expert SQL analyst. Your task is to analyze what metrics we're trying to calculate in this question and suggest how they should be calculated.
    
    Table Schema:
    {schema_text}
    
    Question: {state['rewritten_question']}
    
    Analyze what metrics we're trying to calculate in this question. Consider:
    1. What are the main metrics or measures being requested?
    2. How should these metrics be calculated (sum, count, average, etc.)?
    3. Are there any complex calculations or derived metrics needed?
    4. What SQL functions would be appropriate for these calculations?
    
    Provide a detailed analysis of the metrics and how they should be calculated.
    """
    
    # Generate the metrics analysis
    metrics_analysis = await generate_text_func(prompt)
    
    # Update the state
    state["metrics_analysis"] = metrics_analysis
    state["messages"].append(AIMessage(content=f"Metrics Analysis: {metrics_analysis}"))
    
    return state

async def analyze_joins(state: AgentState, generate_text_func) -> AgentState:
    """Analyze how and what tables should be joined"""
    logger.info(f"Analyzing joins for question: {state['rewritten_question']}")
    
    # Format schema for prompt
    schema_text = format_schema_for_prompt(state["schema"])
    
    # Format additional schemas if available
    additional_schemas_text = ""
    if state.get("additional_schemas"):
        for table_name, schema in state["additional_schemas"].items():
            additional_schemas_text += format_schema_for_prompt(schema) + "\n"
    
    # Create the prompt
    prompt = f"""
    You are an expert SQL analyst. Your task is to analyze how and what tables should be joined to answer this question.
    
    Primary Table Schema:
    {schema_text}
    
    {"Additional Table Schemas:" if additional_schemas_text else ""}
    {additional_schemas_text}
    
    Question: {state['rewritten_question']}
    
    Analyze what tables need to be joined to answer this question. Consider:
    1. What tables contain the necessary data for the question?
    2. What are the join conditions between these tables?
    3. What type of joins should be used (INNER, LEFT, RIGHT, FULL)?
    4. Are there any potential issues with the joins (e.g., many-to-many relationships)?
    
    Provide a detailed analysis of the tables to join and how they should be joined.
    """
    
    # Generate the join analysis
    join_analysis = await generate_text_func(prompt)
    
    # Update the state
    state["join_analysis"] = join_analysis
    state["messages"].append(AIMessage(content=f"Join Analysis: {join_analysis}"))
    
    return state

async def analyze_filters(state: AgentState, generate_text_func) -> AgentState:
    """Analyze what filters should be applied"""
    logger.info(f"Analyzing filters for question: {state['rewritten_question']}")
    
    # Format schema for prompt
    schema_text = format_schema_for_prompt(state["schema"])
    
    # Create the prompt
    prompt = f"""
    You are an expert SQL analyst. Your task is to analyze what filters should be applied to answer this question.
    
    Table Schema:
    {schema_text}
    
    Question: {state['rewritten_question']}
    
    Analyze what filters should be applied to answer this question. Consider:
    1. What conditions or criteria are specified in the question?
    2. What columns should be used in the WHERE clause?
    3. Are there any complex filter conditions (e.g., nested conditions, OR/AND logic)?
    4. Are there any date/time filters needed?
    5. Are there any numeric range filters needed?
    
    Provide a detailed analysis of the filters that should be applied.
    """
    
    # Generate the filter analysis
    filter_analysis = await generate_text_func(prompt)
    
    # Update the state
    state["filter_analysis"] = filter_analysis
    state["messages"].append(AIMessage(content=f"Filter Analysis: {filter_analysis}"))
    
    return state

async def analyze_grouping(state: AgentState, generate_text_func) -> AgentState:
    """Analyze how we should group the tables"""
    logger.info(f"Analyzing grouping for question: {state['rewritten_question']}")
    
    # Format schema for prompt
    schema_text = format_schema_for_prompt(state["schema"])
    
    # Create the prompt
    prompt = f"""
    You are an expert SQL analyst. Your task is to analyze how we should group the data to answer this question.
    
    Table Schema:
    {schema_text}
    
    Question: {state['rewritten_question']}
    
    Analyze how we should group the data to answer this question. Consider:
    1. What columns should be used in the GROUP BY clause?
    2. Are there any aggregations that require grouping?
    3. Is there a need for HAVING clauses to filter grouped results?
    4. Should the results be ordered in a specific way?
    
    Provide a detailed analysis of how the data should be grouped.
    """
    
    # Generate the grouping analysis
    grouping_analysis = await generate_text_func(prompt)
    
    # Update the state
    state["grouping_analysis"] = grouping_analysis
    state["messages"].append(AIMessage(content=f"Grouping Analysis: {grouping_analysis}"))
    
    return state

async def generate_candidate_queries(state: AgentState, generate_text_func, num_candidates: int = 1, temperatures: Optional[List[float]] = None) -> AgentState:
    """Generate candidate SQL queries based on the analyses"""
    logger.info(f"Generating {num_candidates} candidate SQL queries for question: {state['rewritten_question']}")
    
    # Use default temperature if none provided
    if not temperatures:
        temperatures = [VERTEX_AI_TEMPERATURE] * num_candidates
    elif len(temperatures) < num_candidates:
        # Extend with default temperature if list is too short
        temperatures.extend([VERTEX_AI_TEMPERATURE] * (num_candidates - len(temperatures)))
    
    # Format schema for prompt
    schema_text = format_schema_for_prompt(state["schema"])
    
    # Initialize candidate queries list
    if not state.get("candidate_queries"):
        state["candidate_queries"] = []
    
    # Generate multiple candidates with different temperatures
    for i in range(num_candidates):
        temperature = temperatures[i]
        logger.info(f"Generating candidate query {i+1} with temperature {temperature}")
        
        # Create the prompt
        prompt = f"""
        You are an expert SQL query generator. Your task is to generate a SQL query that answers this question based on the provided analyses.
        
        Table Schema:
        {schema_text}
        
        Question: {state['rewritten_question']}
        
        Metrics Analysis:
        {state.get('metrics_analysis', 'No metrics analysis available.')}
        
        Join Analysis:
        {state.get('join_analysis', 'No join analysis available.')}
        
        Filter Analysis:
        {state.get('filter_analysis', 'No filter analysis available.')}
        
        Grouping Analysis:
        {state.get('grouping_analysis', 'No grouping analysis available.')}
        
        Generate a SQL query that answers the question based on these analyses. The response should be in the following format:
        
        SQL: <your SQL query>
        Explanation: <explanation of how the SQL query works>
        
        IMPORTANT INSTRUCTIONS:
        1. The SQL query should be valid BigQuery SQL.
        2. DO NOT include any backticks (`) or triple backticks (```) in your response.
        3. DO NOT use markdown formatting or code block markers.
        4. DO NOT prefix the SQL with "sql" or any other language indicator.
        5. Start your SQL query directly with keywords like SELECT, WITH, etc.
        """
        
        # Generate the SQL query with the specific temperature
        response_text = await generate_text_func(prompt, temperature=temperature)
        
        # Extract SQL and explanation using the dedicated function
        sql, explanation = extract_sql_from_response(response_text)
        
        # Clean the SQL query
        sql = clean_sql_query(sql)
        
        # Log the extracted SQL for debugging
        logger.info(f"Extracted SQL query {i+1} (temp={temperature}): {sql}")
        
        # Add to candidate queries
        state["candidate_queries"].append({
            "sql": sql,
            "explanation": explanation,
            "source": f"candidate_{i+1}",
            "temperature": temperature
        })
        
        state["messages"].append(AIMessage(content=f"Generated SQL Query {i+1} (temp={temperature}): {sql}\n\nExplanation: {explanation}"))
    
    return state

async def evaluate_queries(state: AgentState, generate_text_func) -> AgentState:
    """Evaluate the candidate SQL queries"""
    logger.info(f"Evaluating candidate SQL queries for question: {state['rewritten_question']}")
    
    if not state.get("candidate_queries") or len(state["candidate_queries"]) == 0:
        logger.warning("No candidate queries to evaluate")
        state["evaluation_results"] = {"error": "No candidate queries to evaluate"}
        return state
    
    # Format schema for prompt
    schema_text = format_schema_for_prompt(state["schema"])
    
    # Create the prompt
    evaluation_prompt = f"""
    You are an expert SQL reviewer. Your task is to evaluate multiple SQL queries and select the best one to answer a question.
    
    Table Schema:
    {schema_text}
    
    Question: {state['rewritten_question']}
    
    SQL Options:
    """
    
    for i, option in enumerate(state["candidate_queries"]):
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
    Suggested Improvements: [Any improvements that could be made to the best option]
    """
    
    # Generate the evaluation
    evaluation_response = await generate_text_func(evaluation_prompt)
    
    # Extract the best option
    best_option_index = 0  # Default to the first option
    reasoning = ""
    suggested_improvements = ""
    
    if "Best Option:" in evaluation_response:
        best_option_line = evaluation_response.split("Best Option:", 1)[1].split("\n", 1)[0].strip()
        try:
            # Extract the option number (assuming format like "Option 2" or just "2")
            if "Option" in best_option_line:
                best_option_num = int(best_option_line.replace("Option", "").strip())
            else:
                best_option_num = int(best_option_line)
            
            best_option_index = best_option_num - 1
            if best_option_index < 0 or best_option_index >= len(state["candidate_queries"]):
                best_option_index = 0
        except ValueError:
            logger.warning(f"Failed to extract best option number, defaulting to first option")
    
    # Extract reasoning
    if "Reasoning:" in evaluation_response:
        reasoning_parts = evaluation_response.split("Reasoning:", 1)
        if len(reasoning_parts) > 1:
            if "Suggested Improvements:" in reasoning_parts[1]:
                reasoning = reasoning_parts[1].split("Suggested Improvements:", 1)[0].strip()
            else:
                reasoning = reasoning_parts[1].strip()
    
    # Extract suggested improvements
    if "Suggested Improvements:" in evaluation_response:
        suggested_improvements = evaluation_response.split("Suggested Improvements:", 1)[1].strip()
    
    # Update the state
    state["evaluation_results"] = {
        "best_option_index": best_option_index,
        "reasoning": reasoning,
        "suggested_improvements": suggested_improvements,
        "full_evaluation": evaluation_response
    }
    
    state["messages"].append(AIMessage(content=f"Evaluation Results:\nBest Option: {best_option_index + 1}\nReasoning: {reasoning}\nSuggested Improvements: {suggested_improvements}"))
    
    return state

async def correct_query(state: AgentState, generate_text_func) -> AgentState:
    """Correct the SQL query based on evaluation results"""
    logger.info(f"Correcting SQL query based on evaluation results")
    
    if not state.get("evaluation_results") or "best_option_index" not in state["evaluation_results"]:
        logger.warning("No evaluation results to use for correction")
        return state
    
    best_option_index = state["evaluation_results"]["best_option_index"]
    if best_option_index < 0 or best_option_index >= len(state["candidate_queries"]):
        logger.warning(f"Invalid best option index: {best_option_index}")
        return state
    
    best_query = state["candidate_queries"][best_option_index]
    suggested_improvements = state["evaluation_results"].get("suggested_improvements", "")
    
    if not suggested_improvements:
        # No improvements suggested, use the best query as is
        state["final_sql"] = best_query["sql"]
        state["explanation"] = best_query["explanation"]
        return state
    
    # Format schema for prompt
    schema_text = format_schema_for_prompt(state["schema"])
    
    # Create the prompt
    prompt = f"""
    You are an expert SQL query generator. Your task is to correct a SQL query based on suggested improvements.
    
    Table Schema:
    {schema_text}
    
    Question: {state['rewritten_question']}
    
    Original SQL Query:
    {best_query['sql']}
    
    Suggested Improvements:
    {suggested_improvements}
    
    Provide a corrected SQL query that addresses these suggested improvements. The response should be in the following format:
    
    SQL: <your corrected SQL query>
    Explanation: <explanation of the changes made and how the SQL query works>
    
    IMPORTANT INSTRUCTIONS:
    1. The SQL query should be valid BigQuery SQL.
    2. DO NOT include any backticks (`) or triple backticks (```) in your response.
    3. DO NOT use markdown formatting or code block markers.
    4. DO NOT prefix the SQL with "sql" or any other language indicator.
    5. Start your SQL query directly with keywords like SELECT, WITH, etc.
    """
    
    # Generate the corrected SQL query
    response_text = await generate_text_func(prompt)
    
    # Extract SQL and explanation using the dedicated function
    sql, explanation = extract_sql_from_response(response_text, best_query["sql"], best_query["explanation"])
    
    # Clean the SQL query
    sql = clean_sql_query(sql)
    
    # Log the extracted SQL for debugging
    logger.info(f"Corrected SQL query: {sql}")
    
    # Update the state
    state["final_sql"] = sql
    state["explanation"] = explanation
    state["messages"].append(AIMessage(content=f"Corrected SQL Query: {sql}\n\nExplanation: {explanation}"))
    
    return state

async def test_run_query(state: AgentState, generate_text_func) -> AgentState:
    """Test run the SQL query (dry run)"""
    logger.info(f"Test running SQL query (dry run)")
    
    if not state.get("final_sql"):
        logger.warning("No final SQL query to test run")
        return state
    
    # Format schema for prompt
    schema_text = format_schema_for_prompt(state["schema"])
    
    # Create the prompt
    prompt = f"""
    You are an expert SQL analyst. Your task is to perform a dry run of a SQL query to check for any issues.
    
    Table Schema:
    {schema_text}
    
    SQL Query:
    {state['final_sql']}
    
    Perform a dry run of this SQL query. Check for:
    1. Syntax errors
    2. Logical errors
    3. Performance issues
    4. Any other potential problems
    
    Provide your analysis in the following format:
    
    Dry Run Result: [PASS/FAIL]
    Issues: [List any issues found]
    Suggested Fixes: [Suggest fixes for any issues]
    """
    
    # Generate the dry run analysis
    dry_run_response = await generate_text_func(prompt)
    
    # Extract the dry run result
    dry_run_result = "FAIL"  # Default to fail
    issues = []
    suggested_fixes = ""
    
    if "Dry Run Result:" in dry_run_response:
        result_line = dry_run_response.split("Dry Run Result:", 1)[1].split("\n", 1)[0].strip()
        dry_run_result = result_line
    
    if "Issues:" in dry_run_response:
        issues_part = dry_run_response.split("Issues:", 1)[1]
        if "Suggested Fixes:" in issues_part:
            issues_text = issues_part.split("Suggested Fixes:", 1)[0].strip()
            issues = [issue.strip() for issue in issues_text.split("\n") if issue.strip()]
        else:
            issues_text = issues_part.strip()
            issues = [issue.strip() for issue in issues_text.split("\n") if issue.strip()]
    
    if "Suggested Fixes:" in dry_run_response:
        suggested_fixes = dry_run_response.split("Suggested Fixes:", 1)[1].strip()
    
    # Update the state
    state["dry_run_result"] = {
        "result": dry_run_result,
        "issues": issues,
        "suggested_fixes": suggested_fixes,
        "full_analysis": dry_run_response
    }
    
    state["messages"].append(AIMessage(content=f"Dry Run Result: {dry_run_result}\nIssues: {', '.join(issues) if issues else 'None'}\nSuggested Fixes: {suggested_fixes}"))
    
    return state

async def execute_query(state: AgentState) -> AgentState:
    """Execute the SQL query (placeholder for actual execution)"""
    logger.info(f"Executing SQL query")
    
    if not state.get("final_sql"):
        logger.warning("No final SQL query to execute")
        return state
    
    # In a real implementation, this would execute the query against the database
    # For now, we'll just update the state to indicate that the query was executed
    
    state["execution_result"] = {
        "status": "success",
        "message": "Query executed successfully (simulated)"
    }
    
    state["messages"].append(AIMessage(content=f"Query executed successfully (simulated)"))
    
    return state

async def finalize_result(state: AgentState) -> AgentState:
    """Finalize the result and prepare the response"""
    logger.info(f"Finalizing result")
    
    # Prepare the final response
    final_sql = state.get("final_sql", "")
    explanation = state.get("explanation", "")
    
    # Final cleaning of the SQL query
    if final_sql:
        # Log the SQL before final cleaning
        logger.info(f"Final SQL before cleaning: '{final_sql}'")
        
        # Clean the SQL query
        final_sql = clean_sql_query(final_sql)
        
        # Log the final SQL after cleaning
        logger.info(f"Final SQL after cleaning: '{final_sql}'")
        
        # Update the state with the cleaned SQL
        state["final_sql"] = final_sql
    else:
        logger.warning("No final SQL to clean in finalize_result")
    
    # Add information about the agent process
    agent_process = "\n\nAdvanced SQL Generation Process:\n"
    agent_process += f"- Rewritten the original question to better match table metadata\n"
    agent_process += f"- Analyzed metrics, joins, filters, and grouping requirements\n"
    agent_process += f"- Generated candidate SQL queries\n"
    agent_process += f"- Evaluated and selected the best query\n"
    
    if state.get("dry_run_result"):
        dry_run_result = state["dry_run_result"]["result"]
        agent_process += f"- Performed a dry run of the query (Result: {dry_run_result})\n"
    
    if state.get("execution_result"):
        execution_result = state["execution_result"]["message"]
        agent_process += f"- Executed the query (Result: {execution_result})\n"
    
    # Combine the explanation with the agent process
    combined_explanation = explanation + agent_process if explanation else agent_process
    
    # Update the state
    state["explanation"] = combined_explanation
    
    # Final check to ensure we have a valid SQL query
    if not state.get("final_sql"):
        logger.error("No final SQL query after finalization")
        state["final_sql"] = f"SELECT * FROM {state['schema'].table_name} LIMIT 10"
        state["explanation"] += "\n\nError: Failed to generate a valid SQL query."
    
    return state

# Helper function to format schema for prompts
def format_schema_for_prompt(schema: TableSchema) -> str:
    """Format table schema for prompts"""
    schema_text = f"Table: {schema.table_name}\n"
    if schema.description:
        schema_text += f"Description: {schema.description}\n"
    
    schema_text += "Columns:\n"
    for col in schema.columns:
        col_desc = f"- {col.name} ({col.data_type})"
        if col.description:
            col_desc += f": {col.description}"
        schema_text += col_desc + "\n"
    
    return schema_text

# Helper function to clean SQL queries
def clean_sql_query(sql: str) -> str:
    """
    Clean a SQL query by removing formatting artifacts like backticks, language specifiers, etc.
    
    Args:
        sql: The SQL query string to clean
        
    Returns:
        A cleaned SQL query string
    """
    if not sql:
        return sql
    
    # Log the original SQL for debugging
    logger.info(f"Original SQL before cleaning: '{sql}'")
    
    # Strip whitespace
    sql = sql.strip()
    
    # Check for empty backticks at the beginning (a common issue)
    if sql.startswith('``'):
        sql = sql.replace('``', '', 1)
        sql = sql.strip()
    
    # Check for triple backticks at the beginning and end
    if sql.startswith('```'):
        sql = sql[3:]
        sql = sql.strip()
        
        # Also remove the language specifier if present at the beginning of the first line
        if sql.lower().startswith('sql'):
            # Remove 'sql' at the beginning of the string or at the beginning of the first line
            if sql.startswith('sql') or sql.startswith('SQL'):
                sql = sql[3:].strip()
            else:
                # Handle case where 'sql' is on its own line
                lines = sql.split('\n', 1)
                if len(lines) > 1 and (lines[0].lower() == 'sql' or lines[0].strip().lower() == 'sql'):
                    sql = lines[1].strip()
    
    # Check for triple backticks at the end
    if sql.endswith('```'):
        sql = sql[:-3].strip()
    
    # Check for single backticks at the beginning and end
    if sql.startswith('`') and sql.endswith('`'):
        sql = sql[1:-1].strip()
    # Check for just backticks at the beginning
    elif sql.startswith('`'):
        sql = sql[1:].strip()
    # Check for just backticks at the end
    elif sql.endswith('`'):
        sql = sql[:-1].strip()
    
    # Remove HTML tags if present
    sql = re.sub(r'<[^>]+>', '', sql)
    
    # Remove any non-SQL characters at the beginning (common with LLM outputs)
    sql = re.sub(r'^[^a-zA-Z0-9()\s]+', '', sql)
    
    # Remove 'sql' prefix if it appears at the beginning of the string
    if sql.lower().startswith('sql'):
        # Only remove if followed by whitespace or newline to avoid removing SQL keywords
        match = re.match(r'^sql[\s\n]+(.*)', sql, re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
    
    # Final whitespace strip
    sql = sql.strip()
    
    # Log the cleaned SQL
    logger.info(f"Cleaned SQL: '{sql}'")
    
    return sql

def extract_sql_from_response(response_text: str, default_sql: str = "", default_explanation: str = "") -> tuple[str, str]:
    """
    Extract SQL and explanation from a model response.
    
    Args:
        response_text: The text response from the model
        default_sql: Default SQL to return if extraction fails
        default_explanation: Default explanation to return if extraction fails
        
    Returns:
        A tuple of (sql, explanation)
    """
    sql = default_sql
    explanation = default_explanation
    
    # Log the response for debugging
    logger.info(f"Extracting SQL from response: '{response_text[:100]}...'")
    
    # Try to extract using SQL: and Explanation: format
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
    
    # If that fails, try to extract using code blocks
    elif "```" in response_text:
        # Try to find SQL in code blocks
        code_blocks = re.findall(r"```(?:sql)?(.*?)```", response_text, re.DOTALL)
        if code_blocks:
            sql = code_blocks[0].strip()
            # Try to find explanation outside code blocks
            explanation_parts = re.split(r"```(?:sql)?.*?```", response_text, re.DOTALL)
            if len(explanation_parts) > 1:
                explanation = " ".join(part.strip() for part in explanation_parts if part.strip())
    
    # If still no SQL found, look for SELECT statement
    if not sql:
        # Look for SELECT, WITH, or other SQL keywords
        sql_keywords = ["SELECT", "WITH", "CREATE", "INSERT", "UPDATE", "DELETE", "MERGE"]
        for keyword in sql_keywords:
            if keyword in response_text.upper():
                # Extract from the keyword to the end or next paragraph
                pattern = rf"({keyword}.*?)(?:\n\n|$)"
                matches = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if matches:
                    sql = matches.group(1).strip()
                    break
    
    return sql, explanation

# Create the LangGraph agent
class LangGraphAgent:
    """LangGraph agent for SQL generation"""
    
    @staticmethod
    async def generate_sql(
        question: str,
        schema: TableSchema,
        additional_schemas: Optional[Dict[str, TableSchema]] = None,
        table_description: Optional[str] = None,
        column_descriptions: Optional[Dict[str, str]] = None,
        data_sample: Optional[List[Dict[str, Any]]] = None,
        previous_queries: Optional[List[Dict[str, Any]]] = None,
        generate_text_func=None,
        num_candidates: int = 1,
        temperatures: Optional[List[float]] = None
    ) -> NL2SQLResponse:
        """
        Generate SQL from natural language using a LangGraph agent.
        
        Args:
            question: The natural language question
            schema: The table schema
            additional_schemas: Optional additional table schemas
            table_description: Optional description of the table
            column_descriptions: Optional descriptions for columns
            data_sample: Optional sample data from the table
            previous_queries: List of previously run queries on this table
            generate_text_func: Function to generate text using LLM
            num_candidates: Number of candidate queries to generate
            temperatures: List of temperatures to use for each candidate
            
        Returns:
            NL2SQLResponse with generated SQL and explanation
        """
        logger.info(f"Generating SQL with LangGraph agent for question: {question}")
        
        if not generate_text_func:
            logger.error("No generate_text_func provided")
            return NL2SQLResponse(
                sql=f"SELECT * FROM {schema.table_name} LIMIT 10",
                explanation="Error: No text generation function provided."
            )
        
        try:
            # Initialize the state
            state: AgentState = {
                "question": question,
                "rewritten_question": question,  # Default to original question
                "schema": schema,
                "additional_schemas": additional_schemas,
                "table_description": table_description,
                "column_descriptions": column_descriptions,
                "data_sample": data_sample,
                "previous_queries": previous_queries,
                "metrics_analysis": None,
                "join_analysis": None,
                "filter_analysis": None,
                "grouping_analysis": None,
                "candidate_queries": [],
                "evaluation_results": None,
                "final_sql": None,
                "explanation": None,
                "messages": [HumanMessage(content=f"I need to convert this question to SQL: {question}")]
            }
            
            # Execute the agent steps in sequence
            # Step 1: Rewrite the question
            state = await rewrite_question(state, generate_text_func)
            
            # Create parallel streams for analysis and query generation
            analysis_tasks = [
                analyze_metrics(state, generate_text_func),
                analyze_joins(state, generate_text_func),
                analyze_filters(state, generate_text_func),
                analyze_grouping(state, generate_text_func)
            ]
            
            # Execute analysis tasks in parallel
            analysis_results = await asyncio.gather(*analysis_tasks)
            
            # Merge analysis results into the state
            for result in analysis_results:
                if result.get("metrics_analysis"):
                    state["metrics_analysis"] = result["metrics_analysis"]
                if result.get("join_analysis"):
                    state["join_analysis"] = result["join_analysis"]
                if result.get("filter_analysis"):
                    state["filter_analysis"] = result["filter_analysis"]
                if result.get("grouping_analysis"):
                    state["grouping_analysis"] = result["grouping_analysis"]
            
            # Generate candidate queries with specified parameters
            state = await generate_candidate_queries(
                state, 
                generate_text_func, 
                num_candidates=num_candidates, 
                temperatures=temperatures
            )
            
            # Evaluate queries
            state = await evaluate_queries(state, generate_text_func)
            
            # Correct query
            state = await correct_query(state, generate_text_func)
            
            # Test run query
            state = await test_run_query(state, generate_text_func)
            
            # Execute query (simulated)
            state = await execute_query(state)
            
            # Finalize result
            state = await finalize_result(state)
            
            # Get the final SQL and explanation
            final_sql = state.get("final_sql", "")
            explanation = state.get("explanation", "")
            
            # Final validation check before returning
            if final_sql:
                # Log the SQL before final validation
                logger.info(f"Final SQL before validation in generate_sql: '{final_sql}'")
                
                # One last cleaning to ensure no formatting issues
                final_sql = clean_sql_query(final_sql)
                
                # Log the final SQL after validation
                logger.info(f"Final SQL after validation in generate_sql: '{final_sql}'")
            else:
                logger.error("No SQL query generated")
                final_sql = f"SELECT * FROM {schema.table_name} LIMIT 10"
                explanation = "Error: Failed to generate a valid SQL query."
            
            # Return the response
            return NL2SQLResponse(
                sql=final_sql,
                explanation=explanation,
                rewritten_question=state.get("rewritten_question", question)
            )
        
        except Exception as e:
            logger.exception(f"Error in LangGraph agent: {str(e)}")
            return NL2SQLResponse(
                sql=f"SELECT * FROM {schema.table_name} LIMIT 10",
                explanation=f"Error in LangGraph agent: {str(e)}"
            ) 