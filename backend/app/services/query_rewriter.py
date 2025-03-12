import logging
from typing import Dict, List, Optional, Any
from app.models.models import TableSchema

logger = logging.getLogger(__name__)

class QueryRewriter:
    """
    A service for rewriting natural language questions to better match table metadata
    and make them more precise for SQL generation.
    """
    
    @staticmethod
    async def rewrite_query(
        question: str,
        schema: TableSchema,
        additional_schemas: Optional[Dict[str, TableSchema]] = None,
        generate_text_func=None
    ) -> str:
        """
        Rewrite the natural language question to better match table metadata.
        
        Args:
            question: The original natural language question
            schema: The primary table schema
            additional_schemas: Optional additional table schemas
            generate_text_func: Function to generate text using LLM
            
        Returns:
            The rewritten question
        """
        if not generate_text_func:
            logger.warning("No generate_text_func provided, returning original question")
            return question
            
        try:
            # Format the schema for the prompt
            schema_text = f"Table: {schema.table_name}\n"
            if schema.description:
                schema_text += f"Description: {schema.description}\n"
            
            schema_text += "Columns:\n"
            for col in schema.columns:
                col_desc = f"- {col.name} ({col.data_type})"
                if col.description:
                    col_desc += f": {col.description}"
                schema_text += col_desc + "\n"
            
            # Add additional schemas if provided
            additional_schemas_text = ""
            if additional_schemas:
                for table_name, table_schema in additional_schemas.items():
                    additional_schemas_text += f"\nAdditional Table: {table_name}\n"
                    if table_schema.description:
                        additional_schemas_text += f"Description: {table_schema.description}\n"
                    
                    additional_schemas_text += "Columns:\n"
                    for col in table_schema.columns:
                        col_desc = f"- {col.name} ({col.data_type})"
                        if col.description:
                            col_desc += f": {col.description}"
                        additional_schemas_text += col_desc + "\n"
            
            # Create the prompt for query rewriting
            prompt = f"""
            You are an expert in data analysis and SQL query generation. Your task is to rewrite a natural language question to make it more precise and aligned with the database schema.
            
            The rewritten question should:
            1. Use the exact terminology from the table and column names/descriptions
            2. Be specific about what measures or calculations are needed
            3. Clearly identify the entities involved from the tables
            4. Be concise but complete
            5. Don't use vague terms like "all", "metrics" but be specific
            6. Focus on extracting meaningful insights and measures
            7. Structure the query in a way that is suitable for SQL generation
            8. Maintain the original intent and meaning
            
            Database Schema:
            {schema_text}
            {additional_schemas_text}
            
            Original Question: {question}
            
            Rewritten Question: 
            """
            
            # Generate the rewritten question
            rewritten_question = await generate_text_func(prompt)
            
            # Clean up the response
            if "Rewritten Question:" in rewritten_question:
                rewritten_question = rewritten_question.split("Rewritten Question:", 1)[1].strip()
            
            logger.info(f"Rewrote question: '{question}' -> '{rewritten_question}'")
            return rewritten_question
            
        except Exception as e:
            logger.exception(f"Error rewriting query: {str(e)}")
            return question  # Return original question on error 