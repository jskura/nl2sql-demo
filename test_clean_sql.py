import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def test_clean_sql():
    """Test the clean_sql_query function with various problematic SQL queries"""
    test_cases = [
        "``SELECT * FROM table",
        "```sql\nSELECT * FROM table\n```",
        "`SELECT * FROM table`",
        "```\nSELECT * FROM table\n```",
        "SELECT * FROM table",
        "<div>SELECT * FROM table</div>",
        "sql\nSELECT * FROM table",
        "```sql SELECT * FROM table ```",
        "  SELECT * FROM table  ",
        "```\nsql\nSELECT * FROM table\n```",
        "sql SELECT * FROM table",
        "SQL SELECT * FROM table",
        "```\nsql\nSELECT * FROM table WHERE sql_column = 'value'\n```",
        "``",
        "",
        None
    ]
    
    print("Testing clean_sql_query function:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases):
        if test_case is None:
            print(f"\nTest case {i+1}: None")
            result = clean_sql_query(test_case)
            print(f"Result: {result}")
        else:
            print(f"\nTest case {i+1}: '{test_case}'")
            result = clean_sql_query(test_case)
            print(f"Result: '{result}'")
    
    print("-" * 50)

if __name__ == "__main__":
    test_clean_sql() 