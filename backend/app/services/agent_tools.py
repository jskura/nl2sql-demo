import re
import logging
from typing import List, Dict, Any, Optional
from collections import Counter

from app.models.models import QueryHistoryItem

# Configure logging
logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Tool for analyzing query patterns and providing recommendations"""
    
    @staticmethod
    def extract_sql_patterns(sql: str) -> Dict[str, Any]:
        """Extract patterns from SQL query"""
        patterns = {
            "has_join": bool(re.search(r'\bjoin\b', sql, re.IGNORECASE)),
            "has_group_by": bool(re.search(r'\bgroup\s+by\b', sql, re.IGNORECASE)),
            "has_order_by": bool(re.search(r'\border\s+by\b', sql, re.IGNORECASE)),
            "has_limit": bool(re.search(r'\blimit\b', sql, re.IGNORECASE)),
            "has_where": bool(re.search(r'\bwhere\b', sql, re.IGNORECASE)),
            "has_having": bool(re.search(r'\bhaving\b', sql, re.IGNORECASE)),
            "has_subquery": bool(re.search(r'\(\s*select\b', sql, re.IGNORECASE)),
            "has_with": bool(re.search(r'\bwith\b', sql, re.IGNORECASE)),
            "has_union": bool(re.search(r'\bunion\b', sql, re.IGNORECASE)),
            "has_window_function": bool(re.search(r'\bover\s*\(', sql, re.IGNORECASE)),
            "aggregation_functions": re.findall(r'\b(count|sum|avg|min|max)\s*\(', sql, re.IGNORECASE),
            "tables": re.findall(r'\bfrom\s+([a-zA-Z0-9_]+)', sql, re.IGNORECASE) + 
                      re.findall(r'\bjoin\s+([a-zA-Z0-9_]+)', sql, re.IGNORECASE),
            "columns": re.findall(r'\bselect\s+(.*?)\s+from\b', sql, re.IGNORECASE | re.DOTALL)
        }
        
        # Extract complexity score (simple heuristic)
        complexity = 0
        for key, value in patterns.items():
            if key.startswith("has_") and value:
                complexity += 1
        if patterns["aggregation_functions"]:
            complexity += len(patterns["aggregation_functions"])
        if patterns["tables"]:
            complexity += len(patterns["tables"]) - 1  # -1 because at least one table is expected
            
        patterns["complexity_score"] = complexity
        
        return patterns

    @staticmethod
    def analyze_query_history(history: List[QueryHistoryItem], question: str) -> Dict[str, Any]:
        """Analyze query history to find patterns and make recommendations"""
        if not history:
            return {
                "patterns": {},
                "recommendations": ["No query history available for analysis."],
                "similar_questions": []
            }
        
        # Extract patterns from all queries
        all_patterns = []
        for item in history:
            patterns = QueryAnalyzer.extract_sql_patterns(item.sql)
            patterns["question"] = item.question
            patterns["sql"] = item.sql
            patterns["phase"] = item.phase
            all_patterns.append(patterns)
        
        # Analyze common patterns
        common_patterns = {
            "joins": Counter(),
            "aggregations": Counter(),
            "tables": Counter(),
            "complexity": {
                "min": min(p["complexity_score"] for p in all_patterns),
                "max": max(p["complexity_score"] for p in all_patterns),
                "avg": sum(p["complexity_score"] for p in all_patterns) / len(all_patterns)
            }
        }
        
        for pattern in all_patterns:
            if pattern["has_join"]:
                common_patterns["joins"]["join"] += 1
            for agg in pattern["aggregation_functions"]:
                common_patterns["aggregations"][agg.lower()] += 1
            for table in pattern["tables"]:
                common_patterns["tables"][table.lower()] += 1
        
        # Find similar questions (simple word overlap for demo)
        question_words = set(question.lower().split())
        similar_questions = []
        
        for pattern in all_patterns:
            history_question = pattern["question"]
            history_words = set(history_question.lower().split())
            overlap = len(question_words.intersection(history_words))
            if overlap > 1:  # At least 2 words in common
                similar_questions.append({
                    "question": history_question,
                    "sql": pattern["sql"],
                    "overlap_score": overlap,
                    "phase": pattern["phase"]
                })
        
        # Sort by overlap score
        similar_questions.sort(key=lambda x: x["overlap_score"], reverse=True)
        
        # Generate recommendations
        recommendations = []
        
        if common_patterns["joins"]["join"] > 0:
            recommendations.append("Consider using JOIN operations as they appear in previous queries.")
        
        most_common_agg = common_patterns["aggregations"].most_common(1)
        if most_common_agg:
            recommendations.append(f"The {most_common_agg[0][0].upper()} aggregation function is commonly used in previous queries.")
        
        most_common_tables = common_patterns["tables"].most_common(2)
        if len(most_common_tables) >= 2:
            recommendations.append(f"Tables {most_common_tables[0][0]} and {most_common_tables[1][0]} are frequently used together.")
        
        if similar_questions:
            recommendations.append(f"Found {len(similar_questions)} similar questions in history that might provide useful patterns.")
        
        return {
            "patterns": common_patterns,
            "recommendations": recommendations,
            "similar_questions": similar_questions[:3]  # Return top 3 similar questions
        }

class DataProfiler:
    """Tool for profiling data samples to provide insights"""
    
    @staticmethod
    def profile_data_sample(data_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Profile a data sample to extract insights"""
        if not data_sample or "rows" not in data_sample or not data_sample["rows"]:
            return {
                "insights": ["No data sample available for profiling."],
                "column_stats": {}
            }
        
        rows = data_sample["rows"]
        columns = data_sample["columns"]
        
        # Initialize column statistics
        column_stats = {}
        for col in columns:
            column_stats[col] = {
                "type": "unknown",
                "non_null_count": 0,
                "null_count": 0,
                "unique_values": set(),
                "numeric_stats": {
                    "min": None,
                    "max": None,
                    "sum": 0,
                    "avg": None
                },
                "string_stats": {
                    "min_length": None,
                    "max_length": None,
                    "avg_length": None
                },
                "date_format": None
            }
        
        # Analyze each column
        for row in rows:
            for col in columns:
                value = row.get(col)
                
                # Count nulls and non-nulls
                if value is None:
                    column_stats[col]["null_count"] += 1
                else:
                    column_stats[col]["non_null_count"] += 1
                    column_stats[col]["unique_values"].add(str(value))
                    
                    # Infer type and collect stats
                    if isinstance(value, (int, float)):
                        column_stats[col]["type"] = "numeric"
                        
                        # Update numeric stats
                        if column_stats[col]["numeric_stats"]["min"] is None or value < column_stats[col]["numeric_stats"]["min"]:
                            column_stats[col]["numeric_stats"]["min"] = value
                            
                        if column_stats[col]["numeric_stats"]["max"] is None or value > column_stats[col]["numeric_stats"]["max"]:
                            column_stats[col]["numeric_stats"]["max"] = value
                            
                        column_stats[col]["numeric_stats"]["sum"] += value
                    
                    elif isinstance(value, str):
                        column_stats[col]["type"] = "string"
                        
                        # Update string stats
                        length = len(value)
                        
                        if column_stats[col]["string_stats"]["min_length"] is None or length < column_stats[col]["string_stats"]["min_length"]:
                            column_stats[col]["string_stats"]["min_length"] = length
                            
                        if column_stats[col]["string_stats"]["max_length"] is None or length > column_stats[col]["string_stats"]["max_length"]:
                            column_stats[col]["string_stats"]["max_length"] = length
                        
                        # Check if it might be a date
                        if re.match(r'\d{4}-\d{2}-\d{2}', value):
                            column_stats[col]["date_format"] = "YYYY-MM-DD"
        
        # Calculate averages
        for col in columns:
            if column_stats[col]["type"] == "numeric" and column_stats[col]["non_null_count"] > 0:
                column_stats[col]["numeric_stats"]["avg"] = column_stats[col]["numeric_stats"]["sum"] / column_stats[col]["non_null_count"]
            
            if column_stats[col]["type"] == "string" and column_stats[col]["non_null_count"] > 0 and column_stats[col]["string_stats"]["min_length"] is not None:
                total_length = column_stats[col]["string_stats"]["min_length"] + column_stats[col]["string_stats"]["max_length"]
                column_stats[col]["string_stats"]["avg_length"] = total_length / 2  # Approximation for demo
        
        # Convert sets to lists for JSON serialization
        for col in columns:
            column_stats[col]["unique_values"] = list(column_stats[col]["unique_values"])
            column_stats[col]["unique_count"] = len(column_stats[col]["unique_values"])
            if len(column_stats[col]["unique_values"]) > 10:
                column_stats[col]["unique_values"] = column_stats[col]["unique_values"][:10]  # Limit to 10 examples
        
        # Generate insights
        insights = []
        
        # Check for columns that might be primary keys
        for col in columns:
            if column_stats[col]["unique_count"] == len(rows) and column_stats[col]["null_count"] == 0:
                insights.append(f"Column '{col}' might be a primary key (all values unique and non-null).")
        
        # Check for columns that might be foreign keys
        for col in columns:
            if col.endswith("_id") and column_stats[col]["type"] == "numeric":
                insights.append(f"Column '{col}' might be a foreign key (name ends with _id and contains numeric values).")
        
        # Check for date columns
        date_columns = [col for col in columns if column_stats[col]["date_format"] is not None]
        if date_columns:
            insights.append(f"Columns {', '.join(date_columns)} contain date values and can be used for time-based analysis.")
        
        # Check for high cardinality columns
        for col in columns:
            if column_stats[col]["unique_count"] > 0.8 * len(rows) and len(rows) > 5:
                insights.append(f"Column '{col}' has high cardinality ({column_stats[col]['unique_count']} unique values in {len(rows)} rows).")
        
        return {
            "insights": insights,
            "column_stats": column_stats
        }

def get_agent_tools():
    """Return all available agent tools"""
    return {
        "query_analyzer": QueryAnalyzer(),
        "data_profiler": DataProfiler()
    } 