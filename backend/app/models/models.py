from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ColumnInfo(BaseModel):
    """Information about a column in a BigQuery table"""
    name: str
    data_type: str
    description: Optional[str] = None

class TableSchema(BaseModel):
    """Schema information for a BigQuery table"""
    table_name: str
    columns: List[ColumnInfo]
    description: Optional[str] = None

class DataSample(BaseModel):
    """Sample data from a BigQuery table"""
    table_name: str
    columns: List[str]
    data: List[Dict[str, Any]]
    
class NL2SQLRequest(BaseModel):
    """Request for NL2SQL conversion"""
    question: str
    table_name: str
    additional_tables: Optional[List[str]] = Field(default_factory=list)

class NL2SQLResponse(BaseModel):
    """Response from NL2SQL conversion"""
    sql: str
    explanation: Optional[str] = None
    rewritten_question: Optional[str] = None
    
class QueryHistoryItem(BaseModel):
    """Item in the query history"""
    question: str
    table_name: str
    sql: str
    demo_phase: str  # "basic", "enhanced", or "agent"
    timestamp: datetime = Field(default_factory=datetime.now)
    context: Optional[Dict[str, Any]] = None 