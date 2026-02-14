"""
Database models for legal rules and verification logs
"""
from sqlalchemy import Column, String, Integer, Text, Date, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.session import Base


class LegalRule(Base):
    """Legal rules with temporal versioning"""
    __tablename__ = "legal_rules"
    
    # Primary identifiers
    id = Column(Integer, primary_key=True, index=True)
    rule_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # Legal metadata
    act_title = Column(String(500), nullable=False, index=True)
    section_id = Column(String(100), index=True)
    rule_topic = Column(String(255), index=True)
    rule_text = Column(Text, nullable=False)
    
    # Temporal tracking (CRITICAL for PV-RAG)
    start_date = Column(Date, nullable=False, index=True)
    end_date = Column(Date, nullable=True, index=True)
    
    # Version management
    version_tag = Column(String(50))
    previous_version_id = Column(String(255), ForeignKey('legal_rules.rule_id'), nullable=True)
    
    # Amendment tracking
    amendment_reference = Column(String(500))
    source = Column(String(500))
    source_url = Column(String(1000))
    
    # Status
    status = Column(String(50), default='active')  # active, superseded, repealed
    
    # Additional metadata
    metadata = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    previous_version = relationship("LegalRule", remote_side=[rule_id], foreign_keys=[previous_version_id])
    
    def __repr__(self):
        return f"<LegalRule(rule_id='{self.rule_id}', act='{self.act_title[:50]}', period='{self.start_date}to{self.end_date}')>"


class WebVerificationLog(Base):
    """Logs for web agent verification checks"""
    __tablename__ = "web_verification_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Query information
    query_text = Column(Text, nullable=False)
    rule_id = Column(String(255), ForeignKey('legal_rules.rule_id'), nullable=True)
    
    # Verification results
    verification_date = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    verdict = Column(String(50))  # NO_UPDATE, UPDATE_FOUND, UNCLEAR
    confidence_score = Column(Float)
    
    # Web search details
    search_query = Column(Text)
    sources_checked = Column(JSON)  # List of URLs
    agent_reasoning = Column(Text)
    
    # Results
    found_updates = Column(Boolean, default=False)
    detected_changes = Column(JSON)
    
    # Performance metrics
    execution_time_seconds = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<WebVerificationLog(rule_id='{self.rule_id}', verdict='{self.verdict}', date='{self.verification_date}')>"


class QueryLog(Base):
    """User query logs for analytics"""
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Query details
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))  # HISTORICAL_POINT, LATEST, DATE_RANGE, COMPARISON
    temporal_entities = Column(JSON)  # Extracted dates/years
    
    # Response details
    response_time_seconds = Column(Float)
    confidence_score = Column(Float)
    verification_method = Column(String(50))  # dataset, web_agent, hybrid
    
    # Results
    rules_retrieved = Column(Integer)
    timeline_generated = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<QueryLog(query='{self.query_text[:50]}...', type='{self.query_type}')>"
