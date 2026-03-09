"""
PV-RAG Evaluation Dataset — Gold-Standard QA Pairs for Legal Temporal QA

Each entry contains:
  - query: The user question
  - query_type: HISTORICAL_POINT | LATEST | DATE_RANGE | COMPARISON | GENERAL
  - expected_year: The target year (or None)
  - gold_law: The correct Act name
  - gold_section: The correct section
  - gold_answer_keywords: Key terms that MUST appear in a correct answer
  - gold_temporal_range: (start_year, end_year) of the correct provision version
  - difficulty: easy | medium | hard
  - category: temporal_precision | version_discrimination | amendment_awareness | cross_era | general_knowledge
"""

EVALUATION_DATASET = [
    # ─────────────────────────────────────────────────────────────
    # CATEGORY 1: TEMPORAL PRECISION (correct version for exact year)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "TP-001",
        "query": "What was the penalty for not wearing a helmet while riding a motorcycle in 2010?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 2010,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": "Sec 129",
        "gold_answer_keywords": ["helmet", "motor vehicles act", "penalty", "fine"],
        "gold_temporal_range": (1988, 2019),
        "difficulty": "easy",
        "category": "temporal_precision",
    },
    {
        "id": "TP-002",
        "query": "What was the fine for drunk driving in India in 2005?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 2005,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": "Sec 185",
        "gold_answer_keywords": ["drunk", "driving", "motor vehicles act", "penalty"],
        "gold_temporal_range": (1988, 2019),
        "difficulty": "easy",
        "category": "temporal_precision",
    },
    {
        "id": "TP-003",
        "query": "What was the punishment for murder under IPC in 2020?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 2020,
        "gold_law": "Indian Penal Code, 1860",
        "gold_section": "Sec 302",
        "gold_answer_keywords": ["murder", "death", "imprisonment", "penal code"],
        "gold_temporal_range": (1860, 2023),
        "difficulty": "easy",
        "category": "temporal_precision",
    },
    {
        "id": "TP-004",
        "query": "What was the income tax exemption under section 80C in 2015?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 2015,
        "gold_law": "Income Tax Act, 1961",
        "gold_section": "Sec 80C",
        "gold_answer_keywords": ["80C", "deduction", "income tax"],
        "gold_temporal_range": (2005, 9999),
        "difficulty": "medium",
        "category": "temporal_precision",
    },
    {
        "id": "TP-005",
        "query": "What was the penalty for overspeeding in India in 2000?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 2000,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": "Sec 183",
        "gold_answer_keywords": ["speed", "motor vehicles act", "penalty"],
        "gold_temporal_range": (1988, 2019),
        "difficulty": "medium",
        "category": "temporal_precision",
    },
    {
        "id": "TP-006",
        "query": "What was the punishment for theft under IPC in 1995?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 1995,
        "gold_law": "Indian Penal Code, 1860",
        "gold_section": "Sec 379",
        "gold_answer_keywords": ["theft", "imprisonment", "penal code"],
        "gold_temporal_range": (1860, 2023),
        "difficulty": "easy",
        "category": "temporal_precision",
    },
    {
        "id": "TP-007",
        "query": "What were the rules for bail under CrPC in 2018?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 2018,
        "gold_law": "Code of Criminal Procedure, 1973",
        "gold_section": "Sec 436",
        "gold_answer_keywords": ["bail", "criminal procedure", "bailable"],
        "gold_temporal_range": (1973, 2023),
        "difficulty": "medium",
        "category": "temporal_precision",
    },

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 2: VERSION DISCRIMINATION (distinguish between versions)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "VD-001",
        "query": "What was the helmet fine before the 2019 amendment?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 2018,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": "Sec 129",
        "gold_answer_keywords": ["helmet", "motor vehicles act", "100", "before"],
        "gold_temporal_range": (1988, 2019),
        "difficulty": "hard",
        "category": "version_discrimination",
    },
    {
        "id": "VD-002",
        "query": "What is the helmet fine after the 2019 Motor Vehicles Amendment Act?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 2020,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": "Sec 129",
        "gold_answer_keywords": ["helmet", "motor vehicles act", "1000", "2019"],
        "gold_temporal_range": (2019, 9999),
        "difficulty": "hard",
        "category": "version_discrimination",
    },
    {
        "id": "VD-003",
        "query": "What was the drunk driving fine in 2015 vs current fine?",
        "query_type": "COMPARISON",
        "expected_year": None,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": "Sec 185",
        "gold_answer_keywords": ["drunk", "driving", "2015", "amendment", "increased"],
        "gold_temporal_range": None,
        "difficulty": "hard",
        "category": "version_discrimination",
    },
    {
        "id": "VD-004",
        "query": "How has the penalty for driving without license changed over the years?",
        "query_type": "DATE_RANGE",
        "expected_year": None,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": "Sec 181",
        "gold_answer_keywords": ["license", "motor vehicles act", "penalty"],
        "gold_temporal_range": None,
        "difficulty": "hard",
        "category": "version_discrimination",
    },

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 3: AMENDMENT AWARENESS (correctly identify amendments)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "AA-001",
        "query": "When was the Motor Vehicles Act last amended?",
        "query_type": "GENERAL",
        "expected_year": None,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": None,
        "gold_answer_keywords": ["2019", "amendment", "motor vehicles"],
        "gold_temporal_range": None,
        "difficulty": "medium",
        "category": "amendment_awareness",
    },
    {
        "id": "AA-002",
        "query": "What changes were made to IPC Sec 498A over time?",
        "query_type": "GENERAL",
        "expected_year": None,
        "gold_law": "Indian Penal Code, 1860",
        "gold_section": "Sec 498A",
        "gold_answer_keywords": ["cruelty", "husband", "498A", "dowry"],
        "gold_temporal_range": None,
        "difficulty": "hard",
        "category": "amendment_awareness",
    },
    {
        "id": "AA-003",
        "query": "Has section 377 of IPC been changed?",
        "query_type": "GENERAL",
        "expected_year": None,
        "gold_law": "Indian Penal Code, 1860",
        "gold_section": "Sec 377",
        "gold_answer_keywords": ["377", "unnatural", "penal code"],
        "gold_temporal_range": None,
        "difficulty": "medium",
        "category": "amendment_awareness",
    },

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 4: CROSS-ERA (law replaced entirely — IPC→BNS)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "CE-001",
        "query": "What is the current law for murder in India as of 2025?",
        "query_type": "LATEST",
        "expected_year": 2025,
        "gold_law": "Bharatiya Nyaya Sanhita, 2023",
        "gold_section": None,
        "gold_answer_keywords": ["bharatiya nyaya sanhita", "BNS", "murder"],
        "gold_temporal_range": (2024, 9999),
        "difficulty": "hard",
        "category": "cross_era",
    },
    {
        "id": "CE-002",
        "query": "What replaced the Indian Penal Code?",
        "query_type": "GENERAL",
        "expected_year": None,
        "gold_law": "Bharatiya Nyaya Sanhita, 2023",
        "gold_section": None,
        "gold_answer_keywords": ["bharatiya nyaya sanhita", "2023", "replaced"],
        "gold_temporal_range": None,
        "difficulty": "medium",
        "category": "cross_era",
    },
    {
        "id": "CE-003",
        "query": "If someone committed theft in 2023, which law applies — IPC or BNS?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 2023,
        "gold_law": "Indian Penal Code, 1860",
        "gold_section": "Sec 379",
        "gold_answer_keywords": ["Indian Penal Code", "theft", "2023", "IPC"],
        "gold_temporal_range": (1860, 2023),
        "difficulty": "hard",
        "category": "cross_era",
    },

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 5: GENERAL KNOWLEDGE (standard legal QA)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "GK-001",
        "query": "What is the right to equality under Indian constitution?",
        "query_type": "GENERAL",
        "expected_year": None,
        "gold_law": "Constitution of India",
        "gold_section": "Article 14",
        "gold_answer_keywords": ["equality", "law", "constitution", "article 14"],
        "gold_temporal_range": None,
        "difficulty": "easy",
        "category": "general_knowledge",
    },
    {
        "id": "GK-002",
        "query": "What is the Indian Evidence Act about?",
        "query_type": "GENERAL",
        "expected_year": None,
        "gold_law": "Indian Evidence Act, 1872",
        "gold_section": None,
        "gold_answer_keywords": ["evidence", "proof", "witness"],
        "gold_temporal_range": None,
        "difficulty": "easy",
        "category": "general_knowledge",
    },
    {
        "id": "GK-003",
        "query": "What is the punishment for forgery under Indian law?",
        "query_type": "GENERAL",
        "expected_year": None,
        "gold_law": "Indian Penal Code, 1860",
        "gold_section": "Sec 463",
        "gold_answer_keywords": ["forgery", "imprisonment", "penal code"],
        "gold_temporal_range": None,
        "difficulty": "easy",
        "category": "general_knowledge",
    },
    {
        "id": "GK-004",
        "query": "What is section 144 CrPC?",
        "query_type": "GENERAL",
        "expected_year": None,
        "gold_law": "Code of Criminal Procedure, 1973",
        "gold_section": "Sec 144",
        "gold_answer_keywords": ["144", "order", "nuisance", "magistrate"],
        "gold_temporal_range": None,
        "difficulty": "easy",
        "category": "general_knowledge",
    },

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 6: MULTI-YEAR COMPARISON (explicit two-year)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "MC-001",
        "query": "Compare the penalty for not wearing helmet in 2010 vs 2022",
        "query_type": "COMPARISON",
        "expected_year": None,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": "Sec 129",
        "gold_answer_keywords": ["2010", "2022", "helmet", "increased", "motor vehicles"],
        "gold_temporal_range": None,
        "difficulty": "hard",
        "category": "comparison",
    },
    {
        "id": "MC-002",
        "query": "Compare drunk driving penalties between 2008 and 2020",
        "query_type": "COMPARISON",
        "expected_year": None,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": "Sec 185",
        "gold_answer_keywords": ["2008", "2020", "drunk", "driving", "motor vehicles"],
        "gold_temporal_range": None,
        "difficulty": "hard",
        "category": "comparison",
    },

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 7: TEMPORAL EDGE CASES (boundary years, null end_year)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "EC-001",
        "query": "What was the helmet penalty in 2019?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 2019,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": "Sec 129",
        "gold_answer_keywords": ["helmet", "motor vehicles act", "2019"],
        "gold_temporal_range": (2019, 9999),
        "difficulty": "hard",
        "category": "temporal_edge_case",
    },
    {
        "id": "EC-002",
        "query": "What was the law on dowry prohibition in 1960?",
        "query_type": "HISTORICAL_POINT",
        "expected_year": 1960,
        "gold_law": "Dowry Prohibition Act, 1961",
        "gold_section": None,
        "gold_answer_keywords": ["dowry"],
        "gold_temporal_range": None,
        "difficulty": "hard",
        "category": "temporal_edge_case",
    },

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 8: DATE RANGE QUERIES
    # ─────────────────────────────────────────────────────────────
    {
        "id": "DR-001",
        "query": "How did traffic fines change between 2015 and 2022?",
        "query_type": "DATE_RANGE",
        "expected_year": None,
        "gold_law": "Motor Vehicles Act, 1988",
        "gold_section": None,
        "gold_answer_keywords": ["motor vehicles", "2019", "amendment", "fine"],
        "gold_temporal_range": None,
        "difficulty": "medium",
        "category": "date_range",
    },
    {
        "id": "DR-002",
        "query": "What changes were made to cybercrime laws between 2000 and 2020?",
        "query_type": "DATE_RANGE",
        "expected_year": None,
        "gold_law": "Information Technology Act, 2000",
        "gold_section": None,
        "gold_answer_keywords": ["information technology", "cyber", "IT act"],
        "gold_temporal_range": None,
        "difficulty": "medium",
        "category": "date_range",
    },
]


def get_dataset_by_category(category: str):
    """Filter evaluation dataset by category."""
    return [q for q in EVALUATION_DATASET if q["category"] == category]


def get_dataset_by_difficulty(difficulty: str):
    """Filter evaluation dataset by difficulty."""
    return [q for q in EVALUATION_DATASET if q["difficulty"] == difficulty]


def get_dataset_by_query_type(query_type: str):
    """Filter evaluation dataset by query type."""
    return [q for q in EVALUATION_DATASET if q["query_type"] == query_type]
