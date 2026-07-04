"""Golden Q&A set for evaluating retrieval + generation quality.

Each "course" item declares the course it expects `extract_course_name` to
detect, and keywords that a correctly-retrieved chunk should contain (used as
a cheap proxy for chunk-level relevance, since we don't have hand-labeled
chunk IDs). Each "off_topic" item is a question the bot should NOT answer
with fabricated curriculum details -- it should fall back to the "couldn't
find" response instead of hallucinating.
"""

GOLDEN_DATASET = [
    {
        "id": "ds-overview",
        "type": "course",
        "question": "What topics are covered in the Data Science course?",
        "course": "Data Science",
        "keywords": ["statistics", "machine learning", "exploratory data analysis", "clustering"],
    },
    {
        "id": "ds-ml-modules",
        "type": "course",
        "question": "What machine learning algorithms are taught in the data science program?",
        "course": "Data Science",
        "keywords": ["linear regression", "logistic regression", "decision trees", "k-means", "pca"],
    },
    {
        "id": "ds-tools",
        "type": "course",
        "question": "What tools and technologies are used in the Data Science course?",
        "course": "Data Science",
        "keywords": ["pandas", "numpy", "scikit-learn", "jupyter"],
    },
    {
        "id": "da-overview",
        "type": "course",
        "question": "What is covered in the Data Analyst course?",
        "course": "Data Analyst",
        "keywords": ["sql", "power bi", "tableau", "excel"],
    },
    {
        "id": "da-bi-tools",
        "type": "course",
        "question": "Does the data analyst course teach Power BI and Tableau?",
        "course": "Data Analyst",
        "keywords": ["power bi", "tableau"],
    },
    {
        "id": "soc-overview",
        "type": "course",
        "question": "What will I learn in the SOC Analyst course?",
        "course": "SOC Analyst",
        "keywords": ["siem", "phishing", "incident response", "splunk"],
    },
    {
        "id": "soc-phishing",
        "type": "course",
        "question": "Does the SOC analyst course cover phishing analysis?",
        "course": "SOC Analyst",
        "keywords": ["phishing"],
    },
    {
        "id": "ai-overview",
        "type": "course",
        "question": "What does the Artificial Intelligence course teach?",
        "course": "Artificial Intelligence",
        "keywords": ["neural network", "natural language processing", "computer vision", "search algorithm"],
    },
    {
        "id": "ai-specialist",
        "type": "course",
        "question": "Tell me about the AI specialist course curriculum",
        "course": "Artificial Intelligence",
        "keywords": ["neural network", "deep learning"],
    },
    {
        "id": "ai-ethics",
        "type": "course",
        "question": "What does the AI course teach about ethics?",
        "course": "Artificial Intelligence",
        "keywords": ["ethics"],
    },
    {
        "id": "off-topic-beginners",
        "type": "off_topic",
        "question": "What courses do you have for beginners?",
    },
    {
        "id": "off-topic-weather",
        "type": "off_topic",
        "question": "Tell me about the weather today",
    },
    {
        "id": "off-topic-fees-professionals",
        "type": "off_topic",
        "question": "What is the fee structure for working professionals?",
    },
    {
        "id": "off-topic-nonexistent-course",
        "type": "off_topic",
        # No course by this name is offered/ingested -- unlike DevOps (which
        # IS a real offered course, confirmed by the eval harness catching an
        # earlier mislabeled version of this test case), this should reliably
        # produce a refusal rather than fabricated curriculum content.
        "question": "What is in the Blockchain Development course?",
    },
    {
        "id": "off-topic-unrelated-code",
        "type": "off_topic",
        "question": "Can you write me a Python script to sort a list?",
    },
]
