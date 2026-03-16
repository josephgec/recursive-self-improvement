"""Meta-prompt templates for thinking-model genetic operators."""

META_INIT_PROMPT = """You are a prompt engineer creating a system prompt for an AI assistant.

Domain: {domain_description}

Example tasks the AI must solve:
{example_tasks}

Strategy: {strategy}

Create a high-quality system prompt with the following sections:
- identity: Who the AI is
- task_description: What it should do
- methodology: How it should approach problems
- reasoning_style: How it should think and reason
- output_format: How to present answers
- constraints: Rules and limitations
- examples: Example problem-solving patterns
- error_handling: How to handle edge cases

Return your response as a JSON object with section names as keys and content as values.

Think carefully about what makes an effective prompt for this domain.
Self-critique your draft and improve it before finalizing."""

META_MUTATION_PROMPT = """You are a prompt engineer improving a system prompt.

Current prompt sections:
{current_sections}

Performance analysis:
- Overall fitness: {fitness}
- Weakest section: {weakest_section}
- Weakness details: {weakness_details}

Your task: Improve the weakest section "{weakest_section}" to address the identified weaknesses.
Keep the improvements focused and specific.

Return a JSON object with the section name as key and improved content as value.
Only return the sections you are modifying."""

META_CROSSOVER_PROMPT = """You are a prompt engineer combining the best aspects of two system prompts.

Parent A sections (fitness: {fitness_a}):
{sections_a}

Parent B sections (fitness: {fitness_b}):
{sections_b}

Complementarity analysis:
{complementarity}

For each section, decide:
- TAKE_A: Use parent A's version
- TAKE_B: Use parent B's version
- SYNTHESIZE: Create a new version combining both

Return a JSON object with two keys:
- "decisions": dict mapping section name to "TAKE_A", "TAKE_B", or "SYNTHESIZE"
- "synthesized": dict mapping section name to new content (only for SYNTHESIZE decisions)"""
