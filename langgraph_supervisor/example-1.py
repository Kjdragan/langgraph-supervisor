import os
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from rich.markdown import Markdown
from rich.logging import RichHandler
from langchain_core.runnables.graph import Graph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Set up logging
def setup_logging(log_dir="_logs"):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"supervisor_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RichHandler(rich_tracebacks=True, markup=True),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger("LangGraphSupervisor")

# Initialize logger
logger = setup_logging()

# Initialize rich console
console = Console()

# Load environment variables from .env file
logger.info("Loading environment variables...")
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import TavilySearchAPIRetriever

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

logger.info("Initializing LLM model...")
model = ChatOpenAI(model="gpt-4o")

# Create specialized agents
def add(a: float, b: float) -> float:
    """Add two numbers."""
    result = a + b
    logger.info(f"Math operation: {a} + {b} = {result}")
    console.print(f"[cyan]Math operation:[/cyan] {a} + {b} = {result}")
    return result

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    result = a * b
    logger.info(f"Math operation: {a} × {b} = {result}")
    console.print(f"[cyan]Math operation:[/cyan] {a} × {b} = {result}")
    return result

def web_search(query: str) -> str:
    """Search the web for information using Tavily."""
    logger.info(f"Performing web search for: {query}")
    console.print(Panel(f"[yellow]Searching web for:[/yellow] {query}", title="Web Search"))
    
    retriever = TavilySearchAPIRetriever(k=3)  # Get top 3 results
    docs = retriever.invoke(query)
    
    # Create a table for search results
    table = Table(title="Search Results", show_header=True, header_style="bold magenta")
    table.add_column("Source", style="dim")
    table.add_column("Content")
    
    # Add results to table and log
    results_text = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown source')
        content = doc.page_content.replace('\n', ' ').strip()
        table.add_row(source, content)
        results_text.append(f"Source: {source}\nContent: {content}")
    
    logger.info("Search results:\n" + "\n".join(results_text))
    console.print(table)
    
    # Combine the content from all documents
    results = "\n\n".join(doc.page_content for doc in docs)
    return results

logger.info("Creating specialized agents...")
math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)
logger.info("Created math expert agent with add and multiply tools")

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)
logger.info("Created research expert agent with web search tool")

logger.info("Creating supervisor workflow...")
# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    ),
    output_mode="full_history",  # Get full conversation history
    add_handoff_back_messages=True,  # Add messages when control returns to supervisor
)

# Compile workflow
logger.info("Compiling workflow...")
app = workflow.compile()

# Generate and display the workflow graph
logger.info("Generating workflow visualization...")
try:
    # Get the graph object and generate mermaid diagram
    graph = app.get_graph()
    
    # Save mermaid diagram as PNG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join("_output", f"workflow_graph_{timestamp}.png")
    graph.draw_mermaid_png(output_file_path=png_path)
    logger.info(f"Graph saved to: {png_path}")
    console.print(f"[green]Graph saved to:[/green] {png_path}")
    
    # Display mermaid diagram in console
    mermaid_output = graph.draw_mermaid()
    graph_markdown = f"""
```mermaid
{mermaid_output}
```
"""
    console.print(Markdown(graph_markdown))
except Exception as e:
    error_msg = f"Error generating mermaid graph: {str(e)} ({e.__class__.__name__})"
    logger.error(error_msg)
    console.print(f"[red]{error_msg}[/red]")

# Example query
query = "what's the combined headcount of the FAANG companies in 2024?"
logger.info(f"Processing query: {query}")
console.print(Panel(f"[bold yellow]User Query:[/bold yellow] {query}", title="Input"))

# Create message with the query
messages = [HumanMessage(content=query)]

# Run the workflow
logger.info("Running workflow...")
result = app.invoke({
    "messages": messages
})

# Parse and format the result
def format_conversation(messages):
    """Format conversation messages for display and logging."""
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append(("Human", msg.content))
            logger.info(f"Human message: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_messages.append(("AI", msg.content))
            logger.info(f"AI message: {msg.content}")
    return formatted_messages

# Create a table for the conversation
conversation_table = Table(title="Conversation Flow", show_header=True, header_style="bold magenta")
conversation_table.add_column("Role", style="cyan")
conversation_table.add_column("Message", style="white")

# Add messages to the table
for role, content in format_conversation(result["messages"]):
    conversation_table.add_row(role, content)

# Print the conversation table
logger.info("Displaying conversation flow...")
console.print("\n[bold blue]Conversation Flow:[/bold blue]")
console.print(conversation_table)

# Extract the final answer
final_message = next((msg for msg in reversed(result["messages"]) if isinstance(msg, AIMessage)), None)
if final_message:
    # Format the final answer in a panel
    logger.info(f"Final answer: {final_message.content}")
    console.print("\n[bold blue]Final Answer:[/bold blue]")
    console.print(Panel(
        final_message.content,
        title="Result",
        border_style="green",
        padding=(1, 2)
    ))
else:
    error_msg = "No final answer found in the conversation"
    logger.error(error_msg)
    console.print(f"[red]{error_msg}[/red]")