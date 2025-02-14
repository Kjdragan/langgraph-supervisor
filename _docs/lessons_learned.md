# Lessons Learned

## LangGraph Supervisor Framework

### Architecture
- LangGraph Supervisor is designed for hierarchical multi-agent systems
- A supervisor agent orchestrates specialized agents through tool-based handoffs
- Each specialized agent can have its own set of tools and capabilities
- The supervisor's prompt should clearly define when to use each agent

### Message Handling
- Use LangChain's message types (`HumanMessage`, `AIMessage`) for structured communication
- Messages are passed through the workflow in a standardized format
- The result from the workflow contains a list of messages showing the full conversation flow
- Extract final answers by finding the last `AIMessage` in the result

### Graph Visualization
- Compiled LangGraph workflows can be visualized using Mermaid diagrams
- Access the graph structure using `app.get_graph()` after compilation
- Use `graph.draw_mermaid()` for console/markdown display
- Use `graph.draw_mermaid_png()` to save as PNG file
- Always include timestamps in output filenames to prevent overwrites
- Store generated graphs in a dedicated output directory

### Best Practices
1. **Agent Design**
   - Give each agent a clear, focused responsibility
   - Use descriptive names for agents (e.g., `math_expert`, `research_expert`)
   - Include role-specific instructions in agent prompts
   - Ensure agents don't overlap in responsibilities

2. **Tool Integration**
   - Tools should be atomic and focused
   - Document tools with clear docstrings
   - Return structured data when possible
   - Add logging to tools for better visibility

3. **Output Formatting**
   - Use rich library for enhanced console output
   - Create tables for structured data (search results, conversations)
   - Use panels for important information
   - Color-code different types of output
   - Include progress indicators for long-running operations

4. **Error Handling**
   - Wrap graph generation in try/except blocks
   - Provide detailed error messages
   - Include error class names for better debugging
   - Log errors appropriately

## Environment Management

### Model Configuration
- Using GPT-4o as the default model for consistency
- Model configuration is done through the `ChatOpenAI` class
- Same model instance can be shared across agents

### Environment Variables
- Use `python-dotenv` for environment variable management
- Keep API keys in `.env` file
- Include `.env.example` for documentation
- Add `.env` to `.gitignore`

## Project Structure
- Keep specialized agents in separate modules
- Use clear naming conventions
- Document code organization patterns
- Include example files to demonstrate usage
- Maintain separate directories for outputs and documentation

## Dependencies
- Use `uv` as package manager
- Add new packages with `uv add`
- Keep dependencies updated in `pyproject.toml`
- Document external API requirements

## Working with LangGraph Supervisor

### Workflow Creation
1. Define specialized agents with specific tools
2. Create a supervisor with clear delegation rules
3. Compile the workflow before use
4. Visualize the workflow for verification

### Agent Communication
- Agents communicate through tool-based handoffs
- Supervisor manages all delegation decisions
- Messages flow through a standardized format
- Full conversation history is preserved

### Output Management
1. Save artifacts (graphs, results) with timestamps
2. Use structured formats for better readability
3. Maintain conversation context
4. Format final results clearly

## To Investigate
- Error handling in agent communications
- State management between agent handoffs
- Performance optimization for web searches
- Testing strategies for multi-agent systems
- Advanced graph visualization options
- Custom message handlers and formatters
