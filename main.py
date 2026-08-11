"""Main CLI interface for Agentic RAG System"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src import SimpleRAG, Agent
from src.config import get_config


def main():
    """Main function for CLI interface"""
    parser = argparse.ArgumentParser(
        description="Agentic RAG system for Docling documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest documentation
  python main.py ingest data/docling_docs/
  
  # Query the RAG system
  python main.py query "How do I extract tables from documents?" --k 3
  
  # Test agent with retrieval role
  python main.py agent "document parsing" --role retrieval
  
  # Test agent with synthesis role
  python main.py agent "Explain document parsing" --role synthesis --context "Docling uses AI models"
  
  # Interactive chat with agent (includes podcast and quiz generation)
  python main.py chat
  python main.py chat --backend ollama --model gemma4:26b
  python main.py chat --use-langchain
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        'ingest', 
        help='Ingest documentation into vector database'
    )
    ingest_parser.add_argument(
        'docs_path', 
        help='Path to documentation directory'
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        'query', 
        help='Query the RAG system'
    )
    query_parser.add_argument(
        'query', 
        help='Search query'
    )
    query_parser.add_argument(
        '--k', 
        type=int, 
        default=5, 
        help='Number of results to return (default: 5)'
    )
    
    # Agent command (for testing)
    agent_parser = subparsers.add_parser(
        'agent',
        help='Test agent with different roles'
    )
    agent_parser.add_argument(
        'prompt',
        help='Prompt for the agent'
    )
    agent_parser.add_argument(
        '--role',
        choices=['retrieval', 'synthesis'],
        default='synthesis',
        help='Agent role to use (default: synthesis)'
    )
    agent_parser.add_argument(
        '--context',
        default='',
        help='Additional context for the agent'
    )
    agent_parser.add_argument(
        '--model',
        default=None,
        help='Backend-specific model identifier (Ollama model name or llama.cpp model path)'
    )
    agent_parser.add_argument(
        '--backend',
        choices=['ollama', 'llamacpp', 'mlx'],
        default='ollama',
        help='LLM backend to use (default: ollama)'
    )
    agent_parser.add_argument(
        '--show-trace',
        action='store_true',
        help='Display iteration-by-iteration reasoning, LLM responses, and tool calls'
    )
    
    # Quiz command
    quiz_parser = subparsers.add_parser(
        'quiz',
        help='Generate quiz from documentation'
    )
    quiz_parser.add_argument(
        'topic',
        help='Quiz topic'
    )
    quiz_parser.add_argument(
        '--num',
        type=int,
        default=5,
        help='Number of questions to generate (default: 5)'
    )
    quiz_parser.add_argument(
        '--difficulty',
        choices=['easy', 'medium', 'hard'],
        default='medium',
        help='Difficulty level (default: medium)'
    )
    quiz_parser.add_argument(
        '--show-answers',
        action='store_true',
        help='Display answers and explanations'
    )
    quiz_parser.add_argument(
        '--model',
        default=None,
        help='Backend-specific model identifier (Ollama model name or llama.cpp model path)'
    )
    quiz_parser.add_argument(
        '--backend',
        choices=['ollama', 'llamacpp', 'mlx'],
        default='ollama',
        help='LLM backend to use (default: ollama)'
    )
    quiz_parser.add_argument(
        '--use-langchain',
        action='store_true',
        help='Use LangChain implementation instead of direct backend'
    )
    
    # Podcast command
    podcast_parser = subparsers.add_parser(
        'podcast',
        help='Generate podcast script or audio from documentation'
    )
    podcast_parser.add_argument(
        'topic',
        help='Podcast topic'
    )
    podcast_parser.add_argument(
        '--duration',
        type=int,
        default=5,
        help='Podcast duration in minutes (default: 5)'
    )
    podcast_parser.add_argument(
        '--audio',
        action='store_true',
        help='Generate MP3 audio in addition to the podcast script'
    )
    podcast_parser.add_argument(
        '--model',
        default=None,
        help='Backend-specific model identifier (Ollama model name or llama.cpp model path)'
    )
    podcast_parser.add_argument(
        '--backend',
        choices=['ollama', 'llamacpp', 'mlx'],
        default='ollama',
        help='LLM backend to use (default: ollama)'
    )
    podcast_parser.add_argument(
        '--use-langchain',
        action='store_true',
        help='Use LangChain implementation instead of direct backend'
    )
    
    # Chat command
    chat_parser = subparsers.add_parser(
        'chat',
        help='Interactive chat with the agent'
    )
    chat_parser.add_argument(
        '--model',
        default=None,
        help='Backend-specific model identifier (Ollama model name or llama.cpp model path)'
    )
    chat_parser.add_argument(
        '--backend',
        choices=['ollama', 'llamacpp', 'mlx'],
        default='ollama',
        help='LLM backend to use (default: ollama)'
    )
    chat_parser.add_argument(
        '--show-trace',
        action='store_true',
        help='Display iteration-by-iteration reasoning, LLM responses, and tool calls'
    )
    
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Execute commands
        if args.command == 'ingest':
            print(f"Ingesting documents from: {args.docs_path}\n")
            rag = SimpleRAG()
            rag.ingest(args.docs_path)
            rag.close()
        
        elif args.command == 'query':
            print(f"Querying: {args.query}\n")
            rag = SimpleRAG()
            results = rag.retrieve(args.query, k=args.k)
            
            if not results:
                print("No results found. Make sure documents are ingested first.")
                return
            
            for i, result in enumerate(results, 1):
                print(f"\n{'='*80}")
                print(f"Result {i} (Similarity Score: {result['score']:.3f})")
                print(f"{'='*80}")
                print(f"Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"Type: {result['metadata'].get('item_type', 'Unknown')}")
                print(f"\nContent:\n{result['content'][:300]}...")
                if len(result['content']) > 300:
                    print(f"\n[Content truncated - {len(result['content'])} total characters]")
            
            rag.close()
        
        elif args.command == 'agent':
            # Initialize RAG and create ALL available tools
            rag = SimpleRAG()
            from src.tools import (
                create_rag_tools,
                create_calculator_tool,
                create_final_answer_tool,
            )
            # Combine all tools
            tools = create_rag_tools(rag)              # search_documents
            tools.append(create_calculator_tool())     # calculator
            tools.append(create_final_answer_tool())   # final_answer

            # Build query with context if provided
            query = args.prompt
            if args.context:
                query = f"{args.prompt}\n\nContext:\n{args.context}"
            
            print(f"Testing Agent\n")
            agent = Agent(model=args.model, backend=args.backend, tools=tools)
            
            response = agent.run(
                query=query,
                temperature=0.7,
                verbose=args.show_trace
            )
            
            agent_type = "Agent"
            print(f"\n{'='*80}")
            print(f"{agent_type} Response")
            print(f"{'='*80}\n")
            print(response)
            print(f"\n{'='*80}\n")
        
        elif args.command == 'quiz':
            print(f"Generating quiz on: {args.topic}")
            print(f"Questions: {args.num}, Difficulty: {args.difficulty}\n")
            
            # Initialize components
            rag = SimpleRAG()
            
            # Check if LangChain should be used
            config = get_config()
            use_langchain = args.use_langchain or config.get("llm.use_langchain", False)
            
            # Import agent classes at module level
            from src.langchain_agent import LangChainAgent
            
            # Create agent first (without quiz tool)
            if use_langchain:
                agent = LangChainAgent(model=args.model, backend=args.backend)
            else:
                agent = Agent(model=args.model, backend=args.backend)
            
            # Import tools
            from src.tools import create_rag_tools, create_quiz_tool
            
            # Create tools with the agent
            tools = create_rag_tools(rag)
            tools.append(create_quiz_tool(rag, agent))
            
            # Recreate agent with tools
            if use_langchain:
                agent = LangChainAgent(model=args.model, backend=args.backend, tools=tools)
            else:
                agent = Agent(model=args.model, backend=args.backend, tools=tools)
            
            # Build query for the agent
            query = (
                f"Generate a {args.difficulty} difficulty quiz about {args.topic} "
                f"with {args.num} questions. Use the generate_quiz tool to create it."
            )
            
            # Run agent with tools
            response = agent.run(
                query=query,
                temperature=0.7,
                verbose=True
            )
            
            print(f"\n{'='*80}")
            print(f"Agent Response")
            print(f"{'='*80}\n")
            print(response)
            print(f"\n{'='*80}\n")
            
            # Cleanup
            rag.close()
        
        elif args.command == 'podcast':
            mode_label = "podcast audio" if args.audio else "podcast script"
            print(f"Generating {mode_label} on: {args.topic}")
            print(f"Duration: {args.duration} minutes\n")
            
            # Initialize components
            rag = SimpleRAG()
            
            # Check if LangChain should be used
            config = get_config()
            use_langchain = args.use_langchain or config.get("llm.use_langchain", False)
            
            # Import agent classes at module level
            from src.langchain_agent import LangChainAgent
            
            # Create agent first (without podcast tool)
            if use_langchain:
                agent = LangChainAgent(model=args.model, backend=args.backend)
            else:
                agent = Agent(model=args.model, backend=args.backend)
            
            # Import tools
            from src.tools import create_rag_tools, create_podcast_tool
            
            # Create tools with the agent
            tools = create_rag_tools(rag)
            tools.append(create_podcast_tool(rag, agent))
            
            # Recreate agent with tools
            if use_langchain:
                agent = LangChainAgent(model=args.model, backend=args.backend, tools=tools)
            else:
                agent = Agent(model=args.model, backend=args.backend, tools=tools)
            
            # Build query for the agent
            audio_note = " and generate audio" if args.audio else ""
            query = (
                f"Generate a {args.duration}-minute podcast script about {args.topic}{audio_note}. "
                f"Use the generate_podcast tool to create it."
            )
            
            # Run agent with tools
            response = agent.run(
                query=query,
                temperature=0.7,
                verbose=True
            )
            
            print(f"\n{'='*80}")
            print(f"Agent Response")
            print(f"{'='*80}\n")
            print(response)
            print(f"\n{'='*80}\n")
            
            # Cleanup
            rag.close()
        
        elif args.command == 'chat':
            print("Starting interactive chat. Type 'exit', 'quit', or press Ctrl+C to end.\n")
            
            # Initialize components
            rag = SimpleRAG()
            
            # Create agent first (needed for podcast/quiz tools)
            agent = Agent(model=args.model, backend=args.backend)
            
            # Import all tools
            from src.tools import (
                create_rag_tools,
                create_calculator_tool,
                create_final_answer_tool,
                create_podcast_tool,
                create_quiz_tool
            )
            
            # Create ALL tools including podcast and quiz
            tools = create_rag_tools(rag)
            tools.append(create_calculator_tool())
            tools.append(create_final_answer_tool())
            tools.append(create_podcast_tool(rag, agent))
            tools.append(create_quiz_tool(rag, agent))
            
            # Recreate agent with all tools and max_iterations=5
            direct_agent = Agent(model=args.model, backend=args.backend, tools=tools)
            agent_type = "Agent"
            print(f"Initialized {agent_type} with tools: {[t.name for t in tools]}\n")
            
            # Chat loop for direct agent
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        print("\nGoodbye!")
                        break
                    
                    if not user_input:
                        continue
                    
                    print(f"\n{agent_type}: ", end="", flush=True)
                    response = direct_agent.run(query=user_input, max_iterations=5, temperature=0.2, verbose=args.show_trace)
                    print(response)
                    print()  # Extra newline for readability
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    print("Continuing chat...\n")
            
            # Cleanup
            rag.close()
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
