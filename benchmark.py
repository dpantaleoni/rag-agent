"""Benchmark Ollama, llama.cpp, and MLX backends for content generation tasks."""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import Agent, SimpleRAG
from src.podcast_generator import PodcastGenerator
from src.quiz_generator import QuizGenerator
from src.config import get_config

if TYPE_CHECKING:
    import psutil
    PSUTIL_AVAILABLE = True
else:
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        psutil: Any = None
        PSUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_available_memory():
    """Log current memory status"""
    if not PSUTIL_AVAILABLE:
        return
    
    try:
        mem = psutil.virtual_memory()
        logger.info(f"Memory Status: {mem.percent:.1f}% used, {mem.available / 1e9:.1f}GB available")
    except Exception as e:
        logger.warning(f"Could not check memory: {e}")


def unload_ollama_models():
    """Unload all Ollama models from memory by stopping Ollama service
    
    This ensures that Ollama models are removed from memory before
    other backends attempt to load their models, preventing memory contention.
    """
    logger.info("=" * 80)
    logger.info("UNLOADING OLLAMA MODELS")
    logger.info("=" * 80)
    
    try:
        # Check memory before unloading
        check_available_memory()
        
        # Try to stop Ollama gracefully first
        logger.info("Attempting graceful Ollama shutdown...")
        result = subprocess.run(
            ["pkill", "-TERM", "ollama"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Wait for graceful shutdown
        time.sleep(3)
        
        # Force kill if still running
        check_result = subprocess.run(
            ["pgrep", "ollama"],
            capture_output=True,
            text=True
        )
        
        if check_result.returncode == 0:
            logger.info("Ollama still running, forcing shutdown...")
            subprocess.run(
                ["pkill", "-9", "ollama"],
                capture_output=True,
                text=True,
                timeout=5
            )
            time.sleep(2)
        
        logger.info("✓ Ollama models unloaded successfully")
        
        # Check memory after unloading
        check_available_memory()
        
    except subprocess.TimeoutExpired:
        logger.warning("Timeout while trying to stop Ollama")
    except FileNotFoundError:
        logger.warning("pkill/pgrep not found - cannot unload Ollama (Windows system?)")
    except Exception as e:
        logger.warning(f"Could not unload Ollama models: {e}")
        logger.info("Continuing anyway - next backend may fail if memory is insufficient")


def unload_llamacpp_model():
    """Unload llama.cpp model from memory
    
    llama.cpp models are loaded in-process, so we rely on Python's garbage
    collection. This function triggers cleanup and waits for memory to be freed.
    """
    logger.info("=" * 80)
    logger.info("UNLOADING LLAMA.CPP MODEL")
    logger.info("=" * 80)
    
    try:
        import gc
        
        # Check memory before unloading
        check_available_memory()
        
        logger.info("Triggering garbage collection...")
        gc.collect()
        
        # Wait for memory cleanup
        time.sleep(3)
        
        logger.info("✓ llama.cpp model unloaded successfully")
        
        # Check memory after unloading
        check_available_memory()
        
    except Exception as e:
        logger.warning(f"Could not unload llama.cpp model: {e}")
        logger.info("Continuing anyway - next backend may fail if memory is insufficient")


def get_benchmark_tasks():
    """Return the benchmark task definitions."""
    return {
        "quiz_5q_easy_tables": lambda rag, agent: QuizGenerator(rag, agent).generate(
            "document tables",
            num_questions=5,
            difficulty="easy",
        ),
        "quiz_5q_medium_pdf": lambda rag, agent: QuizGenerator(rag, agent).generate(
            "PDF parsing",
            num_questions=5,
            difficulty="medium",
        ),
        "podcast_3min_quickstart": lambda rag, agent: PodcastGenerator(rag, agent).generate_script(
            "docling quickstart",
            duration_min=3,
        ),
        "podcast_5min_advanced": lambda rag, agent: PodcastGenerator(rag, agent).generate_script(
            "advanced features",
            duration_min=5,
        ),
    }


def get_backend_models():
    """Return backend/model combinations to benchmark."""
    logger.info("Loading backend models configuration...")
    get_config()
    
    models = {
        "ollama": [
            "granite4.1:3b",
            "granite4.1:30b",
        ],
    }
    
    logger.info("Configured backends: ['ollama']")
    logger.info("  ollama: 2 model(s)")
    
    return models


def benchmark_backend_model(backend, model, tasks):
    """Benchmark one backend/model combination across all tasks.
    
    Args:
        backend: LLM backend to use ("ollama", "llamacpp", or "mlx")
        model: Model identifier
        tasks: Dictionary of task functions to run
    """
    logger.info(f"=" * 80)
    logger.info(f"Starting benchmark: backend={backend}, model={model}")
    logger.info(f"=" * 80)
    
    results = []

    logger.info("Initializing RAG system...")
    rag = SimpleRAG()
    
    # Check memory before loading model
    check_available_memory()
    
    try:
        logger.info(f"Loading agent with {backend} backend...")
        load_start = time.time()
        
        logger.info("Creating standard Agent instance...")
        agent = Agent(backend=backend, model=model)
        
        model_load_time = time.time() - load_start
        logger.info(f"✓ Agent loaded successfully in {model_load_time:.2f}s")
        check_available_memory()
        print(f"\n[{backend}:{model}] Model load time: {model_load_time:.2f}s")

        for task_name, task_func in tasks.items():
            # Run each task 3 times
            for run_num in range(1, 4):
                logger.info(f"\n{'─' * 60}")
                logger.info(f"Task: {task_name} (Run {run_num}/3)")
                logger.info(f"{'─' * 60}")
                print(f"\n[{backend}:{model}] Running: {task_name} (Run {run_num}/3)")
                
                start = time.time()
                logger.info(f"Executing task function...")

                try:
                    output = task_func(rag, agent)
                    duration = time.time() - start
                    success = True
                    error = None
                    output_size = len(str(output))
                    logger.info(f"✓ Task completed successfully")
                    logger.info(f"  Duration: {duration:.2f}s")
                    logger.info(f"  Output size: {output_size} characters")
                except Exception as exc:
                    duration = time.time() - start
                    success = False
                    output = str(exc)
                    error = str(exc)
                    output_size = 0
                    logger.error(f"✗ Task failed after {duration:.2f}s")
                    logger.error(f"  Error: {error}")

                result = {
                    "backend": backend,
                    "model": model,
                    "task": task_name,
                    "run": run_num,
                    "model_load_time": model_load_time,
                    "duration": duration,
                    "success": success,
                    "output_size": output_size,
                    "error": error,
                }
                results.append(result)

                status = "✓" if success else "✗"
                print(f"  {status} Duration: {duration:.2f}s")

        logger.info(f"\nCompleted all tasks for {backend}:{model}")
        return results
    except Exception as e:
        logger.error(f"Fatal error during benchmark: {e}", exc_info=True)
        raise
    finally:
        logger.info("Closing RAG system...")
        rag.close()


def print_summary(results):
    """Print a comparison table for benchmark results."""
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(
        f"{'Backend':<12} {'Model':<40} {'Task':<20} "
        f"{'Load (s)':<12} {'Run (s)':<12} {'OK':<6}"
    )
    print("-" * 100)

    for result in results:
        model_label = result["model"]
        if len(model_label) > 39:
            model_label = model_label[:36] + "..."

        print(
            f"{result['backend']:<12} "
            f"{model_label:<40} "
            f"{result['task']:<20} "
            f"{result['model_load_time']:<12.2f} "
            f"{result['duration']:<12.2f} "
            f"{str(result['success']):<6}"
        )


def main():
    """Run benchmark suite for configured backend/model combinations."""
    logger.info("=" * 100)
    logger.info("STARTING BENCHMARK SUITE")
    logger.info("=" * 100)
    
    logger.info("Loading benchmark tasks...")
    tasks = get_benchmark_tasks()
    logger.info(f"Loaded {len(tasks)} tasks: {list(tasks.keys())}")
    
    backend_models = get_backend_models()
    
    # Calculate total combinations
    total_combinations = sum(len(models) for models in backend_models.values())
    logger.info(f"Total benchmark combinations to run: {total_combinations}")

    print("=" * 100)
    print("PERFORMANCE BENCHMARK: OLLAMA ACROSS MULTIPLE MODELS")
    print("=" * 100)

    all_results = []
    combination_count = 0

    for backend, models in backend_models.items():
        logger.info(f"\n{'═' * 100}")
        logger.info(f"BACKEND: {backend.upper()}")
        logger.info(f"{'═' * 100}")
        
        # Unload previous backend models before switching
        if backend == "llamacpp":
            logger.info("\n" + "!" * 80)
            logger.info("SWITCHING TO LLAMA.CPP BACKEND")
            logger.info("!" * 80)
            unload_ollama_models()
            logger.info("Waiting 5 seconds for memory cleanup...")
            time.sleep(5)
            check_available_memory()
            logger.info("!" * 80 + "\n")
        elif backend == "mlx":
            logger.info("\n" + "!" * 80)
            logger.info("SWITCHING TO MLX BACKEND")
            logger.info("!" * 80)
            # Unload previous backend (could be ollama or llamacpp)
            unload_ollama_models()
            unload_llamacpp_model()
            logger.info("Waiting 5 seconds for memory cleanup...")
            time.sleep(5)
            check_available_memory()
            logger.info("!" * 80 + "\n")
        
        for model in models:
            combination_count += 1
            logger.info(f"\n[Progress: {combination_count}/{total_combinations}]")
            logger.info(f"Model: {model}")
            logger.info(f"Running combination: {backend} + {model}")
            
            try:
                results = benchmark_backend_model(backend, model, tasks)
                all_results.extend(results)
                logger.info(f"✓ Combination completed successfully")
            except Exception as e:
                logger.error(f"✗ Combination failed: {e}", exc_info=True)

    logger.info("\n" + "=" * 100)
    logger.info("SAVING RESULTS")
    logger.info("=" * 110)
    
    output_path = Path("outputs/benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)
    logger.info(f"✓ Results saved successfully ({len(all_results)} entries)")

    print_summary(all_results)
    print(f"\n✓ Benchmark complete. Results saved to: {output_path}")
    
    logger.info("\n" + "=" * 100)
    logger.info("BENCHMARK SUITE COMPLETED")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()

# Made with Bob
