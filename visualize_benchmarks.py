#!/usr/bin/env python3
"""
Benchmark Results Visualization Script

This script loads benchmark results from multiple JSON files and creates
comprehensive visualizations to analyze performance across different backends,
models, agent types, and tasks.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Output directory
OUTPUT_DIR = Path("outputs/data-visuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_benchmark_data():
    """Load all benchmark result files and combine them."""
    files = [
        "outputs/benchmark_results.json",
        "outputs/benchmark_results0.json",
        "outputs/benchmark_results1.json",
        "outputs/benchmark_results2.json"
    ]
    
    all_data = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Add source file identifier
                for item in data:
                    item['source_file'] = Path(file_path).stem
                all_data.extend(data)
            print(f"✓ Loaded {len(data)} records from {file_path}")
        except FileNotFoundError:
            print(f"⚠ Warning: {file_path} not found, skipping...")
        except json.JSONDecodeError as e:
            print(f"⚠ Warning: Error parsing {file_path}: {e}")
    
    df = pd.DataFrame(all_data)
    print(f"\n✓ Total records loaded: {len(df)}")
    return df

def clean_model_names(df):
    """Simplify model names for better visualization."""
    df['model_short'] = df['model'].apply(lambda x: 
        'granite-3b' if '3b' in str(x).lower() else
        'granite-30b' if '30b' in str(x).lower() else
        str(x).split('/')[-1][:20]
    )
    return df

def plot_success_rate_by_backend(df):
    """Create bar chart showing success rates by backend."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    success_by_backend = df.groupby('backend')['success'].agg(['sum', 'count'])
    success_by_backend['rate'] = (success_by_backend['sum'] / success_by_backend['count'] * 100)
    
    colors = sns.color_palette("husl", len(success_by_backend))
    bars = ax.bar(success_by_backend.index, success_by_backend['rate'], color=colors)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Backend', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate by Backend', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "success_rate_by_backend.png", dpi=300, bbox_inches='tight')
    print("✓ Created: success_rate_by_backend.png")
    plt.close()

def plot_duration_comparison(df):
    """Create grouped bar chart comparing task durations."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Filter successful tasks only
    df_success = df[df['success'] == True].copy()
    
    # Group by backend and model
    df_success['backend_model'] = df_success['backend'] + '\n' + df_success['model_short']
    
    # Calculate average duration by backend_model and task type
    task_groups = df_success.groupby(['backend_model', 'task'])['duration'].mean().reset_index()
    
    # Pivot for grouped bar chart
    pivot_data = task_groups.pivot(index='backend_model', columns='task', values='duration')
    
    # Plot
    pivot_data.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_xlabel('Backend & Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Task Duration Comparison (Successful Tasks Only)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Task Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "duration_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Created: duration_comparison.png")
    plt.close()

def plot_model_size_performance(df):
    """Create comparison of 3b vs 30b models (Ollama backend only)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter for Ollama backend only (include both successful and failed tasks)
    df_ollama = df[df['backend'] == 'ollama'].copy()
    
    # Success rate comparison for Ollama only
    success_by_model = df_ollama.groupby('model_short')['success'].agg(['sum', 'count'])
    success_by_model['rate'] = (success_by_model['sum'] / success_by_model['count'] * 100)
    success_by_model = success_by_model[success_by_model.index.isin(['granite-3b', 'granite-30b'])]
    
    bars = ax.bar(success_by_model.index, success_by_model['rate'],
                   color=['skyblue', 'coral'])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Size vs Success Rate (Ollama Backend Only)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_size_performance.png", dpi=300, bbox_inches='tight')
    print("✓ Created: model_size_performance.png")
    plt.close()

def plot_agent_type_comparison(df):
    """Compare regular vs langchain agent performance."""
    if 'agent_type' not in df.columns:
        print("⚠ Skipping agent comparison - no agent_type data")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter data with agent_type
    df_agents = df[df['agent_type'].notna()].copy()
    
    if len(df_agents) == 0:
        print("⚠ Skipping agent comparison - no agent_type data available")
        return
    
    # Plot 1: Success rate by agent type
    success_by_agent = df_agents.groupby('agent_type')['success'].agg(['sum', 'count'])
    success_by_agent['rate'] = (success_by_agent['sum'] / success_by_agent['count'] * 100)
    
    bars = ax1.bar(success_by_agent.index, success_by_agent['rate'], 
                   color=['#3498db', '#e74c3c'])
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Agent Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Success Rate by Agent Type', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    
    # Plot 2: Average duration by agent type (successful tasks)
    df_success = df_agents[df_agents['success'] == True]
    duration_by_agent = df_success.groupby('agent_type')['duration'].mean()
    
    bars = ax2.bar(duration_by_agent.index, duration_by_agent.values,
                   color=['#3498db', '#e74c3c'])
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Agent Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Duration (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Duration by Agent Type', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "agent_type_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Created: agent_type_comparison.png")
    plt.close()

def plot_task_analysis(df):
    """Analyze task difficulty and success patterns."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Success rate by task
    success_by_task = df.groupby('task')['success'].agg(['sum', 'count'])
    success_by_task['rate'] = (success_by_task['sum'] / success_by_task['count'] * 100)
    success_by_task = success_by_task.sort_values('rate', ascending=True)
    
    colors = plt.cm.get_cmap('RdYlGn')(success_by_task['rate'] / 100)
    bars = ax1.barh(success_by_task.index, success_by_task['rate'], color=colors)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%',
                ha='left', va='center', fontweight='bold', fontsize=9)
    
    ax1.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Task', fontsize=12, fontweight='bold')
    ax1.set_title('Success Rate by Task Type', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 105)
    
    # Plot 2: Average duration by task (successful only)
    df_success = df[df['success'] == True]
    duration_by_task = df_success.groupby('task')['duration'].mean().sort_values()
    
    bars = ax2.barh(duration_by_task.index, duration_by_task.values, 
                    color=sns.color_palette("viridis", len(duration_by_task)))
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}s',
                ha='left', va='center', fontweight='bold', fontsize=9)
    
    ax2.set_xlabel('Average Duration (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Task', fontsize=12, fontweight='bold')
    ax2.set_title('Average Duration by Task (Successful)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task_analysis.png", dpi=300, bbox_inches='tight')
    print("✓ Created: task_analysis.png")
    plt.close()


def generate_summary_stats(df):
    """Generate and save summary statistics."""
    summary = []
    
    summary.append("=" * 60)
    summary.append("BENCHMARK RESULTS SUMMARY")
    summary.append("=" * 60)
    summary.append(f"\nTotal Benchmark Runs: {len(df)}")
    summary.append(f"Successful Runs: {df['success'].sum()} ({df['success'].mean()*100:.1f}%)")
    summary.append(f"Failed Runs: {(~df['success']).sum()} ({(~df['success']).mean()*100:.1f}%)")
    
    summary.append("\n" + "-" * 60)
    summary.append("BY BACKEND:")
    summary.append("-" * 60)
    for backend in df['backend'].unique():
        backend_df = df[df['backend'] == backend]
        success_rate = backend_df['success'].mean() * 100
        summary.append(f"{backend:15s}: {len(backend_df):3d} runs, {success_rate:5.1f}% success")
    
    summary.append("\n" + "-" * 60)
    summary.append("BY MODEL:")
    summary.append("-" * 60)
    for model in df['model_short'].unique():
        model_df = df[df['model_short'] == model]
        success_rate = model_df['success'].mean() * 100
        avg_duration = model_df[model_df['success']]['duration'].mean()
        summary.append(f"{model:20s}: {len(model_df):3d} runs, {success_rate:5.1f}% success, {avg_duration:6.1f}s avg")
    
    if 'agent_type' in df.columns and df['agent_type'].notna().any():
        summary.append("\n" + "-" * 60)
        summary.append("BY AGENT TYPE:")
        summary.append("-" * 60)
        for agent in df['agent_type'].dropna().unique():
            agent_df = df[df['agent_type'] == agent]
            success_rate = agent_df['success'].mean() * 100
            summary.append(f"{agent:15s}: {len(agent_df):3d} runs, {success_rate:5.1f}% success")
    
    summary.append("\n" + "-" * 60)
    summary.append("BY TASK:")
    summary.append("-" * 60)
    for task in sorted(df['task'].unique()):
        task_df = df[df['task'] == task]
        success_rate = task_df['success'].mean() * 100
        summary.append(f"{task:25s}: {len(task_df):3d} runs, {success_rate:5.1f}% success")
    
    summary.append("\n" + "=" * 60)
    
    summary_text = "\n".join(summary)
    
    # Save to file
    with open(OUTPUT_DIR / "summary_statistics.txt", 'w') as f:
        f.write(summary_text)
    
    print("\n" + summary_text)
    print(f"\n✓ Created: summary_statistics.txt")

def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("BENCHMARK VISUALIZATION GENERATOR")
    print("=" * 60 + "\n")
    
    # Load data
    print("Loading benchmark data...")
    df = load_benchmark_data()
    
    if len(df) == 0:
        print("❌ No data loaded. Exiting.")
        return
    
    # Clean data
    df = clean_model_names(df)
    
    print(f"\nGenerating visualizations in {OUTPUT_DIR}/...\n")
    
    # Generate all visualizations
    plot_success_rate_by_backend(df)
    plot_duration_comparison(df)
    plot_model_size_performance(df)
    plot_agent_type_comparison(df)
    plot_task_analysis(df)
    
    # Generate summary statistics
    generate_summary_stats(df)
    
    print("\n" + "=" * 60)
    print("✓ ALL VISUALIZATIONS COMPLETED!")
    print(f"✓ Output directory: {OUTPUT_DIR.absolute()}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

# Made with Bob
