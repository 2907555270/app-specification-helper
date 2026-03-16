import logging
import sys
import os
import time
from typing import Optional

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nl2sql_v3.agent.leader_agent import InteractiveNL2SQLAgent
from nl2sql_v3.agent.nl2sql_agent import NL2SQLAgent
from nl2sql_v3.client.es_client import es_client
from nl2sql_v3.config import config
from nl2sql_v3.data.evaluator import Evaluator
from nl2sql_v3.data.loader import MetadataLoader, stream_build_index_documents
from nl2sql_v3.recall.fusion import HybridRetriever

logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def cli(ctx):
    """nl2sql_v3 - Natural Language to SQL Table Retrieval CLI Tool"""
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


@cli.command()
def chat():
    """Enter interactive multi-turn conversation mode with BI Assistant."""
    import uuid
    
    click.echo("=" * 60)
    click.echo("nl2sql_v3 Multi-turn Chat Mode")
    click.echo("基于 LangGraph 的智能 BI 助手，支持多轮对话")
    click.echo("=" * 60)
    click.echo("Type 'quit' or 'exit' to exit")
    click.echo("Type 'new' to start a new conversation")
    click.echo("Type 'help' for available commands")
    click.echo("=" * 60)

    conversation_id = str(uuid.uuid4())
    agent = InteractiveNL2SQLAgent()

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                click.echo("Goodbye!")
                break

            if user_input.lower() == "new":
                conversation_id = str(uuid.uuid4())
                click.echo("Started new conversation.")
                continue

            if user_input.lower() == "help":
                click.echo("Available commands:")
                click.echo("  help - Show this help message")
                click.echo("  quit/exit - Exit chat mode")
                click.echo("  new - Start a new conversation")
                click.echo("  <query> - Enter a natural language query")
                continue

            start_time = time.time()
            resp = agent.run(
                user_input=user_input,
                conversation_id=conversation_id
            )
            elapsed = time.time() - start_time
            
            click.echo(f"\n[Response ({elapsed:.2f}s)]")
            click.echo(resp["output"])

        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
            break
        except Exception as e:
            click.echo(f"Error: {e}")
            logger.exception("Chat failed")


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--top-k",
    "-k",
    default=5,
    type=int,
    help="Number of results to return",
)
@click.option(
    "--show-scores",
    "-s",
    is_flag=True,
    help="Show detailed scores",
)
@click.option(
    "--weights",
    "-w",
    default="0.3,0.3,0.4",
    type=str,
    help="Fusion weights (keyword,sparse,dense)",
)
@click.option(
    "--no-keyword",
    is_flag=True,
    help="Disable keyword recall",
)
@click.option(
    "--no-sparse",
    is_flag=True,
    help="Disable sparse vector recall",
)
@click.option(
    "--no-dense",
    is_flag=True,
    help="Disable dense vector recall",
)
def recall(
    query: str,
    weights: str,
    top_k: Optional[int],
    show_scores: bool,
    no_keyword: bool,
    no_sparse: bool,
    no_dense: bool,
):
    """Recall related tables for a natural language query."""
    try:
        weight_list = [float(w) for w in weights.split(",")]
        if len(weight_list) != 3:
            click.echo("Error: Weights must be three comma-separated values", err=True)
            sys.exit(1)

        weights_dict = {
            "keyword": weight_list[0],
            "sparse": weight_list[1],
            "dense": weight_list[2],
        }

        retriever = HybridRetriever(
            weights=weights_dict,
            use_keyword=not no_keyword,
            use_sparse=not no_sparse,
            use_dense=not no_dense
        )

        click.echo(f"Query: {query}")

        results = retriever.retrieve(query)

        if not results:
            click.echo("No related tables found.")
            return

        display_results = results[:top_k] if top_k else results

        click.echo(f"\nTop {len(display_results)} Related Tables:")
        click.echo("-" * 90)

        header = f"{'DB Name':<20} {'Table Name':<20} {'Score':<10} {'Rerank Score':<15} {'Match Type':<12}"

        click.echo(header)
        click.echo("-" * 90)

        for r in display_results:
            rerank_str = f"{r.rerank_score:.4f}" if r.rerank_score is not None else "N/A"
            click.echo(
                f"{r.db_name:<20} {r.table_name:<20} {r.score:<10.4f} {rerank_str:<15} {r.match_type:<12}"
            )


    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Recall failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force rebuild index",
)
@click.option(
    "--batch-size",
    "-b",
    default=100,
    type=int,
    help="Batch size for indexing",
)
def build_index(force: bool, batch_size: int):
    """Build Elasticsearch index from metadata."""
    try:
        click.echo("Building index...")

        es_client.create_index(force=force)

        loader = MetadataLoader()
        tables = loader.load()
        click.echo(f"Loaded {len(tables)} tables")

        click.echo("Generating vectors and indexing...")
        batch = []
        indexed = 0
        for doc in stream_build_index_documents(tables, use_sparse=True, use_dense=True):
            batch.append(doc)
            if len(batch) >= batch_size:
                indexed += es_client.bulk_index(batch)
                click.echo(f"Indexed {indexed} documents...")
                batch = []
        
        if batch:
            indexed += es_client.bulk_index(batch)
        
        click.echo(f"Successfully indexed {indexed} documents")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Build index failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--output",
    "-o",
    default=None,
    type=str,
    help="Output file for results",
)
@click.option(
    "--no-keyword",
    is_flag=True,
    help="Disable keyword recall",
)
@click.option(
    "--no-sparse",
    is_flag=True,
    help="Disable sparse vector recall",
)
@click.option(
    "--no-dense",
    is_flag=True,
    help="Disable dense vector recall",
)
def evaluate(
    output: Optional[str],
    no_keyword: bool,
    no_sparse: bool,
    no_dense: bool,
    top_k_values: Optional[list[int]] = None,
):
    """Evaluate recall performance using query data."""
    try:
        click.echo("Evaluating...")

        evaluator = Evaluator(
            use_keyword=not no_keyword,
            use_sparse=not no_sparse,
            use_dense=not no_dense,
        )

        result = evaluator.evaluate(filter_db=True, top_k_values=top_k_values)

        click.echo("\n" + "=" * 60)
        click.echo("Evaluation Results (With DB Filter)")
        click.echo("=" * 60)
        click.echo(f"Total Queries:        {result.total_queries}")
        click.echo(f"Hit Rate @ 1:        {result.hit_rate_at_1:.2%}")
        click.echo(f"Hit Rate @ 3:        {result.hit_rate_at_3:.2%}")
        click.echo(f"Hit Rate @ 5:        {result.hit_rate_at_5:.2%}")
        click.echo(f"MRR:                 {result.mrr:.4f}")
        click.echo(f"Total Time:          {result.total_time:.2f}s")
        click.echo(f"Avg Time:            {result.avg_time:.3f}s")
        if result.min_rerank_threshold is not None:
            click.echo(f"Min Rerank Threshold: {result.min_rerank_threshold:.4f}")
        click.echo("=" * 60)

        click.echo("\nRunning evaluation WITHOUT DB filter...")
        result_no_filter = evaluator.evaluate(filter_db=False, top_k_values=top_k_values)

        click.echo("\n" + "=" * 60)
        click.echo("Evaluation Results (Without DB Filter)")
        click.echo("=" * 60)
        click.echo(f"Total Queries:        {result_no_filter.total_queries}")
        click.echo(f"Hit Rate @ 1:        {result_no_filter.hit_rate_at_1:.2%}")
        click.echo(f"Hit Rate @ 3:        {result_no_filter.hit_rate_at_3:.2%}")
        click.echo(f"Hit Rate @ 5:        {result_no_filter.hit_rate_at_5:.2%}")
        click.echo(f"MRR:                 {result_no_filter.mrr:.4f}")
        click.echo(f"Total Time:          {result_no_filter.total_time:.2f}s")
        click.echo(f"Avg Time:            {result_no_filter.avg_time:.3f}s")
        if result_no_filter.min_rerank_threshold is not None:
            click.echo(f"Min Rerank Threshold: {result_no_filter.min_rerank_threshold:.4f}")
        click.echo("=" * 60)

        if output:
            import json

            with open(output, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
                json.dump(result_no_filter.model_dump(), f, indent=2, ensure_ascii=False)
            click.echo(f"\nResults saved to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Evaluation failed")
        sys.exit(1)


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--no-fewshot",
    is_flag=True,
    help="Disable few-shot examples in prompt",
)
@click.option(
    "--temperature",
    "-t",
    default=0.0,
    type=float,
    help="LLM temperature",
)
@click.option(
    "--weights",
    "-w",
    default="0.1,0.6,0.3",
    type=str,
    help="Fusion weights (keyword,sparse,dense)",
)
@click.option(
    "--no-keyword",
    is_flag=True,
    help="Disable keyword recall",
)
@click.option(
    "--no-sparse",
    is_flag=True,
    help="Disable sparse vector recall",
)
@click.option(
    "--no-dense",
    is_flag=True,
    help="Disable dense vector recall",
)
@click.option(
    "--no-execute",
    is_flag=True,
    help="Disable SQL execution",
)
def agent(
    query: str,
    no_fewshot: bool,
    temperature: float,
    weights: str = None,
    no_keyword: bool = False,
    no_sparse: bool = False,
    no_dense: bool = False,
    no_execute: bool = False,
):
    """Generate SQL from natural language query using agent."""
    try:
        if weights is None:
            weights_dict = {
                "keyword": config.recall.weights.keyword,
                "sparse": config.recall.weights.sparse,
                "dense": config.recall.weights.dense,
            }
        else:
            weight_list = [float(w) for w in weights.split(",")]
            weights_dict = {
                "keyword": weight_list[0],
                "sparse": weight_list[1],
                "dense": weight_list[2],
            }
        if len(weight_list) != 3:
            click.echo("Error: Weights must be three comma-separated values", err=True)
            sys.exit(1)

        retriever = HybridRetriever(
            weights=weights_dict,
            use_keyword=not no_keyword,
            use_sparse=not no_sparse,
            use_dense=not no_dense,
        )

        nl2sql_agent = NL2SQLAgent(
            retriever=retriever,
            include_fewshot=not no_fewshot,
            temperature=temperature,
        )

        click.echo("=" * 60)
        click.echo(f"Query: {query}")
        click.echo("=" * 60)

        result = nl2sql_agent.run(query)

        click.echo("\n[Timings]")
        click.echo(f"  total: {result.get('timings', {}).get('total', 0):.3f}s")

        click.echo("\n[Recalled Tables]")
        for tbl in result.get("selected_tables", []):
            click.echo(f"  - {tbl}")

        click.echo(f"\n[Generated SQL]")
        click.echo(result.get("sql", ""))

        click.echo(f"\n[Confidence] {result.get('confidence', 0.0):.2f}")
        click.echo(f"\n[Explanation]")
        click.echo(result.get("explanation", ""))

        click.echo(f"\n[Used Columns]")
        for col in result.get("used_columns", []):
            click.echo(f"  - {col}")

        exec_result = result.get("execution_result")
        if exec_result:
            click.echo(f"\n[Execution Result]")
            if "error" in exec_result:
                click.echo(f"  Error: {exec_result['error']}")
            elif isinstance(exec_result, list):
                click.echo(f"  {len(exec_result)} rows returned:")
                for i, row in enumerate(exec_result[:10]):
                    click.echo(f"  Row {i+1}: {row}")
                if len(exec_result) > 10:
                    click.echo(f"  ... and {len(exec_result) - 10} more rows")
            elif isinstance(exec_result, dict):
                click.echo(f"  {exec_result}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Agent failed")
        sys.exit(1)


def main():
    cli(prog_name="python main.py")


if __name__ == "__main__":
    main()
