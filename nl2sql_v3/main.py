import logging
import sys
import os
from typing import Optional

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        ctx.invoke(interactive)


@cli.command()
def interactive():
    """Enter interactive mode - keep asking for queries."""
    click.echo("=" * 60)
    click.echo("nl2sql_v3 Interactive Mode")
    click.echo("=" * 60)
    click.echo("Type 'quit' or 'exit' to exit")
    click.echo("Type 'help' for available commands")
    click.echo("=" * 60)

    loader = MetadataLoader()
    tables = loader.load()
    click.echo(f"Loaded {len(tables)} tables")

    retriever = HybridRetriever(
        tables=tables,
        weights={
            "keyword": config.recall.weights.keyword,
            "sparse": config.recall.weights.sparse,
            "dense": config.recall.weights.dense,
        }
    )

    while True:
        try:
            query = input("\n> ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                click.echo("Goodbye!")
                break

            if query.lower() == "help":
                click.echo("Available commands:")
                click.echo("  help - Show this help message")
                click.echo("  quit/exit - Exit interactive mode")
                click.echo("  <query> - Enter a natural language query to recall tables")
                continue

            results = retriever.retrieve(query)

            if not results:
                click.echo("No related tables found.")
                continue

            click.echo(f"\nTop {len(results)} Related Tables:")
            click.echo("-" * 60)
            header = f"{'DB Name':<20} {'Table Name':<20} {'Score':<10} {'Match Type':<10}"
            click.echo(header)
            click.echo("-" * 60)

            for r in results:
                click.echo(
                    f"{r.db_name:<20} {r.table_name:<20} {r.score:<10.4f} {r.match_type:<10}"
                )

        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
            break
        except Exception as e:
            click.echo(f"Error: {e}")
            logger.exception("Query failed")


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

        loader = MetadataLoader()
        tables = loader.load()

        retriever = HybridRetriever(
            tables=tables,
            weights=weights_dict,
            top_k=top_k,
            use_keyword=not no_keyword,
            use_sparse=not no_sparse,
            use_dense=not no_dense,
        )

        click.echo(f"Query: {query}")

        results = retriever.retrieve(query)

        if not results:
            click.echo("No related tables found.")
            return

        display_results = results[:top_k] if top_k else results

        click.echo(f"\nTop {len(display_results)} Related Tables:")
        click.echo("-" * 70)

        if show_scores:
            header = f"{'DB Name':<20} {'Table Name':<20} {'Score':<10} {'Rerank':<10} {'Match Type':<12}"
        else:
            header = f"{'DB Name':<20} {'Table Name':<20} {'Match Type':<12}"
        click.echo(header)
        click.echo("-" * 70)

        for r in display_results:
            rerank_str = f"{r.rerank_score:.4f}" if r.rerank_score is not None else "N/A"
            if show_scores:
                click.echo(
                    f"{r.db_name:<20} {r.table_name:<20} {r.score:<10.4f} {rerank_str:<10} {r.match_type:<12}"
                )
            else:
                click.echo(
                    f"{r.db_name:<20} {r.table_name:<20} {r.match_type:<12}"
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
    "--db",
    "-d",
    default=None,
    type=str,
    help="Filter by database name",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=str,
    help="Output file for results",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed results",
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
    db: Optional[str],
    output: Optional[str],
    verbose: bool,
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

        result = evaluator.evaluate(db_name=db, filter_db=True, top_k_values=top_k_values)

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
        result_no_filter = evaluator.evaluate(db_name=db, filter_db=False)

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


def main():
    cli(prog_name="python main.py")


if __name__ == "__main__":
    main()
