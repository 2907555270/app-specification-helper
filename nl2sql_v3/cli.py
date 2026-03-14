import logging
import sys
from typing import Optional

import click

from .client.es_client import es_client
from .config import config
from .data.evaluator import Evaluator
from .data.loader import MetadataLoader, QueryLoader, build_index_documents
from .recall.fusion import HybridRetriever

logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """nl2sql_v3 - Natural Language to SQL Table Retrieval CLI Tool"""
    pass


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
    top_k: int,
    show_scores: bool,
    weights: str,
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
        click.echo(f"\nTop {top_k} Related Tables:")
        click.echo("-" * 60)

        results = retriever.retrieve(query)

        if not results:
            click.echo("No related tables found.")
            return

        header = f"{'DB Name':<20} {'Table Name':<20} {'Score':<10} {'Match Type':<10}"
        click.echo(header)
        click.echo("-" * 60)

        for r in results:
            click.echo(
                f"{r.db_name:<20} {r.table_name:<20} {r.score:<10.4f} {r.match_type:<10}"
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

        click.echo("Generating vectors...")
        documents = build_index_documents(tables, use_sparse=True, use_dense=True)

        click.echo(f"Indexing {len(documents)} documents...")
        indexed = es_client.bulk_index(documents, batch_size=batch_size)

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
):
    """Evaluate recall performance using query data."""
    try:
        click.echo("Evaluating...")

        evaluator = Evaluator(
            use_keyword=not no_keyword,
            use_sparse=not no_sparse,
            use_dense=not no_dense,
        )

        result = evaluator.evaluate(db_name=db)

        click.echo("\n" + "=" * 60)
        click.echo("Evaluation Results")
        click.echo("=" * 60)
        click.echo(f"Total Queries:    {result.total_queries}")
        click.echo(f"Hit Rate @ 1:    {result.hit_rate_at_1:.2%}")
        click.echo(f"Hit Rate @ 3:    {result.hit_rate_at_3:.2%}")
        click.echo(f"Hit Rate @ 5:    {result.hit_rate_at_5:.2%}")
        click.echo(f"MRR:             {result.mrr:.4f}")
        click.echo("=" * 60)

        if output:
            import json

            with open(output, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
            click.echo(f"\nResults saved to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Evaluation failed")
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
