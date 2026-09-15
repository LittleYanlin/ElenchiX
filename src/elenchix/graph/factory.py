from elenchix.config import GraphConfig

from .base import GraphStore
from .json_store import JsonGraphStore


def create_graph_store(config: GraphConfig) -> GraphStore:
    return JsonGraphStore(config)
