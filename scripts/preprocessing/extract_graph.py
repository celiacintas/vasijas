from docling_graph import run_pipeline, PipelineConfig
import litellm
from pathlib import Path
import csv

litellm._turn_on_debug()

output_dir = Path("outputs_graph")
output_dir.mkdir(parents=True, exist_ok=True)

config = PipelineConfig(    
    source="data/thesis_texture.pdf",
    template="templates.dummy_template.EntitiesRelationships",
    inference="local",
    model_override="qwen3:235b-a22b", #llama3.1:8b qwen3:235b-a22b
    provider_override="ollama",
    processing_mode="many-to-one",
    #dump_to_disk=True,
    output_dir="outputs_graph/",
    export_format="csv",

)

# Set API key
#export GEMINI_API_KEY="sk-bHknIORbjwO9KaKX9VzYRA"

print("Processing document...")
context = run_pipeline(config)
graph = context.knowledge_graph
print(f"✅ Complete! Extracted {graph.number_of_nodes()} nodes")

# Export nodes
nodes_csv = output_dir / "nodes.csv"
with open(nodes_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['node_id', 'attributes'])
    for node_id, node_data in graph.nodes(data=True):
        writer.writerow([node_id, str(node_data)])

print(f"✓ Nodes exported to {nodes_csv}")

# Export edges/relationships
edges_csv = output_dir / "edges.csv"
with open(edges_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['source', 'target', 'relation'])
    for source, target, edge_data in graph.edges(data=True):
        relation = edge_data.get('relation', '') if isinstance(edge_data, dict) else ''
        writer.writerow([source, target, relation])

print(f"✓ Edges exported to {edges_csv}")
print(f"\nFiles saved in: {output_dir.absolute()}")