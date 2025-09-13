import csv
import os
from pathlib import Path
from neo4j import GraphDatabase
import argparse

# --------- CONFIG ----------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "123123123")

# CSV path can be provided via env or CLI; default to file next to this script
DEFAULT_CSV_PATH = str(
    Path(__file__).resolve().parent / "SUBRELOBJ.csv"
)
CSV_PATH = os.getenv("MATKG_SUBRELOBJ_CSV", DEFAULT_CSV_PATH)
# ---------------------------

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

CREATE_SCHEMA = """
CREATE CONSTRAINT term_name_unique IF NOT EXISTS
FOR (t:Term) REQUIRE t.name IS UNIQUE
"""

# Note: randomUUID() is built-in in Neo4j 5.x; no APOC needed.
MERGE_TERM = """
MERGE (t:Term {name: $name})
ON CREATE SET
  t.id = randomUUID(),
  t.created_at = datetime()
RETURN t.id AS id
"""

# Relationship type is dynamic; must be backticked to allow dashes/spaces.
# We also store the raw type in a property for convenient filtering.
CREATE_REL_TEMPLATE = """
MATCH (a:Term {name: $a}), (b:Term {name: $b})
MERGE (a)-[r:`%s`]->(b)
ON CREATE SET
  r.created_at = datetime(),
  r.original_type = $rel_type
RETURN type(r) AS type
"""

def normalize(s: str) -> str:
    # Strip BOM, trim, collapse inner whitespace
    if s is None:
        return ""
    s = s.replace("\ufeff", "").strip()
    # preserve punctuation; just normalize whitespace
    return " ".join(s.split())

def read_rows(csv_path: str, max_rows: int | None = None):
    """
    Tries TSV first (your sample shows tabs). Falls back to comma CSV.
    Requires headers: Subject, Rel, Object (case-insensitive tolerant).
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    # Try TSV
    with p.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
    is_tsv = ("\t" in sample) and ("Subject" in sample or "subject" in sample)
    delim = "\t" if is_tsv else ","

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        # Normalize header keys once
        field_map = {k.lower(): k for k in reader.fieldnames or []}
        need = ["subject", "rel", "object"]
        if not all(h in field_map for h in need):
            raise ValueError(f"CSV must have headers: Subject, Rel, Object. Found: {reader.fieldnames}")

        subj_key, rel_key, obj_key = field_map["subject"], field_map["rel"], field_map["object"]

        count = 0
        for row in reader:
            subj = normalize(row.get(subj_key, ""))
            rel  = normalize(row.get(rel_key,  ""))
            obj  = normalize(row.get(obj_key,  ""))
            if subj and rel and obj:
                yield subj, rel, obj
                count += 1
                if max_rows is not None and count >= max_rows:
                    break

def main(max_rows: int | None = None, clear_first: bool = False):
    with driver.session() as session:
        # 1) Ensure uniqueness on node names so no duplicates are created
        session.run(CREATE_SCHEMA)
        if clear_first:
            session.run("MATCH (n) DETACH DELETE n")

        count_nodes = 0
        count_rels  = 0

        for subj, rel, obj in read_rows(CSV_PATH, max_rows=max_rows):
            # 2) Create/merge nodes (each cell is a node; label Term)
            session.run(MERGE_TERM, name=subj)
            session.run(MERGE_TERM, name=obj)
            count_nodes += 2  # merged (no dupes thanks to constraint)

            # 3) Create/merge relationship with the raw Rel as the type
            #    Relationship types can include dashes/spaces → backtick the type.
            cypher = CREATE_REL_TEMPLATE % rel.replace("`", "ˋ")  # guard accidental backticks
            session.run(cypher, a=subj, b=obj, rel_type=rel)
            count_rels += 1

        # Optional: small stats query
        stats_nodes = session.run("MATCH (t:Term) RETURN count(t) AS c").single()["c"]
        stats_rels  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    driver.close()
    print("Done.")
    print(f"Processed rows: ~{count_rels}")
    print(f"Distinct nodes in DB (label :Term): {stats_nodes}")
    print(f"Total relationships in DB: {stats_rels}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Term graph in Neo4j from SUBRELOBJ.csv")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to SUBRELOBJ.csv")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to ingest")
    parser.add_argument("--clear-first", action="store_true", help="Clear database before ingest")
    args = parser.parse_args()

    # Allow overriding CSV_PATH via CLI
    if args.csv:
        CSV_PATH = args.csv

    main(max_rows=args.max_rows, clear_first=args.clear_first)
