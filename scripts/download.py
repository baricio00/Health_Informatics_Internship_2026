#!/usr/bin/env python3
"""
Scarica gli ultimi N file `summary.json` (fold_0) dal datastore AzureML
e crea un file aggregato con tutti i risultati.

Requisiti:
  pip install azure-identity azure-storage-blob azureml-core

Esempio:
  python scripts/download.py --output_dir ./azureml_summaries_ROI_128

"""
import argparse
import json
import logging
from pathlib import Path
from typing import List

from azureml.core import Workspace, Datastore
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential


def get_blob_service_client(datastore):
    # Estrae account/container/credenziali dal Datastore
    account_name = getattr(datastore, "account_name", None) or getattr(datastore, "storage_account_name", None)
    container_name = getattr(datastore, "container_name", None)

    if account_name is None or container_name is None:
        raise RuntimeError("Datastore non espone account/container info. Controlla il datastore.")

    sas_token = getattr(datastore, "sas_token", None)
    account_key = getattr(datastore, "account_key", None)

    if account_key:
        credential = account_key
    elif sas_token:
        credential = sas_token
    else:
        # Fallback: usa DefaultAzureCredential (richiede az login o variabili d'ambiente)
        credential = DefaultAzureCredential()

    service = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=credential)
    return service, container_name


def list_summary_blobs(container_client, prefix: str, suffix: str) -> List:
    # Restituisce oggetti BlobProperties per i file che terminano con `suffix`
    out = []
    for b in container_client.list_blobs(name_starts_with=prefix):
        if b.name.endswith(suffix):
            out.append(b)
    return out


def download_blob_to_file(container_client, blob_name: str, dst_path: str):
    blob_client = container_client.get_blob_client(blob_name)
    with open(dst_path, "wb") as f:
        stream = blob_client.download_blob()
        f.write(stream.readall())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription_id", default="ab211f7b-463f-4833-9605-d260e596a35a")
    parser.add_argument("--resource_group", default="73da10b4-5dff-54e2-db0d-3a1fab882485")
    parser.add_argument("--workspace_name", default="73da10b45dff54e2db0d3a1fab882485")
    parser.add_argument("--datastore", default="workspaceblobstore")
    parser.add_argument("--prefix", default="azureml/")
    parser.add_argument("--suffix", default="output_model/fold_0/summary.json")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--output_dir", default="./azureml_summaries_ROI_128")
    parser.add_argument("--make_csv", action="store_true", help="Produce anche un CSV appiattito (se summary.json è un dict)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Connessione a Workspace...")
    ws = Workspace(subscription_id=args.subscription_id, resource_group=args.resource_group, workspace_name=args.workspace_name)

    logging.info(f"Recupero Datastore '{args.datastore}'...")
    ds = Datastore.get(ws, args.datastore)

    blob_service, container_name = get_blob_service_client(ds)
    container_client = blob_service.get_container_client(container_name)

    logging.info(f"Elenco blob con prefisso '{args.prefix}' e suffisso '{args.suffix}'...")
    blobs = list_summary_blobs(container_client, prefix=args.prefix, suffix=args.suffix)

    if not blobs:
        logging.info("Nessun summary.json trovato con il pattern richiesto.")
        return

    # Ordina per last_modified (se non presente, lascia come è)
    blobs_sorted = sorted(blobs, key=lambda b: getattr(b, "last_modified", None) or getattr(b, "creation_time", None) or 0, reverse=True)
    selected = blobs_sorted[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregated = []

    for b in selected:
        blob_name = b.name
        parts = blob_name.split("/")
        run_id = parts[1] if len(parts) > 1 else parts[0].replace('/', '_')
        local_run_dir = out_dir / run_id
        local_run_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_run_dir / "summary.json"

        logging.info(f"Scarico {blob_name} -> {local_file} (modified: {b.last_modified})")
        try:
            download_blob_to_file(container_client, blob_name, str(local_file))
            with open(local_file, 'r', encoding='utf-8') as fh:
                try:
                    data = json.load(fh)
                except Exception:
                    logging.exception(f"Impossibile fare parse JSON per {local_file}")
                    data = None
        except Exception:
            logging.exception(f"Errore scaricando {blob_name}")
            data = None

        aggregated.append({
            "run_id": run_id,
            "blob_path": blob_name,
            "last_modified": b.last_modified.isoformat() if hasattr(b.last_modified, 'isoformat') else str(b.last_modified),
            "summary": data,
        })

    agg_file = out_dir / "aggregated_summaries.json"
    with open(agg_file, 'w', encoding='utf-8') as fh:
        json.dump(aggregated, fh, indent=2, ensure_ascii=False)

    logging.info(f"Aggregati {len(aggregated)} summary salvati in: {agg_file}")

    if args.make_csv:
        import csv

        csv_file = out_dir / "aggregated_summaries.csv"
        # Raccogli tutte le chiavi top-level presenti nei summary
        keys = set()
        for item in aggregated:
            if isinstance(item.get('summary'), dict):
                keys.update(item['summary'].keys())
        keys = sorted(keys)

        with open(csv_file, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            header = ['run_id', 'blob_path', 'last_modified'] + keys
            writer.writerow(header)
            for item in aggregated:
                row = [item['run_id'], item['blob_path'], item['last_modified']]
                summary = item.get('summary') or {}
                for k in keys:
                    v = summary.get(k, '')
                    if isinstance(v, (dict, list)):
                        v = json.dumps(v, ensure_ascii=False)
                    row.append(v)
                writer.writerow(row)

        logging.info(f"CSV appiattito salvato in: {csv_file}")


if __name__ == '__main__':
    main()