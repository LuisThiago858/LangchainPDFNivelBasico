# 1) Backup do arquivo atual

# 2) Sobrescrever com a versão atualizada (com --index-mode)

# 01_ingest.py
# Ingestão de Dados (ETL) para Vertex AI Vector Search (idempotente)
#
# Adições:
# - --index-mode {latest,new}: escolha entre reutilizar o índice mais recente (ou do .ingest_state.json)
#   ou criar SEMPRE um índice novo.
# - Persistência de index_id/endpoint_id em .ingest_state.json para reuso confiável.
#
# Formatos suportados:
#   1) "linha por valor" (colunas: mes_ano, valor, tipo, empresa_id)
#   2) planilha com "Mês"/"Mes" + colunas de anos (ex.: 2023, 2024...)
#
# IDs estáveis por registro: empresa:sheet:ano:mes:tipo
# Manifesto: mapa id -> hash( conteúdo + metadados )
#
# merge:   insere/atualiza apenas o que mudou (pula inalterados)
# replace: apaga IDs conhecidos da empresa e reinsere
#
# Requisitos:
#   pip install -U python-dotenv pandas google-cloud-aiplatform `
#                  langchain langchain-google-vertexai
#
# Variáveis de ambiente (ou .env):
#   GOOGLE_CLOUD_PROJECT
#   GOOGLE_CLOUD_LOCATION          (ex.: us-central1)
#   VERTEX_STAGING_BUCKET          (ex.: gs://meu-bucket)
#   VERTEX_INDEX_NAME              (default: indices-financeiros-gisa-v1)
#   VERTEX_ENDPOINT_NAME           (default: endpoint-financeiros-gisa-v1)
#   VERTEX_VECTOR_DIM              (default: 768)
#   INGEST_BATCH                   (opcional, default: 500)
#   DRY_RUN                        (opcional; "1" = não grava no índice)
#   VERTEX_FORCE_NEW_INDEX         (compatibilidade; CLI --index-mode tem prioridade)

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# LangChain + Vertex
from langchain.schema import Document
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore
from google.cloud import aiplatform
from google.api_core.exceptions import PermissionDenied
from google.auth import default as ga_default

# ========= 0. ENV/Config ========= #
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
REGION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
BUCKET = os.getenv("VERTEX_STAGING_BUCKET", "").strip()

INDEX_DISPLAY_NAME = os.getenv("VERTEX_INDEX_NAME", "indices-financeiros-gisa-v1").strip()
ENDPOINT_DISPLAY_NAME = os.getenv("VERTEX_ENDPOINT_NAME", "endpoint-financeiros-gisa-v1").strip()
VECTOR_DIMENSIONS = int(os.getenv("VERTEX_VECTOR_DIM", "768").strip() or "768")

INGEST_BATCH = int(os.getenv("INGEST_BATCH", "500"))
DRY_RUN = os.getenv("DRY_RUN", "0").strip() == "1"

# ========= Helpers para estado do índice/endpoint ========= #
STATE_PATH = Path(".ingest_state.json")

def _load_ingest_state() -> tuple[str | None, str | None]:
    if not STATE_PATH.exists():
        return None, None
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return data.get("index_id"), data.get("endpoint_id")
    except Exception:
        return None, None

def _save_ingest_state(index_id: str, endpoint_id: str) -> None:
    STATE_PATH.write_text(
        json.dumps({"index_id": index_id, "endpoint_id": endpoint_id}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def _pick_latest_by_displayprefix(objs, name_prefix: str):
    """
    Escolhe o mais recente cujo display_name seja exatamente name_prefix
    ou comece com name_prefix (cobre sufixos com timestamp).
    """
    filtered = []
    for o in objs:
        try:
            dn = o.display_name
        except Exception:
            try:
                dn = o.gca_resource.display_name
            except Exception:
                dn = ""
        if dn == name_prefix or (dn and dn.startswith(name_prefix)):
            filtered.append(o)
    if not filtered:
        return None

    # ordena por create_time (quando disponível)
    def _ctime(obj):
        try:
            ct = obj.gca_resource.create_time  # RFC3339
            return pd.to_datetime(ct)
        except Exception:
            return pd.Timestamp.min

    filtered.sort(key=_ctime, reverse=True)
    return filtered[0]

# ========= 1. Meses/normalização ========= #
MONTH_ALIAS = {
    "Jan": "Janeiro", "Fev": "Fevereiro", "Mar": "Março", "Abr": "Abril",
    "Mai": "Maio", "Jun": "Junho", "Jul": "Julho", "Ago": "Agosto",
    "Set": "Setembro", "Out": "Outubro", "Nov": "Novembro", "Dez": "Dezembro",
}
MONTH_PT = {
    1: ("Janeiro", "Jan"),  2: ("Fevereiro", "Fev"), 3: ("Março", "Mar"),
    4: ("Abril", "Abr"),    5: ("Maio", "Mai"),      6: ("Junho", "Jun"),
    7: ("Julho", "Jul"),    8: ("Agosto", "Ago"),    9: ("Setembro", "Set"),
    10:("Outubro", "Out"), 11:("Novembro", "Nov"),  12:("Dezembro", "Dez"),
}
MONTH_NAME_TO_ABBR = {
    "janeiro": "Jan", "fevereiro": "Fev", "março": "Mar", "marco": "Mar",
    "abril": "Abr", "maio": "Mai", "junho": "Jun", "julho": "Jul",
    "agosto": "Ago", "setembro": "Set", "outubro": "Out",
    "novembro": "Nov", "dezembro": "Dez",
    "jan": "Jan","fev":"Fev","mar":"Mar","abr":"Abr","mai":"Mai","jun":"Jun",
    "jul":"Jul","ago":"Ago","set":"Set","out":"Out","nov":"Nov","dez":"Dez",
}

def normalize_tipo(txt: Optional[str]) -> Optional[str]:
    if not txt:
        return None
    s = str(txt).lower().strip()
    # remover acentos simples
    s = (s.replace("í", "i").replace("ó", "o").replace("é", "e")
           .replace("á", "a").replace("â", "a").replace("ã", "a"))
    s = s.replace(" ", "_").replace("-", "_")
    if s in {"ticket", "ticket_medio", "ticketmedio", "ticket_médio"}:
        return "ticket_medio"
    if s in {"faturamento", "faturacao"}:
        return "faturamento"
    if s in {"demanda"}:
        return "demanda"
    return s

# ========= 2. Index/Endpoint ========= #
def _perm_msg(project_id: str) -> str:
    return (
        "Sem permissão no projeto ou API/billing desabilitados.\n"
        f"Projeto: {project_id}\n"
        "Verifique:\n"
        " - GOOGLE_CLOUD_PROJECT correto\n"
        " - gcloud auth application-default login\n"
        " - APIs habilitadas: aiplatform, storage, compute\n"
        " - Billing ativo\n"
        " - IAM: sua conta com roles/aiplatform.admin e acesso ao bucket\n"
        " - Service Agent do Vertex com acesso ao bucket (storage.objectAdmin)\n"
    )

def ensure_vertex_index_and_endpoint(
    project_id: str,
    region: str,
    bucket: str,
    index_display_name: str,
    endpoint_display_name: str,
    dimensions: int = 768,
    distance: str = "DOT_PRODUCT_DISTANCE",
    index_mode: str = "latest",  # <-- NOVO
) -> Tuple[str, str]:
    """
    index_mode:
      - "new":     cria SEMPRE um índice novo (com sufixo timestamp) e salva no .ingest_state.json
      - "latest":  reusa o estado (.ingest_state.json) ou pega o mais recente por display_name;
                   se não existir, cria e salva no estado.

    Observação: se VERTEX_FORCE_NEW_INDEX=1 estiver setado e index_mode NÃO for passado,
    trata como "new". O CLI (--index-mode) tem prioridade sobre o env.
    """
    try:
        aiplatform.init(project=project_id, location=region, staging_bucket=bucket)
    except PermissionDenied as e:
        raise RuntimeError(_perm_msg(project_id)) from e

    # Compatibilidade com env (só se CLI não for "new"/"latest" explicitamente)
    env_force_new = os.getenv("VERTEX_FORCE_NEW_INDEX", "").strip() == "1"
    if index_mode not in {"new", "latest"}:
        index_mode = "new" if env_force_new else "latest"

    def _ensure_endpoint(ep_display: str):
        eps = aiplatform.MatchingEngineIndexEndpoint.list()
        picked = _pick_latest_by_displayprefix(eps, ep_display)
        return picked if picked else aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=ep_display,
            public_endpoint_enabled=True,
        )

    def _deploy_if_needed(endpoint, index):
        ep_obj = aiplatform.MatchingEngineIndexEndpoint(endpoint.resource_name)
        deployed_ids = [d.index.split("/")[-1] for d in ep_obj.gca_resource.deployed_indexes]
        idx_id = index.resource_name.split("/")[-1]
        if idx_id not in deployed_ids:
            dep_id = f"dep_{idx_id[:8]}"
            op = ep_obj.deploy_index(index=index, deployed_index_id=dep_id)
            try:
                op.result(timeout=3600)
            except Exception:
                pass
        return idx_id, ep_obj.resource_name.split("/")[-1]

    # ============ MODO "new": criar SEMPRE ============
    if index_mode == "new":
        ts = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        idx_name = f"{index_display_name}-{ts}"
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=idx_name,
            dimensions=dimensions,
            approximate_neighbors_count=150,
            distance_measure_type=distance,
            index_update_method="STREAM_UPDATE",
        )
        endpoint = _ensure_endpoint(endpoint_display_name)
        index_id, endpoint_id = _deploy_if_needed(endpoint, index)
        _save_ingest_state(index_id, endpoint_id)
        return index_id, endpoint_id

    # ============ MODO "latest": reusar ou criar ============
    # 1) Tentar reusar estado .ingest_state.json
    state_index_id, state_endpoint_id = _load_ingest_state()
    if state_index_id and state_endpoint_id:
        try:
            _ = aiplatform.MatchingEngineIndex(state_index_id)
            _ = aiplatform.MatchingEngineIndexEndpoint(state_endpoint_id)
            return state_index_id, state_endpoint_id
        except Exception:
            pass  # estado inválido → continua

    # 2) Tentar pegar o MAIS RECENTE por prefixo do display_name
    idx_list = aiplatform.MatchingEngineIndex.list()
    picked_idx = _pick_latest_by_displayprefix(idx_list, index_display_name)
    if picked_idx is None:
        # 3) Criar índice "raiz" (sem timestamp) se não houver
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=index_display_name,
            dimensions=dimensions,
            approximate_neighbors_count=150,
            distance_measure_type=distance,
            index_update_method="STREAM_UPDATE",
        )
    else:
        index = picked_idx

    endpoint = _ensure_endpoint(endpoint_display_name)
    index_id, endpoint_id = _deploy_if_needed(endpoint, index)
    _save_ingest_state(index_id, endpoint_id)
    return index_id, endpoint_id

def open_vector_vertexai(bucket: str, index_id: str, endpoint_id: str) -> VectorSearchVectorStore:
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket)
    embedding = VertexAIEmbeddings(model_name="text-embedding-005")  # 768 dims
    return VectorSearchVectorStore.from_components(
        project_id=PROJECT_ID,
        region=REGION,
        gcs_bucket_name=bucket.replace("gs://", ""),
        index_id=index_id,
        endpoint_id=endpoint_id,
        embedding=embedding,
        stream_update=True,
    )

# ========= 3. Loaders ========= #
def choose_loader(path: str) -> str:
    df = pd.read_excel(path, nrows=5)
    cols = {str(c).lower().strip() for c in df.columns}
    return "line" if {"mes_ano", "valor", "tipo", "empresa_id"}.issubset(cols) else "excel"

def _infer_tipo_from_context(context: str, sheet_name: Optional[str] = None) -> Optional[str]:
    txt = f"{context or ''} || {sheet_name or ''}".lower()
    if "faturamento" in txt:
        return "faturamento"
    if "demanda" in txt:
        return "demanda"
    if "ticket" in txt or "tícket" in txt:
        return "ticket_medio"
    return None

def excel_table_loader(path: str, empresa_id: int) -> List[Document]:
    """
    Lê todas as sheets; assume:
      - Primeira linha "contexto"
      - Segunda linha cabeçalho
      - Demais linhas = dados, contendo col "Mês/Mes" + colunas de anos (ex.: 2024, 2025)
    Gera um Document por (mês, ano).
    """
    dfs = pd.read_excel(path, sheet_name=None, header=None)
    docs: List[Document] = []
    for sheet_name, raw in dfs.items():
        if raw is None or raw.empty:
            continue

        first_row_idx = raw.first_valid_index()
        if first_row_idx is None:
            continue

        context = str(raw.iloc[first_row_idx, 0]).strip()
        header_idx = first_row_idx + 1
        header = [str(h).strip() for h in raw.iloc[header_idx]]

        data = raw.iloc[header_idx + 1 :].copy()
        data.columns = [str(c).strip() for c in header]
        data = data.dropna(how="all")

        tipo_ctx = normalize_tipo(_infer_tipo_from_context(context, sheet_name))

        # Normaliza "Mes" -> "Mês"
        has_mes = ("Mês" in data.columns) or ("Mes" in data.columns)
        if not has_mes:
            # bloco único
            meta = {"sheet": str(sheet_name), "empresa_id": str(empresa_id)}
            if tipo_ctx:
                meta["tipo"] = tipo_ctx
            texto = f"{context}\n{data.to_csv(index=False, sep='|')}"
            docs.append(Document(texto, metadata=meta))
            continue

        if "Mes" in data.columns and "Mês" not in data.columns:
            data.rename(columns={"Mes": "Mês"}, inplace=True)

        # Espera colunas de ano (ex.: 2024, 2025)
        anos_cols = [c for c in data.columns if re.fullmatch(r"\d{4}", str(c))]
        if not anos_cols:
            # fallback: bloco único
            meta = {"sheet": str(sheet_name), "empresa_id": str(empresa_id)}
            if tipo_ctx:
                meta["tipo"] = tipo_ctx
            texto = f"{context}\n{data.to_csv(index=False, sep='|')}"
            docs.append(Document(texto, metadata=meta))
            continue

        long_df = pd.melt(
            data,
            id_vars=["Mês"],
            value_vars=anos_cols,
            var_name="Ano",
            value_name="Saldo"
        ).dropna(subset=["Saldo"])

        for _, row in long_df.iterrows():
            m_abrev = str(row["Mês"]).strip().replace(".", "")  # "Mai." -> "Mai"
            m_full  = MONTH_ALIAS.get(m_abrev, m_abrev)
            ano     = int(row["Ano"])
            saldo   = row["Saldo"]
            texto = f"Contexto: {context}\nEmpresa: {empresa_id}\nMês: {m_full} ({m_abrev})\nAno: {ano}\nSaldo: {saldo}"
            meta: Dict[str, str] = {
                "sheet": str(sheet_name),
                "mes": m_abrev,
                "ano": str(ano),
                "empresa_id": str(empresa_id),
            }
            if tipo_ctx:
                meta["tipo"] = tipo_ctx
            docs.append(Document(texto, metadata=meta))
    return docs

def line_per_value_loader(path: str, only_empresa: Optional[int] = None) -> List[Document]:
    """
    Espera colunas: mes_ano, valor, tipo, empresa_id
    Gera um Document por linha com metadados normalizados.
    Se only_empresa estiver definido, filtra o DataFrame por esse empresa_id.
    """
    df = pd.read_excel(path)
    df.columns = [str(c).lower().strip() for c in df.columns]
    req = {"mes_ano", "valor", "tipo", "empresa_id"}
    if not req.issubset(set(df.columns)):
        raise ValueError("Colunas esperadas: mes_ano, valor, tipo, empresa_id.")

    if only_empresa is not None:
        before = len(df)
        df = df[df["empresa_id"].astype("int64") == int(only_empresa)]
        removed = before - len(df)
        if removed > 0:
            print(f"   ↳ Filtrando empresa_id={only_empresa}: {removed} linhas de outras empresas foram ignoradas.")

    sheet_from_file = Path(path).stem  # opcional para diferenciar por arquivo
    docs: List[Document] = []
    for _, row in df.iterrows():
        dt = pd.to_datetime(row["mes_ano"])
        ano, mesN = int(dt.year), int(dt.month)
        mes_full, mes_abrev = MONTH_PT[mesN]
        valor = row["valor"]
        tipo  = normalize_tipo(row["tipo"])
        emp   = str(int(row["empresa_id"]))  # string para filtros textuais
        texto = (
            f"Empresa ID: {emp}\n"
            f"Tipo de indicador: {tipo}\n"
            f"Ano: {ano}\n"
            f"Mês: {mes_full} ({mes_abrev})\n"
            f"Valor: {valor}"
        )
        md: Dict[str, str] = {
            "empresa_id": emp,
            "ano": str(ano),
            "mes": mes_abrev,
            "sheet": sheet_from_file,  # ajuda a separar origens
        }
        if tipo:
            md["tipo"] = tipo
        docs.append(Document(texto, metadata=md))
    return docs

# ========= 4. IDs estáveis + Hash/Manifesto ========= #
def make_doc_id(d: Document) -> str:
    """
    ID determinístico por registro. Atualiza payload se o mesmo (empresa, sheet, ano, mes, tipo) reaparecer.
    Para docs sem mês/ano, usa marcador 'all'.
    """
    md = d.metadata or {}
    emp   = str(md.get("empresa_id", "x"))
    sheet = str(md.get("sheet", "na")).replace(" ", "_").lower()
    ano   = str(md.get("ano", "all"))
    mes   = str(md.get("mes", "all"))
    tipo  = str(md.get("tipo", "na")).lower()
    key = f"{emp}:{sheet}:{ano}:{mes}:{tipo}"
    return key

def doc_fingerprint(text: str, metadata: Dict[str, str]) -> str:
    """
    Hash estável do conteúdo do documento + metadados normalizados.
    Se texto/metadata não mudarem, o hash permanece igual entre execuções.
    """
    norm_text = (text or "").strip()
    payload = json.dumps({"t": norm_text, "m": metadata}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def load_manifest(path: str) -> Dict[str, str]:
    """
    Carrega manifesto como { id: hash, ... }.
    Retrocompatível com versão antiga (lista de IDs) -> converte para {id: ""}.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # CORREÇÃO: Itera sobre a lista 'data' para criar o dicionário.
            return {str(item_id): "" for item_id in data}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}
    except Exception:
        return {}

def save_manifest(path: str, id_to_hash: Dict[str, str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(id_to_hash, ensure_ascii=False, indent=2), encoding="utf-8")

def _add_in_batches(store, texts, metas, ids, batch_size=500):
    for i in range(0, len(ids), batch_size):
        store.add_texts(
            texts=texts[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size],
        )

def upsert_docs_idempotente(
    store: VectorSearchVectorStore,
    docs: List[Document],
    empresa_id: int,
    mode: str = "merge",
    manifest_path: Optional[str] = None,
) -> None:
    """
    - 'merge':
        * ID novo                      -> insere
        * ID igual + hash igual        -> pula (não faz nada)
        * ID igual + hash diferente    -> atualiza (upsert)
    - 'replace':
        * deleta IDs conhecidos da empresa e reinsere
    """
    if not docs:
        print("⚠️  Nenhum documento para ingerir.")
        return

    manifest_default = f".ingest_manifest/empresa_{empresa_id}.json"
    manifest_path = manifest_path or manifest_default

    known_map: Dict[str, str] = load_manifest(manifest_path)

    # 1) Normaliza, gera IDs e fingerprints
    texts_all, metas_all, ids_all, hashes_all = [], [], [], []
    for d in docs:
        md = {k: str(v) for k, v in (d.metadata or {}).items()}
        d.metadata = md
        doc_id = make_doc_id(d)
        doc_hash = doc_fingerprint(d.page_content, md)
        texts_all.append(d.page_content)
        metas_all.append(md)
        ids_all.append(doc_id)
        hashes_all.append(doc_hash)

    # 2) Colapsa duplicatas por ID no mesmo lote (último vence)
    by_id: Dict[str, Tuple[str, Dict[str, str], str]] = {}
    dup_count = 0
    for t, m, i, h in zip(texts_all, metas_all, ids_all, hashes_all):
        if i in by_id:
            dup_count += 1
        by_id[i] = (t, m, h)

    if dup_count:
        print(f"⚠️  {dup_count} ocorrências duplicadas no lote foram colapsadas por ID (último vence).")

    ids_unique = list(by_id.keys())
    texts_unique = [by_id[i][0] for i in ids_unique]
    metas_unique = [by_id[i][1] for i in ids_unique]
    hashes_unique = [by_id[i][2] for i in ids_unique]

    # 3) REPLACE: remover IDs existentes da empresa
    to_delete: List[str] = []
    if mode == "replace":
        to_delete = [i for i in list(known_map.keys()) if i.startswith(f"{empresa_id}:")]
        if to_delete and not DRY_RUN:
            try:
                store.delete(ids=to_delete)
                print(f"🧹 Removidos {len(to_delete)} IDs (empresa={empresa_id}).")
            except Exception as e:
                print(f"⚠️  Falha ao deletar {len(to_delete)} IDs (seguindo mesmo assim): {e}")
        # remove do manifesto
        for i in to_delete:
            known_map.pop(i, None)

    # 4) MERGE: decidir NEW / UNCHANGED / CHANGED
    new_texts, new_metas, new_ids, new_hashes = [], [], [], []
    upd_texts, upd_metas, upd_ids, upd_hashes = [], [], [], []
    skipped_unchanged = 0

    for t, m, i, h in zip(texts_unique, metas_unique, ids_unique, hashes_unique):
        old_hash = known_map.get(i)
        if old_hash is None:
            new_texts.append(t); new_metas.append(m); new_ids.append(i); new_hashes.append(h)
        else:
            if old_hash == h:
                skipped_unchanged += 1
            else:
                upd_texts.append(t); upd_metas.append(m); upd_ids.append(i); upd_hashes.append(h)

    # 5) Executar upserts em lotes
    total_inserted = 0
    total_updated = 0

    if new_ids:
        if DRY_RUN:
            print(f"[DRY_RUN] Inserções pendentes: {len(new_ids)} (não enviado ao índice).")
        else:
            _add_in_batches(store, new_texts, new_metas, new_ids, batch_size=INGEST_BATCH)
            total_inserted = len(new_ids)
            for i, h in zip(new_ids, new_hashes):
                known_map[i] = h

    if upd_ids:
        if DRY_RUN:
            print(f"[DRY_RUN] Atualizações pendentes: {len(upd_ids)} (não enviado ao índice).")
        else:
            _add_in_batches(store, upd_texts, upd_metas, upd_ids, batch_size=INGEST_BATCH)
            total_updated = len(upd_ids)
            for i, h in zip(upd_ids, upd_hashes):
                known_map[i] = h

    print(
        f"✔ Ingestão concluída (mode={mode}). "
        f"Novos: {total_inserted} | Atualizados: {total_updated} | Iguais (pulado): {skipped_unchanged}"
    )

    # 6) Salvar manifesto atualizado
    save_manifest(manifest_path, known_map)
    print(f"🗂  Manifesto atualizado em: {manifest_path} (IDs rastreados: {len(known_map)})")

# ========= 5. CLI ========= #
def main():
    parser = argparse.ArgumentParser(description="Ingestão (ETL) para Vertex AI Vector Search (idempotente)")
    parser.add_argument("--file", required=True, help="Caminho do arquivo Excel (.xlsx)")
    parser.add_argument("--empresa", required=True, type=int, help="empresa_id (inteiro)")
    parser.add_argument("--mode", choices=["merge", "replace"], default="merge", help="merge (default) ou replace")
    parser.add_argument("--manifest", default=None, help="Caminho do manifesto (opcional)")
    parser.add_argument("--index-name", default=INDEX_DISPLAY_NAME, help="Display name do Index")
    parser.add_argument("--endpoint-name", default=ENDPOINT_DISPLAY_NAME, help="Display name do Endpoint")
    parser.add_argument(
        "--index-mode",
        choices=["latest", "new"],
        default="latest",
        help='Escolha "latest" (reusar último índice) ou "new" (criar um novo). Default: latest.',
    )
    parser.add_argument("--all-empresas", action="store_true",
                        help="(Somente loader linha-por-valor) Ingere TODAS as empresas encontradas no arquivo. "
                             "Por padrão, ingere apenas a empresa passada em --empresa.")
    args = parser.parse_args()

    file_path: str = args.file
    empresa_id: int = args.empresa
    mode: str = args.mode
    manifest_path: Optional[str] = args.manifest
    index_name: str = args.index_name
    endpoint_name: str = args.endpoint_name
    index_mode: str = args.index_mode
    all_empresas: bool = args.all_empresas

    # Sanity check de credenciais e env
    try:
        creds, adc_project = ga_default()
        print(f"[DEBUG] ADC project: {adc_project}")
    except Exception as e:
        print(f"[DEBUG] ADC indisponível: {e}")

    print(f"[DEBUG] GOOGLE_CLOUD_PROJECT={PROJECT_ID}")
    print(f"[DEBUG] GOOGLE_CLOUD_LOCATION={REGION}")
    print(f"[DEBUG] VERTEX_STAGING_BUCKET={BUCKET}")

    if not PROJECT_ID or not BUCKET:
        raise RuntimeError("Defina GOOGLE_CLOUD_PROJECT e VERTEX_STAGING_BUCKET no ambiente ou .env.")

    print("✔ Garantindo Index/Endpoint...")
    index_id, endpoint_id = ensure_vertex_index_and_endpoint(
        project_id=PROJECT_ID,
        region=REGION,
        bucket=BUCKET,
        index_display_name=index_name,
        endpoint_display_name=endpoint_name,
        dimensions=VECTOR_DIMENSIONS,
        distance="DOT_PRODUCT_DISTANCE",
        index_mode=index_mode,  # <-- usa o modo escolhido no CLI
    )
    print(f"   Index ID: {index_id}")
    print(f"   Endpoint ID: {endpoint_id}\n")

    print("✔ Abrindo Vector Store...")
    store = open_vector_vertexai(bucket=BUCKET, index_id=index_id, endpoint_id=endpoint_id)

    print("\n✔ Carregando documentos do Excel...")
    kind = choose_loader(file_path)
    if kind == "line":
        # Por padrão, filtramos pela empresa informada para evitar misturar empresas.
        only_emp = None if all_empresas else empresa_id
        docs = line_per_value_loader(file_path, only_empresa=only_emp)
    else:
        docs = excel_table_loader(file_path, empresa_id)

    # NÃO sobrescrever empresa_id: apenas garantir que exista.
    for d in docs:
        d.metadata = d.metadata or {}
        if "empresa_id" not in d.metadata or not str(d.metadata["empresa_id"]).strip():
            d.metadata["empresa_id"] = str(empresa_id)
        # normaliza tipo, se vier errado
        if "tipo" in d.metadata and d.metadata["tipo"]:
            d.metadata["tipo"] = normalize_tipo(d.metadata["tipo"])

    # Telemetria resumida
    print(f"   Total de documentos lidos: {len(docs)} (mode={mode})")
    counter_sheet_ano = Counter((d.metadata.get("sheet", "na"), d.metadata.get("ano", "all")) for d in docs)
    for (sheet, ano), qtd in sorted(counter_sheet_ano.items()):
        print(f"   - Sheet={sheet} Ano={ano}: {qtd} registros")

    counter_tipo = defaultdict(int)
    for d in docs:
        counter_tipo[d.metadata.get("tipo", "na")] += 1
    if counter_tipo:
        print("   Por tipo:", dict(sorted(counter_tipo.items())))

    print("\n✔ Executando upsert idempotente...")
    upsert_docs_idempotente(store, docs, empresa_id=empresa_id, mode=mode, manifest_path=manifest_path)

    print("\n🎉 Ingestão finalizada.")

if __name__ == "__main__":
    main()

"""
Como usar

1) Criar um índice NOVO (primeira empresa) e salvar no .ingest_state.json:
    python 01_ingest.py --file indicadores_base_2.xlsx --empresa 12 --mode replace --index-mode new

2) Reusar o MESMO índice para outra empresa:
    python 01_ingest.py --file indicadores_base_2.xlsx --empresa 31 --mode merge --index-mode latest

3) (Opcional) Ingerir TODAS as empresas do arquivo “linha por valor”:
    python 01_ingest.py --file base_linha_por_valor.xlsx --empresa 0 --mode merge --all-empresas --index-mode latest

Dicas:
- Evite definir VERTEX_FORCE_NEW_INDEX=1 quando quiser reaproveitar o índice.
- Para "resetar" manualmente, apague o .ingest_state.json e use --index-mode new na primeira ingestão.
"""
