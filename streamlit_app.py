import io, os, json, hashlib
import pandas as pd
import streamlit as st
from datetime import datetime

def _sha1(b: bytes) -> str:
    import hashlib
    return hashlib.sha1(b).hexdigest()[:10]

def gdrive_upload_bytes(name: str, data: bytes, mime: str):
    # Upload robusto (PyDrive2 non ha SetContentBinary)
    import tempfile, os
    drive, folder_id = _drive_client()
    f = drive.CreateFile({
        "title": name,
        "parents": [{"id": folder_id}],
        "mimeType": mime
    })
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        tmp_name = tmp.name
    try:
        f.SetContentFile(tmp_name)
        f.Upload()
    finally:
        try: os.remove(tmp_name)
        except: pass
    return f["id"], f["title"]

def gdrive_list_folder():
    drive, folder_id = _drive_client()
    q = f"'{folder_id}' in parents and trashed=false"
    return drive.ListFile({"q": q}).GetList()

def gdrive_get_file_content(file_id: str) -> bytes:
    import tempfile
    drive, _ = _drive_client()
    f = drive.CreateFile({"id": file_id})
    with tempfile.NamedTemporaryFile() as tmp:
        f.GetContentFile(tmp.name)
        tmp.seek(0)
        return tmp.read()

# --- Manifest (file JSON nella cartella Drive) ---
_MANIFEST_NAME = "manifest.json"

def _find_manifest():
    items = gdrive_list_folder()
    for it in items:
        if it["title"] == _MANIFEST_NAME:
            return it["id"]
    return None

def _save_manifest(man: dict):
    payload_str = json.dumps(man, ensure_ascii=False, indent=2)
    drive, folder_id = _drive_client()
    mf_id = _find_manifest()
    f = drive.CreateFile({"id": mf_id}) if mf_id else drive.CreateFile({
        "title": _MANIFEST_NAME,
        "parents": [{"id": folder_id}],
        "mimeType": "application/json",
    })
    f.SetContentString(payload_str)
    f.Upload()

def _load_manifest() -> dict:
    mf_id = _find_manifest()
    if not mf_id:
        return {}
    raw = gdrive_get_file_content(mf_id)
    try:
        return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except Exception:
        return {}
def set_b2c_baseline_record(rec: dict):
    man = _load_manifest()
    man["b2c_baseline"] = rec
    _save_manifest(man)

def get_b2c_baseline_record() -> dict | None:
    return _load_manifest().get("b2c_baseline")

def save_uploaded_file_drive(uploaded_file_like, kind: str) -> dict:
    """
    uploaded_file_like: oggetto con .getvalue() e .name
    kind: "movements" | "stock"
    """
    raw = uploaded_file_like.getvalue()
    ext = uploaded_file_like.name.split(".")[-1].lower()
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    h = _sha1(raw)
    title = f"{kind}_{ts}_{h}.{ext}"
    mime = "text/csv" if ext == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    file_id, _ = gdrive_upload_bytes(title, raw, mime)

    man = _load_manifest()
    man[kind] = {
        "file_id": file_id,
        "title": title,
        "uploaded_at_utc": ts,
        "orig_name": uploaded_file_like.name,
        "size_bytes": len(raw),
        "kind": kind,
        "ext": ext
    }
    _save_manifest(man)
    return man[kind]

def get_last_saved_drive(kind: str) -> dict | None:
    return _load_manifest().get(kind)



# --- Google Drive via Service Account (PyDrive2) ---
def _drive_client():
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    import json

    folder_id = (st.secrets.get("GDRIVE_FOLDER_ID") or os.getenv("GDRIVE_FOLDER_ID"))
    if not folder_id:
        raise RuntimeError("GDRIVE_FOLDER_ID non trovato in st.secrets o env")

    subject = (st.secrets.get("IMPERSONATE_EMAIL") or os.getenv("IMPERSONATE_EMAIL"))

    # --- carica credenziali SA ---
    sa_dict = None
    if "google_sa" in st.secrets:
        sa_dict = dict(st.secrets["google_sa"])          # già dict
    else:
        sa_json = st.secrets.get("GDRIVE_SA_JSON") or os.getenv("GDRIVE_SA_JSON")
        if not sa_json:
            raise RuntimeError("Credenziali SA non trovate: definisci [google_sa] o GDRIVE_SA_JSON")
        if isinstance(sa_json, str):
            # è già stringa JSON → usala così com'è
            client_json_str = sa_json
        elif isinstance(sa_json, dict):
            # è dict → DUMPS a stringa JSON
            client_json_str = json.dumps(sa_json)
        else:
            raise RuntimeError("GDRIVE_SA_JSON deve essere stringa JSON o dict")

    # se avevi [google_sa], serializza a stringa JSON qui
    if sa_dict is not None:
        client_json_str = json.dumps(sa_dict)

    settings = {
        "client_config_backend": "service",
        "service_config": {
            # 👉 PyDrive2 vuole una STRINGA JSON
            "client_json": client_json_str
        },
        "oauth_scope": ["https://www.googleapis.com/auth/drive"]
    }
    if subject:
        settings["service_config"]["client_user_email"] = subject  # impersonation

    gauth = GoogleAuth(settings=settings)
    gauth.ServiceAuth()
    drive = GoogleDrive(gauth)

        # 👇 Diagnostica: chi sono?
    try:
        about = drive.GetAbout()
        st.info(f"🔐 Autenticato come: {about.get('user', {}).get('emailAddress', 'sconosciuto')}")
    except Exception as e:
        st.warning(f"Impossibile leggere About: {e}")

    return drive, folder_id


def _sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:10]

def gdrive_upload_bytes(name: str, data: bytes, mime: str):
    import tempfile
    drive, folder_id = _drive_client()
    f = drive.CreateFile({
        "title": name,
        "parents": [{"id": folder_id}],
        "mimeType": mime
    })
    # PyDrive2 NON ha SetContentBinary: usa un file temporaneo
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        tmp_name = tmp.name
    try:
        f.SetContentFile(tmp_name)
        f.Upload()
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass
    return f["id"], f["title"]

def gdrive_list_folder():
    drive, folder_id = _drive_client()
    q = f"'{folder_id}' in parents and trashed=false"
    return drive.ListFile({"q": q}).GetList()

def gdrive_get_file_content(file_id: str) -> bytes:
    import tempfile
    drive, _ = _drive_client()
    f = drive.CreateFile({"id": file_id})
    with tempfile.NamedTemporaryFile() as tmp:
        f.GetContentFile(tmp.name)  # scarica su file
        tmp.seek(0)
        return tmp.read()

# --- Manifest (file JSON nella cartella Drive) ---
_MANIFEST_NAME = "manifest.json"

def _find_manifest():
    items = gdrive_list_folder()
    for it in items:
        if it["title"] == _MANIFEST_NAME:
            return it["id"]
    return None

def _save_manifest(man: dict):
    # salva manifest.json come TESTO (SetContentString)
    payload_str = json.dumps(man, ensure_ascii=False, indent=2)

    drive, folder_id = _drive_client()
    mf_id = _find_manifest()

    if mf_id:
        f = drive.CreateFile({"id": mf_id})
    else:
        f = drive.CreateFile({
            "title": _MANIFEST_NAME,
            "parents": [{"id": folder_id}],
            "mimeType": "application/json",
        })
    f.SetContentString(payload_str)
    f.Upload()



def _load_manifest() -> dict:
    mf_id = _find_manifest()
    if not mf_id:
        return {}
    raw = gdrive_get_file_content(mf_id)
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    except Exception:
        return {}

def save_uploaded_file_drive(uploaded_file, kind: str) -> dict:
    """
    kind: "movements" | "stock"
    Ritorna dict con metadata {file_id, title, uploaded_at_utc, orig_name, size_bytes, kind}
    """
    raw = uploaded_file.getvalue()
    ext = uploaded_file.name.split(".")[-1].lower()
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    h = _sha1(raw)
    title = f"{kind}_{ts}_{h}.{ext}"

    mime = "text/csv" if ext == "csv" else \
           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    file_id, _ = gdrive_upload_bytes(title, raw, mime)

    man = _load_manifest()
    rec = {
        "file_id": file_id,
        "title": title,
        "uploaded_at_utc": ts,
        "orig_name": uploaded_file.name,
        "size_bytes": len(raw),
        "kind": kind,
        "ext": ext
    }
    man[kind] = rec
    _save_manifest(man)
    return rec

def get_last_saved_drive(kind: str) -> dict | None:
    man = _load_manifest()
    return man.get(kind)




import sys
print("USING PYTHON:", sys.executable)

import os
import math
import calendar
import warnings
from datetime import datetime, timedelta

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode, DataReturnMode

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()  # carica .env se presente

# -------------------------------
# Configurazione pagina
# -------------------------------
st.set_page_config(
    page_title="Small Giants - Inventory Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# AUTH
# -------------------------------
def gate():
    import os
    import streamlit as st

    expected_user = os.getenv("APP_USERNAME", "SG")
    expected_pwd  = os.getenv("APP_PASSWORD", "change-me")

    st.markdown("## 🔐 Login")

    with st.form("login_form", clear_on_submit=False):
        user = st.text_input("Username", key="login_user")
        pwd  = st.text_input("Password", type="password", key="login_pwd")
        submit = st.form_submit_button("Entra")

    if submit:
        if user == expected_user and pwd == expected_pwd:
            st.session_state["_auth_ok"] = True
        else:
            st.error("Credenziali non valide")

    if not st.session_state.get("_auth_ok", False):
        st.stop()  # blocca il resto dell'app finché non logghi

# -------------------------------
# CSS
# -------------------------------
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    word-break: break-word;
    overflow-wrap: anywhere;
    white-space: normal;
}
.metric-container {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.status-critical { color: #ff4444; font-weight: bold; }
.status-warning { color: #ffaa00; font-weight: bold; }
.status-good { color: #00aa44; font-weight: bold; }
.small-note { font-size: 0.85rem; color: #666; }
.ag-theme-balham .ag-cell-wrap-text {white-space: normal !important;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Traduzioni
# -------------------------------
def load_translations():
    return {
        'it': {
            'title': 'Small Giants - Previsione Inventario',
            'upload_file': 'Carica File Excel/CSV',
            'forecast_days': 'Giorni di Previsione',
            'safety_stock': 'Scorta di Sicurezza (giorni)',
            'lead_time': 'Tempo di Consegna (giorni)',
            'select_product': 'Seleziona Prodotto',
            'language': 'Lingua',
            'historical_data': 'Dati Storici',
            'forecast': 'Previsione',
            'inventory_status': 'Stato Inventario',
            'order_recommendations': 'Raccomandazioni Ordini',
            'critical': 'Critico',
            'warning': 'Attenzione',
            'good': 'Buono',
            'units_to_order': 'Unità da Ordinare',
            'current_stock': 'Stock Attuale',
            'predicted_demand': 'Domanda Prevista',
            'loading': 'Caricamento in corso...',
            'no_data': 'Nessun dato disponibile per questo prodotto',
            'file_uploaded': 'File caricato con successo!',
            'processing': 'Processando i dati...'
        },
        'en': {
            'title': 'Small Giants - Inventory Forecast',
            'upload_file': 'Upload Excel/CSV File',
            'forecast_days': 'Forecast Days',
            'safety_stock': 'Safety Stock (days)',
            'lead_time': 'Lead Time (days)',
            'select_product': 'Select Product',
            'language': 'Language',
            'historical_data': 'Historical Data',
            'forecast': 'Forecast',
            'inventory_status': 'Inventory Status',
            'order_recommendations': 'Order Recommendations',
            'critical': 'Critical',
            'warning': 'Warning',
            'good': 'Good',
            'units_to_order': 'Units to Order',
            'current_stock': 'Current Stock',
            'predicted_demand': 'Predicted Demand',
            'loading': 'Loading...',
            'no_data': 'No data available for this product',
            'file_uploaded': 'File uploaded successfully!',
            'processing': 'Processing data...'
        }
    }

# -------------------------------
# Utils
# -------------------------------
def find_column(df, possible_names):
    cols_norm = {str(c): str(c).lower().strip().replace('\ufeff','') for c in df.columns}
    for col, norm in cols_norm.items():
        for name in possible_names:
            if name.lower() in norm:
                return col
    return None

def process_stock_file(df):
    if df is None or df.empty or len(df.columns) == 0:
        st.error("❌ File stock non valido o vuoto")
        return pd.DataFrame()

    try:
        st.write("🔍 **Colonne stock trovate nel file:**", list(df.columns))
        df.columns = df.columns.astype(str).str.replace('\ufeff','', regex=False).str.strip()

        if 'export' in str(df.columns[0]).lower() or 'unnamed' in str(df.columns[0]).lower():
            st.info("📋 Rilevate intestazioni nella prima riga del file stock. Riprocessando...")
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            df.columns = df.columns.astype(str).str.strip()
            st.write("🔍 **Colonne stock corrette:**", list(df.columns))

        physical_warehouse_col = find_column(df, ['magazzino fisico', 'physical warehouse', 'warehouse physical', 'deposito fisico', 'wms', 'warehouse'])
        if physical_warehouse_col:
            initial_rows = len(df)
            norm = df[physical_warehouse_col].astype(str).str.upper().str.replace(r'\s+', '', regex=True)
            warehouse_values = ['SMALLGIANTSAVAILABLE', 'SMALLGIANTSAVAILABLEWMS', 'SMALLGIANTSAVAILABLE-1']
            physical_match = norm.isin(warehouse_values) | norm.str.contains('SMALLGIANTSAVAILABLE')
            st.write("🔍 Debug filtro magazzino:")
            st.write(f"   • Colonna usata: {physical_warehouse_col}")
            st.write(f"   • Valori unici trovati: {df[physical_warehouse_col].unique()}")
            st.write(f"   • Match trovati: {physical_match.sum()}")
            df = df[physical_match].copy()
            filtered_count = initial_rows - len(df)
            if filtered_count > 0:
                st.info(f"🔍 Esclusi {filtered_count} prodotti non-SMALLGIANTSAVAILABLE (selezionati {len(df)} prodotti)")
            if len(df) == 0:
                st.warning("⚠️ Nessun prodotto con esattamente 'SMALLGIANTSAVAILABLE' trovato nel file stock")
                return pd.DataFrame()
            st.success(f"✅ Filtrati solo prodotti SMALLGIANTSAVAILABLE: {len(df)} prodotti")
        else:
            st.error("❌ Colonna magazzino non trovata. Impossibile filtrare.")
            return pd.DataFrame()

        sku_column = find_column(df, ['sku interno', 'internal sku', 'sku', 'codice', 'code'])
        available_column = find_column(df, ['disponibile', 'available', 'qtà disponibile', 'q.tà disponibile'])
        in_stock_column = find_column(df, ['in stock', 'stock', 'qtà in stock', 'q.tà in stock'])
        product_name_column = find_column(df, ['nome del prodotto', 'product name', 'nome', 'name', 'prodotto'])
        reserved_column = find_column(df, ['prenotate', 'booked', 'reserved', 'qtà prenotate', 'q.tà prenotate'])
        incoming_column = find_column(df, ['attesa di ricezione', 'incoming', 'in attesa', 'ricezione', 'required'])

        if sku_column is None:
            st.error("❌ Colonna SKU non trovata nel file stock")
            return pd.DataFrame()
        if available_column is None and in_stock_column is None:
            st.error("❌ Nessuna colonna quantità trovata nel file stock")
            return pd.DataFrame()

        st.success("✅ Colonne stock mappate:")
        st.success(f"   • SKU: '{sku_column}'")
        if available_column: st.success(f"   • Disponibile: '{available_column}'")
        if in_stock_column:  st.success(f"   • In Stock: '{in_stock_column}'")
        if product_name_column: st.success(f"   • Nome Prodotto: '{product_name_column}'")
        if reserved_column: st.success(f"   • Prenotate: '{reserved_column}'")
        if incoming_column: st.success(f"   • In Arrivo: '{incoming_column}'")

        column_mapping = {sku_column: 'sku'}
        if available_column:     column_mapping[available_column] = 'qty_available'
        if in_stock_column:      column_mapping[in_stock_column] = 'qty_in_stock'
        if product_name_column:  column_mapping[product_name_column] = 'product_name'
        if reserved_column:      column_mapping[reserved_column] = 'qty_reserved'
        if incoming_column:      column_mapping[incoming_column] = 'qty_incoming'
        df = df.rename(columns=column_mapping)

        for col in ['qty_available','qty_in_stock','qty_reserved','qty_incoming']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Calcolo stock effettivo evitando doppia sottrazione dei prenotati
        available_is_net = False
        if 'qty_available' in df.columns:
            df['current_stock'] = df['qty_available']  # già netto
            available_is_net = True
        elif 'qty_in_stock' in df.columns:
            df['current_stock'] = df['qty_in_stock']   # lordo
        else:
            df['current_stock'] = 0

        if 'qty_reserved' in df.columns and not available_is_net:
            df['current_stock'] = df['current_stock'] - df['qty_reserved']

        if 'qty_incoming' in df.columns:
            df['current_stock'] = df['current_stock'] + df['qty_incoming']

        df['current_stock'] = df['current_stock'].clip(lower=0)

        df = df.dropna(subset=['sku'])
        df['sku'] = df['sku'].astype(str).str.strip()

        st.info("🔄 Rimuovendo prodotti duplicati (mantenendo prima occorrenza)...")
        df_before = len(df)
        df = df.drop_duplicates(subset=['sku'], keep='first')
        df_after = len(df)
        if df_before > df_after:
            st.success(f"✅ Rimossi {df_before - df_after} duplicati (da {df_before} a {df_after} righe)")

        st.success(f"✅ Stock processato: {len(df)} prodotti")
        keep_cols = ['sku','current_stock','product_name'] + [c for c in df.columns if c.startswith('qty_')]
        keep_cols = [c for c in keep_cols if c in df.columns]
        return df[keep_cols]

    except Exception as e:
        st.error(f"Errore nel processamento del file stock: {str(e)}")
        return pd.DataFrame()

def process_excel_data(df):
    """
    Ritorna result_df con colonne:
    - date, sku, product_name, units_sold, units_sold_b2b, units_sold_b2c, on_hand_end
    """
    try:
        st.write("🔍 **Colonne trovate nel file:**", list(df.columns))
        df.columns = df.columns.astype(str).str.strip()

        if 'export' in str(df.columns[0]).lower() or 'unnamed' in str(df.columns[0]).lower():
            st.info("📋 Rilevate intestazioni nella prima riga. Riprocessando...")
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            df.columns = df.columns.astype(str).str.strip()
            st.write("🔍 **Colonne corrette:**", list(df.columns))

        date_column = find_column(df, ['date', 'data', 'datetime', 'time', 'giorno'])
        movement_column = find_column(df, ['type', 'mov', 'movement', 'tipo'])
        sku_column = find_column(df, ['internal sku', 'sku', 'codice', 'code', 'item'])
        quantity_column = find_column(df, ['quantity', 'quantit', 'qty', 'pezzi', 'qta'])
        name_column = find_column(df, ['name', 'nome', 'descrizione', 'description', 'product'])
        order_column = find_column(df, ['order number', 'order', 'ordine', 'order id', 'order_id', 'id ordine'])

        missing = []
        if date_column is None:      missing.append("data/date")
        if movement_column is None:  missing.append("tipo/type")
        if sku_column is None:       missing.append("SKU/codice")
        if quantity_column is None:  missing.append("quantità/quantity")
        if missing:
            st.error(f"❌ Colonne mancanti: {', '.join(missing)}")
            st.error(f"Colonne disponibili: {list(df.columns)}")
            return pd.DataFrame()

        st.success("✅ Colonne mappate:")
        st.success(f"   • Data: '{date_column}'")
        st.success(f"   • Movimento: '{movement_column}'")
        st.success(f"   • SKU: '{sku_column}'")
        st.success(f"   • Quantità: '{quantity_column}'")
        if name_column:  st.success(f"   • Nome: '{name_column}'")

        column_mapping = {
            date_column: 'Date',
            movement_column: 'Type',
            sku_column: 'Internal_SKU',
            quantity_column: 'Quantity'
        }
        if name_column:   column_mapping[name_column] = 'Name'
        if order_column:  column_mapping[order_column] = 'Order_ID'
        df = df.rename(columns=column_mapping)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['Date'])

        # Channel B2B/B2C
        channel_available = False
        if 'Order_ID' in df.columns:
            df['Order_ID'] = df['Order_ID'].astype(str)
            df['Channel'] = df['Order_ID'].apply(lambda x: 'B2C' if str(x).strip().isdigit() else 'B2B')
            channel_available = True
            st.write("📊 Distribuzione ordini B2B/B2C:")
            ch = df['Channel'].value_counts()
            st.write(f"   • B2B: {ch.get('B2B', 0)} ordini")
            st.write(f"   • B2C: {ch.get('B2C', 0)} ordini")

        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df = df.dropna(subset=['Quantity'])
        df['Type'] = df['Type'].astype(str).str.lower().str.strip()

        st.write("🔍 **Tipi di movimento trovati:**", df['Type'].unique())

        subtype_column = find_column(df, ['subtype', 'sub type', 'sub-type', 'sottotipo'])
        warehouse_column = find_column(df, ['warehouse', 'magazzino', 'deposito', 'wms'])

        st.write("🔍 Colonne trovate per filtri addizionali:")
        if subtype_column:   st.write(f"   • SubType: '{subtype_column}'")
        if warehouse_column: st.write(f"   • Warehouse: '{warehouse_column}'")

        sales_condition = df['Type'].str.lower().str.strip() == 'out'
        if subtype_column:
            sales_condition &= (df[subtype_column].astype(str).str.lower().str.strip() != 'inventory')
        if warehouse_column:
            sales_condition &= (df[warehouse_column].astype(str).str.strip() == 'SMALLGIANTSAVAILABLE')

        total_rows = len(df)
        sales_data = df[sales_condition]
        st.write("📊 Filtri applicati:")
        st.write(f"   • Righe totali: {total_rows}")
        st.write(f"   • Righe dopo filtro Type='out': {len(df[df['Type'].str.lower().str.strip() == 'out'])}")
        if subtype_column:
            st.write(f"   • Righe dopo esclusione inventory: {len(df[(df['Type'].str.lower().str.strip() == 'out') & (df[subtype_column].astype(str).str.lower().str.strip() != 'inventory')])}")
        if warehouse_column:
            st.write(f"   • Righe finali dopo filtro SMALLGIANTSAVAILABLE: {len(sales_data)}")

        st.write(f"📊 **Dati di vendita trovati:** {len(sales_data)} righe")
        if len(sales_data) == 0:
            st.warning("⚠️ Nessun movimento di vendita trovato. Verifica i valori nella colonna Type.")
            sales_data = df.copy()

        # Costruzione timeline
        stock_data = []
        for sku in df['Internal_SKU'].dropna().unique():
            sku_data = df[df['Internal_SKU'] == sku].copy().sort_values('Date')
            stock_balance = 0
            for _, row in sku_data.iterrows():
                movement_type = str(row['Type']).lower()
                quantity = row['Quantity']
                if movement_type == 'out':
                    stock_balance -= quantity
                    units_sold = quantity
                    # split canale: B2C se Order_ID è tutto numerico; B2B = Totale - B2C
                    if channel_available:
                        units_b2c = quantity if str(row.get('Channel')) == 'B2C' else 0
                    else:
                        units_b2c = 0
                    units_b2b = max(0, quantity - units_b2c)
                else:
                    stock_balance += quantity
                    units_sold = 0
                    units_b2b = 0
                    units_b2c = 0

                product_name = row.get('Name', f'Prodotto {sku}')
                stock_data.append({
                    'date': row['Date'],
                    'sku': str(sku),
                    'product_name': product_name,
                    'units_sold': units_sold,
                    'units_sold_b2b': units_b2b,
                    'units_sold_b2c': units_b2c,
                    'on_hand_end': max(0, stock_balance)
                })

        result_df = pd.DataFrame(stock_data)
        if result_df.empty:
            st.error("❌ Nessun dato processabile trovato")
            return pd.DataFrame()

        # Aggrega per giorno/SKU
        agg_cols = {
            'units_sold': 'sum',
            'units_sold_b2b': 'sum',
            'units_sold_b2c': 'sum',
            'on_hand_end': 'last'
        }
        result_df = result_df.groupby(['date','sku','product_name']).agg(agg_cols).reset_index()
        st.success(f"✅ Dati processati: {len(result_df)} record per {result_df['sku'].nunique()} prodotti")
        return result_df

    except Exception as e:
        st.error(f"Errore nel processamento dei dati: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()

def _load_b2c_baseline_from_drive() -> pd.DataFrame:
    """
    Legge la baseline B2C (CSV Shopify) da Drive, la normalizza e ritorna
    colonne: date, sku, product_name, units_sold
    """
    import io
    man = _load_manifest()
    rec = man.get("b2c_baseline")
    if not rec or not rec.get("file_id"):
        return pd.DataFrame()

    raw = gdrive_get_file_content(rec["file_id"])
    bio = io.BytesIO(raw)

    # Leggi tutto come stringhe per non perdere formati
    df = pd.read_csv(bio, dtype=str, keep_default_na=False)

    # Helper per trovare colonne, prima match esatto poi substring
    def _pick(df, names_exact, names_sub):
        cols = list(df.columns)
        norm = {c: str(c).strip() for c in cols}
        # esatto
        for target in names_exact:
            for c in cols:
                if norm[c].lower() == target.lower():
                    return c
        # substring
        for target in names_sub:
            for c in cols:
                if target.lower() in norm[c].lower():
                    return c
        return None

    date_col = _pick(
        df,
        names_exact=["Created at", "Processed at", "Paid at", "Order Date"],
        names_sub=["created", "processed", "paid", "date"]
    )
    sku_col = _pick(
        df,
        names_exact=["Lineitem sku", "Variant SKU", "SKU"],
        names_sub=["sku"]
    )
    qty_col = _pick(
        df,
        names_exact=["Lineitem quantity", "Quantity"],
        names_sub=["quantity", "qty"]
    )
    name_col = _pick(
        df,
        names_exact=["Lineitem name", "Product Title", "Title", "Name"],
        names_sub=["name", "title"]
    )

    if not date_col or not sku_col or not qty_col:
        # colonne minime non trovate → ritorna vuoto
        return pd.DataFrame()

    out = df[[date_col, sku_col, qty_col] + ([name_col] if name_col else [])].copy()

    # Parse datetime ROBUSTO
    out["date"] = pd.to_datetime(out[date_col], errors="coerce", utc=False)
    # Nota: se Shopify esporta in ISO con timezone, puoi forzare:
    # out["date"] = pd.to_datetime(out[date_col], errors="coerce", utc=True).dt.tz_convert(None)

    out = out.dropna(subset=["date"])
    out["sku"] = out[sku_col].astype(str).str.strip()
    out["product_name"] = out[name_col].astype(str).str.strip() if name_col else out["sku"]

    # Quantità
    q = pd.to_numeric(out[qty_col].str.replace(",", ".", regex=False), errors="coerce").fillna(0)
    out["units_sold"] = q.astype(int)

    # Solo righe utili (sku non vuoti e qty>0)
    out = out[(out["sku"] != "") & (out["units_sold"] > 0)].copy()

    # Aggrega per giorno/SKU (se nel CSV ci sono più righe al giorno)
    out = (
        out.groupby([out["date"].dt.normalize(), "sku", "product_name"], as_index=False)["units_sold"]
           .sum()
           .rename(columns={"date": "date"})
           .sort_values("date")
           .reset_index(drop=True)
    )

    # Garantisce dtype datetime64[ns]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out[["date", "sku", "product_name", "units_sold"]]

@st.cache_data
def _parse_movements_from_bytes(data: bytes, name: str) -> pd.DataFrame:
    bio = io.BytesIO(data); bio.name = name
    if name.lower().endswith('.csv'):
        try:
            df = pd.read_csv(bio, encoding='utf-8-sig', sep=',')
            if len(df.columns) == 1:
                bio.seek(0); df = pd.read_csv(bio, encoding='utf-8-sig', sep=';')
        except:
            bio.seek(0); df = pd.read_csv(bio, encoding='latin-1', sep=',')
    else:
        df = pd.read_excel(bio)
    return process_excel_data(df)

@st.cache_data
def _parse_stock_from_bytes(data: bytes, name: str) -> pd.DataFrame:
    bio = io.BytesIO(data); bio.name = name
    if name.lower().endswith('.csv'):
        try:
            df = pd.read_csv(bio, encoding='utf-8-sig', sep=',')
            if len(df.columns) == 1:
                bio.seek(0); df = pd.read_csv(bio, encoding='utf-8-sig', sep=';')
        except:
            bio.seek(0); df = pd.read_csv(bio, encoding='latin-1', sep=',')
    else:
        df = pd.read_excel(bio)
    df.columns = [str(c).strip().replace('\ufeff','') for c in df.columns]
    return process_stock_file(df)

def simple_forecast(ts_data, periods):
    if len(ts_data) < 2:
        avg_value = ts_data.mean() if len(ts_data) > 0 else 1
        forecast_data = []
        last_date = ts_data.index[-1] if len(ts_data) > 0 else datetime.now()
        for i in range(1, periods + 1):
            forecast_date = last_date + timedelta(days=i)
            forecast_data.append({
                'ds': forecast_date,
                'yhat': avg_value,
                'yhat_lower': max(0, avg_value * 0.7),
                'yhat_upper': avg_value * 1.3
            })
        return pd.DataFrame(forecast_data)

    x = np.arange(len(ts_data))
    y = ts_data.values
    z = np.polyfit(x, y, 1)
    trend_slope = z[0]

    if len(y) >= 7:
        y_smooth = pd.Series(y).rolling(window=7, min_periods=1).mean().values
    else:
        y_smooth = y
    last_value = y_smooth[-1]

    weekly_pattern = np.ones(7)
    if len(ts_data) >= 14:
        for day in range(7):
            day_values = [y[i] for i in range(len(y)) if i % 7 == day]
            if day_values:
                weekly_pattern[day] = np.mean(day_values) / np.mean(y)

    forecast_data = []
    last_date = ts_data.index[-1]
    for i in range(1, periods + 1):
        forecast_date = last_date + timedelta(days=i)
        base_value = last_value + trend_slope * i
        day_of_week = (forecast_date.weekday()) % 7
        seasonal_factor = weekly_pattern[day_of_week]
        forecast_value = max(0, base_value * seasonal_factor)
        std_dev = np.std(y) if len(y) > 1 else forecast_value * 0.2
        forecast_data.append({
            'ds': forecast_date,
            'yhat': forecast_value,
            'yhat_lower': max(0, forecast_value - 1.96 * std_dev),
            'yhat_upper': forecast_value + 1.96 * std_dev
        })
    return pd.DataFrame(forecast_data)

# --- OOS, pesi e stagionalità: RESAMPLE GIORNALIERO ---
def mark_out_of_stock_days(daily_series, min_run_days=14):
    s = daily_series.copy().sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s.resample('D').sum().fillna(0)

    zero = (s == 0)
    run_id = (zero != zero.shift()).cumsum()
    run_length = zero.groupby(run_id).transform('sum')
    oos = zero & (run_length >= min_run_days)
    oos.index = s.index
    return oos

def calculate_monthly_average_excluding_oos(daily_series, min_run_days=14, min_weight=0.2):
    try:
        if not isinstance(daily_series.index, pd.DatetimeIndex):
            daily_series.index = pd.to_datetime(daily_series.index)
        s = daily_series.copy().sort_index().resample('D').sum().fillna(0)

        today = datetime.now()
        start_of_current_month = datetime(today.year, today.month, 1)
        s = s[s.index < start_of_current_month]

        if s.empty:
            return {'monthly_avg':0.0,'monthly_avg_simple':0.0,'total_sales':0.0,
                    'valid_months':0,'total_months':0,'oos_months':0,
                    'weighted_months':0.0,'first_sale_date':None,'last_sale_date':None,
                    'monthly_breakdown':[]}

        oos_days = mark_out_of_stock_days(s, min_run_days=min_run_days)

        rows = []
        for m in s.index.to_period('M').unique():
            idx = (s.index.to_period('M') == m)
            total = int(idx.sum())
            oos_cnt = int(oos_days[idx].sum())
            valid = total - oos_cnt
            sales_non_oos = float(s[idx][~oos_days[idx]].sum())
            proj = (sales_non_oos/valid*30.0) if valid>0 else 0.0
            if oos_cnt >= total:
                w = 0.0
            elif oos_cnt == 0:
                w = 1.0
            else:
                w = max(min_weight, 1.0 - oos_cnt/total)
            rows.append({'year': m.year, 'month_num': m.month, 'month': m,
                         'sales': proj, 'actual_sales': sales_non_oos,
                         'oos_days': oos_cnt, 'valid_days': valid,
                         'total_days': total, 'weight': w})

        dfm = pd.DataFrame(rows)
        total_months = len(dfm)
        oos_months = int((dfm['valid_days']==0).sum())
        valid_df = dfm[dfm['valid_days']>0].copy()

        if valid_df.empty:
            by_month = s.groupby(s.index.to_period('M')).sum()
            monthly_avg_simple = float((by_month[by_month>0]).mean()) if (by_month>0).any() else 0.0
            return {'monthly_avg': monthly_avg_simple, 'monthly_avg_simple': monthly_avg_simple,
                    'total_sales': float(s.sum()), 'valid_months': 0, 'total_months': total_months,
                    'oos_months': oos_months, 'weighted_months': 0.0,
                    'first_sale_date': s[s>0].index.min() if (s>0).any() else None,
                    'last_sale_date': s[s>0].index.max() if (s>0).any() else None,
                    'monthly_breakdown': dfm.to_dict('records')}

        monthly_avg = float(valid_df['sales'].mean())
        tot_sales_valid = float(valid_df['actual_sales'].sum())
        weighted_months = float(valid_df['weight'].sum())

        first_sale = s[s>0].index.min() if (s>0).any() else s.index.min()
        last_sale = s[s>0].index.max() if (s>0).any() else s.index.max()

        return {'monthly_avg': monthly_avg, 'monthly_avg_simple': monthly_avg,
                'total_sales': tot_sales_valid, 'valid_months': int(len(valid_df)),
                'total_months': total_months, 'oos_months': oos_months,
                'weighted_months': weighted_months, 'first_sale_date': first_sale,
                'last_sale_date': last_sale, 'monthly_breakdown': dfm.to_dict('records')}
    except Exception:
        return {'monthly_avg':0.0,'monthly_avg_simple':0.0,'total_sales':0.0,
                'valid_months':0,'total_months':0,'oos_months':0,
                'weighted_months':0.0,'first_sale_date':None,'last_sale_date':None,
                'monthly_breakdown':[]}

def compute_oos_month_weights(daily_series, min_run_days=14, min_weight=0.2):
    s = daily_series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s.sort_index().resample('D').sum().fillna(0)
    oos_days_series = mark_out_of_stock_days(s, min_run_days=min_run_days)
    weights = {}
    for m in s.index.to_period('M').unique():
        mask = (s.index.to_period('M') == m)
        total_days = int(mask.sum())
        oos_count = int(oos_days_series[mask].sum())
        if oos_count == 0:         w = 1.0
        elif oos_count >= total_days: w = 0.0
        else:                      w = max(min_weight, 1.0 - (oos_count / total_days))
        weights[m] = float(w)
    return weights

def calculate_monthly_seasonality(all_products_data, single_product_data=None, daily_series_for_weights=None, min_run_days=14, min_weight=0.2):
    def _calc_seasonality(data, weights=None):
        try:
            a = data.copy()
            if 'date' in a.columns:
                a = a.set_index('date')
            a.index = pd.to_datetime(a.index)

            monthly_by_year = a.groupby([a.index.year, a.index.month])['units_sold'].sum().reset_index()
            monthly_by_year.columns = ['year', 'month', 'units_sold']
            if monthly_by_year.empty:
                return {i: 1.0 for i in range(1, 13)}

            monthly_by_year['period'] = monthly_by_year.apply(
                lambda r: pd.Period(year=int(r['year']), month=int(r['month']), freq='M'), axis=1)

            if weights is None:
                if daily_series_for_weights is not None:
                    weights = compute_oos_month_weights(daily_series_for_weights, min_run_days=min_run_days, min_weight=min_weight)
                else:
                    daily_from_data = a['units_sold'].groupby(a.index).sum()
                    weights = compute_oos_month_weights(daily_from_data, min_run_days=min_run_days, min_weight=min_weight)

            monthly_by_year['weight'] = monthly_by_year['period'].map(weights).fillna(1.0)
            monthly_by_year = monthly_by_year[monthly_by_year['weight'] > 0]
            if monthly_by_year.empty:
                return {i: 1.0 for i in range(1, 13)}

            def weighted_avg(group):
                w = group['weight']
                vals = group['units_sold']
                return (vals * w).sum() / w.sum()

            monthly_avg = monthly_by_year.groupby('month').apply(weighted_avg)
            overall_avg = monthly_avg.mean()
            seasonal_factors = (monthly_avg / overall_avg).to_dict()
            for m in range(1, 13):
                if m not in seasonal_factors or not np.isfinite(seasonal_factors[m]):
                    seasonal_factors[m] = 1.0
            return seasonal_factors
        except Exception:
            return {i: 1.0 for i in range(1, 13)}

    global_seasonality = _calc_seasonality(all_products_data)
    product_seasonality = {i: 1.0 for i in range(1, 13)}
    if single_product_data is not None and len(single_product_data) >= 12:
        product_seasonality = _calc_seasonality(single_product_data, weights=compute_oos_month_weights(daily_series_for_weights) if daily_series_for_weights is not None else None)

    final_seasonality = {m: max(global_seasonality.get(m,1.0), product_seasonality.get(m,1.0)) for m in range(1,13)}
    return final_seasonality

def calculate_growth_rate(ts_data, all_products_data=None, months_to_compare=3):
    """
    Calcola il tasso di crescita come media dei growth YoY degli ultimi `months_to_compare`
    MESI COMPLETI (esclude sempre il mese corrente e include l'ultimo completo).
    Se manca il pari mese dell'anno precedente, usa la media dei 6 mesi precedenti a quel mese.
    """
    def _prep_monthly(series):
        s = series.copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        # escludi SEMPRE il mese corrente
        start_current_month = pd.Timestamp(year=pd.Timestamp.now().year, month=pd.Timestamp.now().month, day=1)
        s = s[s.index < start_current_month]
        # aggrega a mese
        dfm = s.to_frame(name='sales')
        dfm['year'] = dfm.index.year
        dfm['month'] = dfm.index.month
        monthly = dfm.groupby(['year','month'])['sales'].sum().reset_index()
        monthly['ym'] = pd.to_datetime(monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2))
        monthly = monthly.sort_values('ym').reset_index(drop=True)
        return monthly

    def _growth_from_monthly(monthly_df):
        if len(monthly_df) == 0:
            return 0.0

        # prendi gli ultimi N MESI COMPLETI (già escluso il corrente a monte)
        recent = monthly_df.tail(months_to_compare)
        growth_list = []

        for _, row in recent.iterrows():
            y, m, ym, cur_sales = int(row['year']), int(row['month']), row['ym'], float(row['sales'])
            # pari mese anno precedente
            prev = monthly_df[(monthly_df['year'] == y - 1) & (monthly_df['month'] == m)]
            if not prev.empty:
                prev_sales = float(prev['sales'].iloc[0])
                if prev_sales > 0:
                    growth_list.append((cur_sales - prev_sales) / prev_sales)
            else:
                # fallback: media dei 6 mesi precedenti al mese corrente (escluso)
                window_start = ym - pd.DateOffset(months=6)
                mask = (monthly_df['ym'] < ym) & (monthly_df['ym'] >= window_start)
                prev6 = monthly_df[mask]
                if not prev6.empty:
                    avg_prev6 = float(prev6['sales'].mean())
                    if avg_prev6 > 0:
                        growth_list.append((cur_sales - avg_prev6) / avg_prev6)

        if growth_list:
            return float(np.mean(growth_list))
        return 0.0

    # prodotto
    product_monthly = _prep_monthly(ts_data)
    product_growth = _growth_from_monthly(product_monthly)

    # globale
    global_growth = 0.0
    if all_products_data is not None:
        if isinstance(all_products_data, pd.DataFrame):
            # somma vendite di tutti gli SKU per giorno → poi _prep_monthly
            if 'units_sold' in all_products_data.columns:
                daily_sum = all_products_data['units_sold'].groupby(all_products_data.index).sum()
                global_monthly = _prep_monthly(daily_sum)
                global_growth = _growth_from_monthly(global_monthly)

    return max(product_growth, global_growth)

def forecast_with_monthly_seasonality(data, periods, all_products_data=None, current_sku=None):
    if len(data) < 2:
        return simple_forecast(data, periods)
    try:
        data_df = data.reset_index()
        data_df.columns = ['date','sales']
        data_df['month'] = data_df['date'].dt.month
        data_df['year'] = data_df['date'].dt.year
        last_date = data_df['date'].max()

        monthly_sales = data_df.groupby(['year','month'])['sales'].sum().reset_index()
        growth_rate = calculate_growth_rate(data, months_to_compare=3)
        st.info(f"📈 Tasso di crescita rilevato: {growth_rate*100:.1f}%")

        daily_series_for_weights = data.copy().sort_index().resample('D').sum().fillna(0)
        month_weights = compute_oos_month_weights(daily_series_for_weights, min_run_days=14, min_weight=0.2)

        if all_products_data is None:
            all_products_data = data.to_frame(name='units_sold')
        seasonal_factors = calculate_monthly_seasonality(all_products_data,
                                                         daily_series_for_weights=daily_series_for_weights,
                                                         min_run_days=14, min_weight=0.2)

        recent_avg = float(data.tail(30).mean()) if len(data) >= 30 else float(data.mean()) if len(data) > 0 else 0.0

        forecast_data = []
        for i in range(1, periods + 1):
            forecast_date = last_date + timedelta(days=i)
            forecast_month = forecast_date.month
            forecast_year = forecast_date.year
            same_month_last_year = monthly_sales[(monthly_sales['year']==forecast_year-1)&(monthly_sales['month']==forecast_month)]
            period_prev_year = pd.Period(year=forecast_year-1, month=forecast_month, freq='M')

            if not same_month_last_year.empty:
                base_monthly_sales = float(same_month_last_year['sales'].iloc[0])
                adjusted_monthly_sales = base_monthly_sales * (1 + growth_rate)
                days_in_month = calendar.monthrange(forecast_year, forecast_month)[1]
                month_weight = month_weights.get(period_prev_year, 1.0)
                if month_weight == 0.0:
                    seasonal_factor = seasonal_factors.get(forecast_month, 1.0)
                    base_value = recent_avg * seasonal_factor * (1 + growth_rate * 0.5)
                elif month_weight < 1.0:
                    val_from_lastyear = adjusted_monthly_sales / days_in_month
                    val_from_recent = recent_avg * seasonal_factors.get(forecast_month, 1.0) * (1 + growth_rate * 0.5)
                    base_value = month_weight * val_from_lastyear + (1.0 - month_weight) * val_from_recent
                else:
                    base_value = adjusted_monthly_sales / days_in_month
            else:
                seasonal_factor = seasonal_factors.get(forecast_month, 1.0)
                base_value = recent_avg * seasonal_factor * (1 + growth_rate * 0.5)

            forecast_value = max(0, base_value)
            std_dev = data.std() if len(data) > 1 else forecast_value * 0.3
            forecast_data.append({
                'ds': forecast_date,
                'yhat': forecast_value,
                'yhat_lower': max(0, forecast_value - 1.5 * std_dev),
                'yhat_upper': forecast_value + 1.5 * std_dev
            })

        return pd.DataFrame(forecast_data)
    except Exception as e:
        st.warning(f"Errore nel forecasting stagionale: {str(e)}. Uso modello semplificato.")
        return simple_forecast(data, periods)

def get_central_forecast_series(forecast_df):
    if forecast_df is None or forecast_df.empty:
        return pd.Series(dtype=float)
    if 'ds' in forecast_df.columns:
        idx = pd.to_datetime(forecast_df['ds'])
    else:
        idx = forecast_df.index
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        central = (forecast_df['yhat_lower'].astype(float) + forecast_df['yhat_upper'].astype(float)) / 2.0
    elif 'yhat' in forecast_df.columns:
        central = forecast_df['yhat'].astype(float)
    else:
        central = pd.Series(0.0, index=forecast_df.index)
    central = pd.Series(central.values, index=idx)
    return central

def forecast_with_prophet(data, periods, all_products_data=None, current_sku=None):
    return forecast_with_monthly_seasonality(data, periods, all_products_data, current_sku)

def calculate_order_recommendation(forecast_df, current_stock, safety_days, lead_time_days,
                                   qty_incoming=0, qty_reserved=0, safety_margin=0.1, moq=1):
    if forecast_df is None or forecast_df.empty:
        return 0, "good", {}

    total_days = lead_time_days + safety_days
    central_series = get_central_forecast_series(forecast_df)
    forecast_demand_central = central_series.head(total_days).sum()
    forecast_demand = float(forecast_demand_central) * (1 + float(safety_margin))

    effective_stock = current_stock - qty_reserved + qty_incoming
    units_needed = forecast_demand - effective_stock
    units_to_order = max(0, units_needed)
    units_to_order = math.ceil(units_to_order)
    if units_to_order > 0 and units_to_order < moq:
        units_to_order = moq
    if units_to_order > 0 and moq > 1:
        units_to_order = math.ceil(units_to_order / moq) * moq

    if forecast_demand > 0:
        days_of_stock = effective_stock / (forecast_demand / total_days)
    else:
        days_of_stock = float('inf')

    if days_of_stock < lead_time_days:
        status = "critical"
    elif days_of_stock < lead_time_days + safety_days:
        status = "warning"
    else:
        status = "good"

    details = {
        'forecast_demand': forecast_demand,
        'forecast_demand_central': float(central_series.head(total_days).sum()),
        'current_stock': current_stock,
        'qty_reserved': qty_reserved,
        'qty_incoming': qty_incoming,
        'effective_stock': effective_stock,
        'days_of_stock': days_of_stock if days_of_stock != float('inf') else 999
    }
    return int(units_to_order), status, details

def _aggregate_series_for_display(daily_hist_series, forecast_df, freq='W'):
    """
    Aggrega storico + forecast per la UI (settimanale 'W' o mensile 'M').
    """
    # storico
    h = daily_hist_series.copy()
    if not isinstance(h.index, pd.DatetimeIndex):
        h.index = pd.to_datetime(h.index)
    h = h.resample(freq).sum()

    # forecast
    if forecast_df is None or forecast_df.empty:
        f = pd.Series(dtype=float)
    else:
        fc = get_central_forecast_series(forecast_df)
        f = fc.resample(freq).sum()

    return h, f

def create_forecast_chart(historical_data, forecast_data, product_name, freq='W'):
    """
    Grafico UI aggregato per settimana/mese con trend line (storico).
    """
    hist, fc = _aggregate_series_for_display(historical_data, forecast_data, freq=freq)

    fig = go.Figure()
    if len(hist) > 0:
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist.values,
            mode='lines+markers',
            name=f'Vendite Storiche ({ "Settimana" if freq=="W" else "Mese" })',
            line=dict(width=2),
            marker=dict(size=6)
        ))

        # --- Trend line (storico) ---
        try:
            x = np.arange(len(hist))
            y = hist.values.astype(float)
            if len(x) >= 2 and np.isfinite(y).all():
                a, b = np.polyfit(x, y, 1)  # y = a*x + b
                trend_y = a * x + b
                fig.add_trace(go.Scatter(
                    x=hist.index, y=trend_y,
                    mode='lines',
                    name='Trend Storico',
                    line=dict(width=2, dash='dot')
                ))
        except Exception:
            pass

    if forecast_data is not None and not forecast_data.empty:
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.values,
            mode='lines',
            name=f'Previsione ({ "Settimana" if freq=="W" else "Mese" })',
            line=dict(width=2, dash='dash')
        ))

    fig.update_layout(
        title=f'Vendite & Previsione - {product_name} ({ "Settimanale" if freq=="W" else "Mensile" })',
        xaxis_title='Periodo',
        yaxis_title='Unità',
        hovermode='x unified',
        height=480,
        showlegend=True
    )
    return fig


def load_product_settings():
    try:
        if os.path.exists('product_settings.csv'):
            settings_df = pd.read_csv('product_settings.csv')
            settings_df['sku'] = settings_df['sku'].astype(str).str.strip()
            return settings_df
        return pd.DataFrame(columns=['sku', 'moq', 'lead_time'])
    except Exception as e:
        st.error(f"Errore nel caricamento delle impostazioni prodotti: {str(e)}")
        return pd.DataFrame(columns=['sku', 'moq', 'lead_time'])

def save_product_settings(settings_df):
    try:
        settings_df.to_csv('product_settings.csv', index=False)
    except Exception as e:
        st.error(f"Errore nel salvataggio delle impostazioni prodotti: {str(e)}")

def get_stock_fields(stock_info, sku):
    if stock_info is None or stock_info.empty:
        return 0, 0, 0
    row = stock_info[stock_info['sku'].astype(str).str.strip() == str(sku).strip()]
    if row.empty:
        return 0, 0, 0
    qty_reserved = int(row['qty_reserved'].iloc[0]) if 'qty_reserved' in row.columns else 0
    qty_incoming = int(row['qty_incoming'].iloc[0]) if 'qty_incoming' in row.columns else 0
    current_stock = int(row['current_stock'].iloc[0]) if 'current_stock' in row.columns else 0
    return current_stock, qty_reserved, qty_incoming

def _bytes_key(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()

@st.cache_data
def _compute_all(results_key: str, df_mov: pd.DataFrame, stock_info: pd.DataFrame, forecast_days: int, safety_stock_days: int, safety_margin: float, product_settings: pd.DataFrame):
    # Qui metti il calcolo di:
    # - products_df
    # - results_df (tabella riepilogo)
    # - all_sales_data (serie globale)
    # Ritorna un dict con questi oggetti pronti.
    # NB: NON usare st.* qui dentro.
    twelve_months_ago = datetime.now() - timedelta(days=365)

    active_products = []
    for sku in df_mov['sku'].unique():
        pd_sku = df_mov[df_mov['sku']==sku].copy().sort_values('date')
        pd_sku.set_index('date', inplace=True)
        recent = pd_sku[pd_sku.index >= twelve_months_ago]['units_sold'].sum()
        cur, _, _ = get_stock_fields(stock_info, sku)
        if cur > 0 or recent > 0:
            name = pd_sku['product_name'].iloc[0] if 'product_name' in pd_sku.columns else f"Prodotto {sku}"
            active_products.append({'sku': sku, 'name': name, 'current_stock': int(cur), 'recent_movements': int(recent)})

    products_df = pd.DataFrame(active_products)

    # tasso crescita globale
    all_sales_data = df_mov.copy().sort_values('date').set_index('date')
    global_series = all_sales_data.groupby(all_sales_data.index)['units_sold'].sum()
    global_growth = calculate_growth_rate(global_series, None)

    # costruisci results_df come nel tuo loop attuale (puoi riusare il codice invariato),
    # ma SENZA st.write e st.*; restituisci il DataFrame finale.
    results = []

    for _, prow in products_df.iterrows():
        sku = prow['sku']; product_name = prow['name']
        product_data = df_mov[df_mov['sku']==sku].copy().sort_values('date').set_index('date')
        daily_sales = product_data['units_sold'].resample('D').sum().fillna(0)
        if 'units_sold_b2b' in product_data.columns:
            daily_b2b = product_data['units_sold_b2b'].resample('D').sum().fillna(0)
            daily_b2c = product_data['units_sold_b2c'].resample('D').sum().fillna(0)
        else:
            daily_b2b = pd.Series(0, index=daily_sales.index)
            daily_b2c = pd.Series(0, index=daily_sales.index)

        current_stock, qty_reserved, qty_incoming = get_stock_fields(stock_info, sku)
        product_growth = calculate_growth_rate(daily_sales, None)
        growth_rate = max(product_growth, global_growth)

        all_products_sales = df_mov.copy().set_index('date')
        forecast_total = forecast_with_prophet(daily_sales, forecast_days, all_products_data=all_products_sales, current_sku=sku)

        central_total = get_central_forecast_series(forecast_total)
        forecast_tot = int(central_total.head(forecast_days).sum()) if not central_total.empty else 0

        if not product_settings[product_settings['sku'] == str(sku)].empty:
            product_moq = int(product_settings.loc[product_settings['sku']==str(sku), 'moq'].iloc[0]) if 'moq' in product_settings.columns else 1
            product_lead_time = int(product_settings.loc[product_settings['sku']==str(sku), 'lead_time'].iloc[0]) if 'lead_time' in product_settings.columns else 7
        else:
            product_moq = 1; product_lead_time = 7

        units_to_order, status, details = calculate_order_recommendation(
            forecast_total, current_stock, safety_stock_days, product_lead_time,
            qty_incoming=qty_incoming, qty_reserved=qty_reserved,
            safety_margin=safety_margin, moq=product_moq
        )

        monthly_stats = calculate_monthly_average_excluding_oos(daily_sales, min_run_days=14)
        avg_monthly = monthly_stats['monthly_avg']
        last_month_avg = float(daily_sales.tail(30).mean() * 30)

        results.append({
            'sku': sku, 'name': product_name,
            'current_stock': int(current_stock),
            'forecast_tot': int(forecast_tot),
            'qty_reserved': int(qty_reserved),
            'qty_incoming': int(qty_incoming),
            'status': status,
            'units_to_order': int(units_to_order),
            'monthly_avg': int(round(avg_monthly)),
            'last_month_avg': int(round(last_month_avg)),
            'moq': int(product_moq),
            'lead_time': int(product_lead_time)
        })

    results_df = pd.DataFrame(results)
    return {
        'products_df': products_df,
        'results_df': results_df,
        'all_sales_data': all_sales_data
    }


# -------------------------------
# MAIN
# -------------------------------

def main():
    translations = load_translations()

    if 'product_settings' not in st.session_state:
        st.session_state.product_settings = load_product_settings()

    with st.sidebar:
        st.image("https://eatsmallgiants.com/cdn/shop/files/logo.svg?v=1700560810&width=100", width=100)
        language = st.selectbox("🌐 Language / Lingua", ["it", "en"], index=0)
        t = translations[language]
        st.markdown("---")

        # 1) CARICA FILE STOCK
        stock_file = st.file_uploader(
            "📦 Carica File Stock Attuale",
            type=['xlsx', 'xls', 'csv'],
            help="File con stock: SKU interno, Q.tà disponibile, Q.tà in stock, ecc.",
            key="stock_file"
        )

        # 2) BOTTONI STOCK
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            use_last_stock = st.button("⬇️ Usa ultimo stock (Drive)", key="use_last_stock")
        with col_s2:
            save_stock = st.button("⬆️ Salva stock su Drive", key="save_stock")

        st.markdown("---")

        # 3) CARICA FILE MOVIMENTI
        uploaded_file = st.file_uploader(
            "📊 " + t['upload_file'] + " (Movimenti)",
            type=['xlsx', 'xls', 'csv'],
            help="File con movimenti: Date, Type, Internal SKU, Quantity, ecc.",
            key="movements_file"
        )

        # 4) BOTTONI MOVIMENTI
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            use_last_mov = st.button("⬇️ Usa ultimo movimenti (Drive)", key="use_last_mov")
        with col_m2:
            save_mov = st.button("⬆️ Salva movimenti su Drive", key="save_mov")

        st.markdown("---")

        # PARAMETRI
        st.markdown("### ⚙️ Parametri")
        forecast_days = st.slider(t['forecast_days'], 30, 365, 90)
        safety_stock_days = st.slider(t['safety_stock'], 7, 30, 14)

        st.markdown("### 📦 Parametri Avanzati")
        safety_margin = st.slider("Margine di Sicurezza (%)", 0, 50, 10) / 100

        st.markdown("### 📈 Visualizzazione")
        view_freq = st.radio("Granularità grafici", options=["Mensile","Settimanale"], index=0, horizontal=True)
        freq = 'M' if view_freq == "Mensile" else 'W'


                # --- Diagnostica rapida Drive ---
        st.markdown("---")
        if st.button("🧪 Test Drive"):
            try:
                drive, folder_id = _drive_client()
                st.success(f"OK Drive. Folder: {folder_id}")
                mf = _load_manifest()
                if mf:
                    st.write("📄 Manifest trovato:", mf)
                else:
                    st.info("📄 Manifest assente o vuoto (verrà creato al primo salvataggio).")
            except Exception as e:
                st.error(f"Drive KO: {e}")

        with st.sidebar.expander("🛠️ Admin – Baseline B2C Shopify", expanded=False):
            st.markdown("Carica il CSV *Orders with line items* (Shopify).")
            baseline_file = st.file_uploader("CSV baseline B2C Shopify", type=["csv"], key="baseline_b2c_csv")

            overwrite = st.checkbox("Sovrascrivi se esiste già", value=False, help="Aggiorna il manifest e sostituisci il file su Drive")
            do_register = st.button("📌 Imposta/aggiorna baseline B2C")

            if do_register:
                try:
                    if baseline_file is None:
                        st.error("Carica un file CSV prima.")
                    else:
                        # 1) Leggi bytes e calcola hash
                        csv_bytes = baseline_file.getvalue()
                        sha = _sha1(csv_bytes)

                        # 2) Carica su Drive con nome speciale
                        fname = "b2c_baseline_shopify.csv"
                        file_id, title = gdrive_upload_bytes(fname, csv_bytes, "text/csv")

                        # 3) Se già esiste e non vuoi sovrascrivere → blocca
                        existing = get_b2c_baseline_record()
                        if existing and existing.get("locked") and not overwrite:
                            st.warning("Esiste già una baseline con locked=true. Abilita 'Sovrascrivi' per sostituire.")
                        else:
                            # 4) Scrivi nel manifest
                            rec = {
                                "file_id": file_id,
                                "title": title,
                                "orig_name": baseline_file.name,
                                "uploaded_at_utc": datetime.utcnow().strftime("%Y%m%d-%H%M%S"),
                                "sha1": sha,
                                "locked": True
                            }
                            set_b2c_baseline_record(rec)
                            st.success(f"Baseline registrata: {title} (sha1={sha})")

                            # 5) Sanity check lettura immediata
                            test_df = _load_b2c_baseline_from_drive()
                            if test_df.empty:
                                st.warning("Baseline caricata ma parsing vuoto: verifica le colonne dell'export.")
                            else:
                                # garantisci tipo datetime
                                test_df["date"] = pd.to_datetime(test_df["date"], errors="coerce")
                                dmin = test_df["date"].min()
                                dmax = test_df["date"].max()
                                st.info(f"Baseline OK. Righe utili: {len(test_df)} | Periodo: {dmin:%Y-%m-%d} → {dmax:%Y-%m-%d}")
                except Exception as e:
                    st.error(f"Errore durante la registrazione baseline: {e}")


    # --- Flag bottoni Drive lettura ---
    want_last_stock = ('use_last_stock' in locals() and use_last_stock)
    want_last_mov   = ('use_last_mov'   in locals() and use_last_mov)

    # --- Persistenza & acquisizione MOVIMENTI ---
    if uploaded_file is not None:
        st.session_state['mov_file_bytes'] = uploaded_file.getvalue()
        st.session_state['mov_file_name']  = uploaded_file.name
    elif want_last_mov:
        last = get_last_saved_drive("movements")
        if last:
            try:
                data_bytes = gdrive_get_file_content(last["file_id"])
                st.session_state['mov_file_bytes'] = data_bytes
                st.session_state['mov_file_name']  = last["title"]
                st.success(f"✅ Recuperato movimenti da Drive: {last['title']}")
            except Exception as e:
                st.error(f"Errore nel recupero movimenti da Drive: {e}")

    # --- Persistenza & acquisizione STOCK ---
    if stock_file is not None:
        st.session_state['stock_file_bytes'] = stock_file.getvalue()
        st.session_state['stock_file_name']  = stock_file.name
    elif want_last_stock:
        last_s = get_last_saved_drive("stock")
        if last_s:
            try:
                s_bytes = gdrive_get_file_content(last_s["file_id"])
                st.session_state['stock_file_bytes'] = s_bytes
                st.session_state['stock_file_name']  = last_s["title"]
                st.success(f"✅ Recuperato stock da Drive: {last_s['title']}")
            except Exception as e:
                st.error(f"Errore nel recupero stock da Drive: {e}")

    # --- Gate: procedi SOLO se ho ENTRAMBI i file in sessione ---
    missing = []
    if 'mov_file_bytes' not in st.session_state:  missing.append("file **Movimenti**")
    if 'stock_file_bytes' not in st.session_state: missing.append("file **Stock**")

    if missing:
        st.markdown("## 👋 Benvenuto")
        st.warning("Per procedere servono **entrambi** i file:")
        for m in missing: st.write(f"• {m}")
        st.stop()

    # --- Ho entrambi: leggo SOLO dai bytes in sessione ---
    mov_bytes   = st.session_state['mov_file_bytes']
    mov_name    = st.session_state['mov_file_name']
    stock_bytes = st.session_state['stock_file_bytes']
    stock_name  = st.session_state['stock_file_name']

    # --- Salvataggi su Drive, se richiesti (riusando i bytes in sessione) ---
    if ('save_mov' in locals() and save_mov) and ('mov_file_bytes' in st.session_state):
        class _MemUp:
            def __init__(self, d, n): self._d=d; self.name=n
            def getvalue(self): return self._d
        try:
            rec = save_uploaded_file_drive(_MemUp(mov_bytes, mov_name), "movements")
            st.success(f"✅ Movimenti salvati su Drive come **{rec['title']}**")
        except Exception as e:
            st.error(f"Errore salvataggio movimenti su Drive: {e}")

    if ('save_stock' in locals() and save_stock) and ('stock_file_bytes' in st.session_state):
        class _MemUpS:
            def __init__(self, d, n): self._d=d; self.name=n
            def getvalue(self): return self._d
        try:
            rec_s = save_uploaded_file_drive(_MemUpS(stock_bytes, stock_name), "stock")
            st.success(f"✅ Stock salvato su Drive come **{rec_s['title']}**")
        except Exception as e:
            st.error(f"Errore salvataggio stock su Drive: {e}")



    # ------- Processamento completo -------
    with st.spinner(translations['it']['processing']):

        # === MOVIMENTI: leggi sempre dai bytes in sessione ===
        try:
            bio = io.BytesIO(mov_bytes)
            bio.name = mov_name
            if mov_name.lower().endswith('.csv'):
                try:
                    raw_df = pd.read_csv(bio, encoding='utf-8-sig', sep=',')
                    if len(raw_df.columns) == 1:
                        bio.seek(0); raw_df = pd.read_csv(bio, encoding='utf-8-sig', sep=';')
                except Exception:
                    bio.seek(0); raw_df = pd.read_csv(bio, encoding='latin-1', sep=',')
            else:
                raw_df = pd.read_excel(bio)

            st.write(f"📁 **File caricato (movimenti):** {mov_name}")
            st.write(f"📊 **Righe totali:** {len(raw_df)}")

            df = process_excel_data(raw_df)
            if df.empty:
                st.error("❌ Nessun dato valido trovato nel file movimenti."); st.stop()
        except Exception as e:
            st.error(f"Errore durante l'elaborazione del file movimenti: {str(e)}"); st.stop()

            # >>> Shopify B2C baseline (se presente nel manifest)
            b2c_base = _load_b2c_baseline_from_drive()
            if not b2c_base.empty:
                # uniforma eventuali colonne mancanti
                for c in ["date","sku","product_name","units_sold","units_sold_b2b","units_sold_b2c"]:
                    if c not in df.columns:
                        df[c] = 0 if c.startswith("units_") else df.get(c, "")

                merged = pd.concat(
                    [df[["date","sku","product_name","units_sold","units_sold_b2b","units_sold_b2c"]],
                    b2c_base[["date","sku","product_name","units_sold","units_sold_b2b","units_sold_b2c"]]],
                    ignore_index=True
                )
                df = (merged.groupby(["date","sku","product_name"], as_index=False)
                        .agg({"units_sold":"sum",
                                "units_sold_b2b":"sum",
                                "units_sold_b2c":"sum"}))
                st.info(f"🧩 Storico B2C extra unito: {len(b2c_base)} righe baseline aggiunte")


        # === STOCK: leggi sempre dai bytes in sessione ===
        try:
            sbio = io.BytesIO(stock_bytes)
            sbio.name = stock_name
            if stock_name.lower().endswith('.csv'):
                try:
                    stock_raw_df = pd.read_csv(sbio, encoding='utf-8-sig', sep=',')
                    if len(stock_raw_df.columns) == 1:
                        sbio.seek(0); stock_raw_df = pd.read_csv(sbio, encoding='utf-8-sig', sep=';')
                except Exception:
                    sbio.seek(0); stock_raw_df = pd.read_csv(sbio, encoding='latin-1', sep=',')
            else:
                stock_raw_df = pd.read_excel(sbio)

            stock_raw_df.columns = [str(c).strip().replace('\ufeff','') for c in stock_raw_df.columns]
            st.write(f"📋 **Colonne stock trovate:** {list(stock_raw_df.columns)}")
            st.write(f"📊 **Righe stock:** {len(stock_raw_df)}")

            stock_info = process_stock_file(stock_raw_df)
            if stock_info.empty:
                st.error("❌ Stock non valido."); st.stop()
        except Exception as e:
            st.error(f"Errore nel processare il file stock: {str(e)}"); st.stop()



        # === COMPUTE ALL CACHED, PRIMA DELLA TABELLA ===
        # Key stabile: dipende dai bytes dei file + parametri UI
        # Fingerprint delle impostazioni per invalidare la cache quando cambi MOQ/lead_time
        if 'product_settings' in st.session_state and not st.session_state.product_settings.empty:
            _ps = st.session_state.product_settings.sort_values("sku").astype(str)
            settings_fp = hashlib.sha256(_ps.to_csv(index=False).encode("utf-8")).hexdigest()[:10]
        else:
            settings_fp = "no-settings"

        results_key = "|".join([
            _bytes_key(mov_bytes),
            _bytes_key(stock_bytes),
            str(forecast_days),
            str(safety_stock_days),
            str(safety_margin),
            settings_fp
        ])


        bundle = _compute_all(
            results_key,
            df,
            stock_info,
            forecast_days,
            safety_stock_days,
            safety_margin,
            st.session_state.product_settings.copy() if 'product_settings' in st.session_state else pd.DataFrame(columns=['sku','moq','lead_time'])
        )

        products_df   = bundle['products_df']
        results_df    = bundle['results_df']
        all_sales_data = bundle['all_sales_data']


        # ------- TABELLA RIEPILOGO (IN ALTO) -------
        st.markdown('<a id="riepilogo-prodotti"></a>', unsafe_allow_html=True)
        st.markdown("### 📊 Riepilogo Prodotti")

        # 🔎 Cerca per nome prodotto (server-side)
        search_name = st.text_input(
            "Cerca per nome prodotto",
            value="",
            placeholder="es. bites",
            key="df_filter_by_name"
        )

        # Mappa stato ed etichette
        status_map = {'critical':'🔴 Critico','warning':'🟡 Attenzione','good':'🟢 OK'}
        results_df['status'] = results_df['status'].map(status_map)

        # Etichetta dinamica per la previsione totale (basata sullo slider)
        prev_col_label = f"Prev Tot ({forecast_days} gg)"

        show_df = results_df[[
            'sku',
            'name',
            'current_stock',
            'forecast_tot',
            'monthly_avg',
            'last_month_avg',
            'status',
            'units_to_order',
            'moq',
            'lead_time'
        ]].rename(columns={
            'sku': 'SKU',
            'name': 'Nome Prodotto',
            'current_stock': 'Stock Attuale',
            # 'forecast_tot' resta uguale come field
            'monthly_avg': 'Media Mensile',
            'last_month_avg': 'Media Ultimo Mese',
            'status': 'Stato',
            'units_to_order': 'Da Ordinare',
            'moq': 'MOQ',
            'lead_time': 'Lead Time'
        })


        # Applica filtro solo su Nome Prodotto
        if search_name:
            mask = show_df["Nome Prodotto"].astype(str).str.contains(search_name, case=False, na=False)
            show_df = show_df[mask].reset_index(drop=True)

            # Selezione persistente: se la riga è nascosta dal filtro, NON azzero
            selected_hidden = False
            if st.session_state.get('selected_sku'):
                sku_sel = str(st.session_state['selected_sku'])
                visible_skus = set(show_df['SKU'].astype(str))
                if sku_sel not in visible_skus:
                    selected_hidden = True
                    # prova a recuperare il nome dal dataset completo (results_df), non filtrato
                    try:
                        rec_all = results_df[results_df['sku'].astype(str) == sku_sel].head(1)
                        if not rec_all.empty:
                            st.session_state['selected_name'] = rec_all['name'].iloc[0]
                    except Exception:
                        pass



        # Costruisci le opzioni Ag-Grid
        gb = GridOptionsBuilder.from_dataframe(show_df)

        # ID riga stabile = SKU (serve anche a non perdere la selezione)
        gb.configure_grid_options(
            getRowId=JsCode("function(p){return p.data && (p.data.SKU || p.data.sku);}"),
        )

        # Metti il checkbox nella colonna SKU (prima colonna)
        gb.configure_column(
            "SKU",
            pinned="left",                # SKU resta la prima a sinistra
            checkboxSelection=True,       # checkbox dentro la colonna SKU
            headerCheckboxSelection=False,
            suppressSizeToFit=False
        )

        # Lascia "Nome Prodotto" non per forza pinned, così lo SKU+checkbox rimangono davvero primi
        gb.configure_column(
            "Nome Prodotto",
            pinned=None,                  # <— togli il "left" qui se vuoi davvero prima lo SKU
            suppressSizeToFit=True,
            wrapText=True,
            autoHeight=True,
            headerTooltip="Nome Prodotto",
            cellStyle={"white-space": "normal"},
            filter="agTextColumnFilter",
            floatingFilter=True
        )


        # Etichetta UI desiderata
        prev_col_label = f"Prev Tot ({forecast_days} gg)"

        # Header dinamico sulla colonna forecast tenendo il field fisso
        gb.configure_column(
            "forecast_tot",
            headerName=prev_col_label,
            type=["numericColumn"]
        )

        # Default: niente filtri sulle colonne
        gb.configure_default_column(resizable=True, filter=False, floatingFilter=False)


        # SKU: si adatta (NO suppress), allineamento a sinistra
        gb.configure_column("SKU", pinned=None, suppressSizeToFit=False)

        gb.configure_column(
            "Nome Prodotto",
            pinned="left",
            suppressSizeToFit=True,
            wrapText=True,
            autoHeight=True,
            headerTooltip="Nome Prodotto",
            cellStyle={"white-space": "normal"},
            filter="agTextColumnFilter",     # ← filtro SOLO su questa colonna
            floatingFilter=True
        )

        # Tutte le altre colonne si adattano con flex
        for col in [c for c in show_df.columns if c not in ["Nome Prodotto"]]:
            gb.configure_column(col, flex=1, suppressSizeToFit=False)

        # Editing per MOQ e Lead Time
        gb.configure_column("MOQ", editable=True, type=["numericColumn"])
        gb.configure_column("Lead Time", editable=True, type=["numericColumn"])
        gb.configure_grid_options(stopEditingWhenCellsLoseFocus=True)

        # Selezione riga singola con checkbox
        gb.configure_selection(selection_mode="single", use_checkbox=True)

        # No paginazione
        gb.configure_grid_options(pagination=False, domLayout='normal')

        # JS: fit di tutte le colonne tranne "Nome Prodotto"; poi imposta una minWidth decente al nome
        gb.configure_grid_options(
            onFirstDataRendered=JsCode("""
                function(params){
                    // 1) Fit di tutte le colonne (quelle con suppressSizeToFit=True sono escluse)
                    params.api.sizeColumnsToFit();

                    // 2) Dai un minimo di larghezza al Nome Prodotto e abilita header wrap
                    try {
                        var col = params.columnApi.getColumn('Nome Prodotto');
                        if (col && col.getColDef()) {
                            // min larghezza per leggere bene
                            params.columnApi.setColumnWidths([{ key: 'Nome Prodotto', newWidth: Math.max(col.getActualWidth(), 260) }]);
                            col.getColDef().wrapHeaderText = true;
                            col.getColDef().autoHeaderHeight = true;
                            params.api.refreshHeader();
                        }
                    } catch(e) {}
                }
            """)
        )

        grid_options = gb.build()


        grid_response = AgGrid(
            show_df,
            gridOptions=grid_options,
            data_return_mode=DataReturnMode.AS_INPUT,
            update_mode=GridUpdateMode.SELECTION_CHANGED,   # ← niente VALUE_CHANGED
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=False,
            theme="balham",
            height=800,
            key="main_grid",
            enable_enterprise_modules=False
        )

                # --- Persistenza MOQ / Lead Time da griglia ---
        edited_df = pd.DataFrame(grid_response.get("data", show_df)).copy()
        # Normalizza colonne attese
        if not edited_df.empty:
            for col in ["SKU", "MOQ", "Lead Time"]:
                if col not in edited_df.columns:
                    st.error(f"Manca la colonna {col} nella griglia, impossibile salvare impostazioni.")
                    st.stop()
            # Prepara il dataframe impostazioni
            settings_df = edited_df[["SKU", "MOQ", "Lead Time"]].rename(columns={"SKU":"sku","Lead Time":"lead_time","MOQ":"moq"})
            # Coerce numerici
            settings_df["sku"] = settings_df["sku"].astype(str).str.strip()
            settings_df["moq"] = pd.to_numeric(settings_df["moq"], errors="coerce").fillna(1).astype(int)
            settings_df["lead_time"] = pd.to_numeric(settings_df["lead_time"], errors="coerce").fillna(7).astype(int)

            # Salva in sessione per uso immediato nel dettaglio
            st.session_state.product_settings = settings_df.copy()

            # Scrivi su disco
            save_product_settings(settings_df)

        # --- Gestione selezione robusta ---
        selected_rows = grid_response.get("selected_rows", [])
        if isinstance(selected_rows, pd.DataFrame):
            selected_rows = selected_rows.to_dict("records")

        # Aggiorna la sessione solo quando l'utente seleziona
        if selected_rows:
            sr = selected_rows[0]
            sku_clicked  = sr.get("SKU") or sr.get("sku")
            name_clicked = sr.get("Nome Prodotto") or sr.get("name")
            if sku_clicked:
                st.session_state["selected_sku"]  = sku_clicked
                st.session_state["selected_name"] = name_clicked

        # Usa sempre la selezione salvata
        sku_show  = st.session_state.get("selected_sku")
        name_show = st.session_state.get("selected_name")

        # Se la riga selezionata è nascosta dal filtro, NON azzerare: prova a recuperare il nome dal dataset completo
        selected_hidden = False
        if sku_show is not None:
            visible_skus = set(show_df["SKU"].astype(str))
            if str(sku_show) not in visible_skus:
                selected_hidden = True
                try:
                    rec_all = results_df[results_df["sku"].astype(str) == str(sku_show)].head(1)
                    if not rec_all.empty:
                        name_show = rec_all["name"].iloc[0]
                        st.session_state["selected_name"] = name_show
                except Exception:
                    pass

        if not sku_show:
            st.info("Seleziona un prodotto nella tabella per vedere il dettaglio.")
            st.stop()

        if selected_hidden:
            st.caption("La riga selezionata è temporaneamente nascosta dal filtro. Il dettaglio resta visibile.")


        selected_row = selected_rows[0]
        sku_show  = selected_row.get("SKU") or selected_row.get("sku")
        name_show = selected_row.get("Nome Prodotto") or selected_row.get("name")
        st.session_state['selected_sku']  = sku_show
        st.session_state['selected_name'] = name_show


        # Memorizza la selezione corrente
        sku_show  = selected_row.get("SKU") or selected_row.get("sku")
        name_show = selected_row.get("Nome Prodotto") or selected_row.get("name")

        if not sku_show:
            st.error(f"Chiavi disponibili nella riga selezionata: {list(selected_row.keys())}")
            st.stop()

        # Leggi eventuali modifiche a MOQ/Lead Time e salvale come prima
        edited_df = pd.DataFrame(grid_response["data"])
        new_settings = []
        for _, row in edited_df.iterrows():
            new_settings.append({
                'sku': row['SKU'],
                'moq': int(row['MOQ']),
                'lead_time': int(row['Lead Time'])
            })
        st.session_state.product_settings = pd.DataFrame(new_settings)
        save_product_settings(st.session_state.product_settings)

        # --- Recupero riga selezionata da Ag-Grid (robusto) ---
        selected_rows = grid_response.get("selected_rows", [])

        # Se AgGrid restituisce un DataFrame, converti in list-of-dicts
        if isinstance(selected_rows, pd.DataFrame):
            selected_rows = selected_rows.to_dict("records")

        # Controllo “vuoto” sicuro
        if selected_rows is None or len(selected_rows) == 0:
            st.info("Seleziona un prodotto nella tabella per vedere il dettaglio.")
            st.stop()

        # Prima (e unica) riga selezionata
        selected_row = selected_rows[0]

        # Nomi colonne robusti (a seconda dei rename fatti a monte)
        sku_show  = selected_row.get("SKU") or selected_row.get("sku")
        name_show = selected_row.get("Nome Prodotto") or selected_row.get("name")


        st.markdown("<p class='small-note'>Suggerimento: seleziona una sola riga per aprire il pannello dettagli.</p>", unsafe_allow_html=True)

        # ------- DETTAGLIO PRODOTTO (apre da tabella) -------

        product_data_show = df[df['sku']==sku_show].copy().sort_values('date').set_index('date')
        daily_sales_show = product_data_show['units_sold'].resample('D').sum().fillna(0)

        # split canale per il dettaglio
        if 'units_sold_b2b' in product_data_show.columns:
            daily_b2c_show = product_data_show['units_sold_b2c'].resample('D').sum().fillna(0)
            daily_b2b_show = (daily_sales_show - daily_b2c_show).clip(lower=0)
        else:
            daily_b2b_show = pd.Series(0, index=daily_sales_show.index)
            daily_b2c_show = pd.Series(0, index=daily_sales_show.index)

        forecast_show = forecast_with_prophet(daily_sales_show, forecast_days, all_products_data=df.set_index('date'), current_sku=sku_show)
        forecast_b2b_show = forecast_with_prophet(daily_b2b_show, forecast_days, all_products_data=df.set_index('date'), current_sku=f"{sku_show}_B2B")
        forecast_b2c_show = forecast_with_prophet(daily_b2c_show, forecast_days, all_products_data=df.set_index('date'), current_sku=f"{sku_show}_B2C")

        central_series_ui = get_central_forecast_series(forecast_show)

        # metriche quick
        current_stock, qty_reserved, qty_incoming = get_stock_fields(stock_info, sku_show)
        product_settings = st.session_state.product_settings
        if not product_settings[product_settings['sku'] == str(sku_show)].empty:
            product_lead_time = int(product_settings.loc[product_settings['sku']==str(sku_show), 'lead_time'].iloc[0]) if 'lead_time' in product_settings.columns else 7
            product_moq = int(product_settings.loc[product_settings['sku']==str(sku_show), 'moq'].iloc[0]) if 'moq' in product_settings.columns else 1
        else:
            product_lead_time = 7
            product_moq = 1

        units_to_order, status, details = calculate_order_recommendation(
            forecast_show, current_stock, safety_stock_days, product_lead_time,
            qty_incoming=qty_incoming, qty_reserved=qty_reserved,
            safety_margin=safety_margin, moq=product_moq
        )

        monthly_stats_show = calculate_monthly_average_excluding_oos(daily_sales_show, min_run_days=14)

        st.markdown("---")
        st.subheader(f"🔎 Dettaglio: {name_show} ({sku_show})")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label=translations['it']['current_stock'], value=f"{current_stock:,}")
        with col2:
            predicted_demand = int(central_series_ui.head(product_lead_time + safety_stock_days).sum()) if not central_series_ui.empty else 0
            st.metric(label=translations['it']['predicted_demand'], value=f"{predicted_demand:,}")
        with col3:
            st.metric(label="📊 Media Mensile Storica", value=f"{monthly_stats_show['monthly_avg']:,.0f}")
        with col4:
            st.metric(label=translations['it']['units_to_order'], value=f"{units_to_order:,}")
            status_colors = {"critical":"#ff4444","warning":"#ffaa00","good":"#00aa44"}
            st.markdown(f"<p style='color:{status_colors[status]};font-weight:bold;text-align:center;'>Status: {translations['it'][status]}</p>", unsafe_allow_html=True)

        # Forecast 30/90 per canale (mostra anche qui)
        central_b2c_show = get_central_forecast_series(forecast_b2c_show)
        # Totale centrale già calcolato: central_series_ui
        central_b2b_show = (central_series_ui - central_b2c_show).clip(lower=0)
        f30_b2b_show = int(central_b2b_show[:30].sum()) if not central_b2b_show.empty else 0
        f30_b2c_show = int(central_b2c_show[:30].sum()) if not central_b2c_show.empty else 0
        f90_b2b_show = int(central_b2b_show[:90].sum()) if not central_b2b_show.empty else 0
        f90_b2c_show = int(central_b2c_show[:90].sum()) if not central_b2c_show.empty else 0

        st.markdown(f"**Previsione 30gg** → B2B: **{f30_b2b_show}** · B2C: **{f30_b2c_show}**  |  **Previsione 90gg** → B2B: **{f90_b2b_show}** · B2C: **{f90_b2c_show}**")

        with st.expander("🔍 Dettagli Calcolo Ordine"):
            st.markdown("#### 📊 Statistiche Storiche")
            st.write(f"Media Mensile (ponderata): {monthly_stats_show['monthly_avg']:,.1f}")
            st.write(f"Periodo: {monthly_stats_show['first_sale_date']} → {monthly_stats_show['last_sale_date']}")
            st.write(f"Mesi totali: {monthly_stats_show['total_months']}")
            st.write(f"Mesi validi: {monthly_stats_show['valid_months']}")
            st.write(f"Mesi OOS: {monthly_stats_show['oos_months']}")
            st.write(f"Mesi ponderati equivalenti: {monthly_stats_show['weighted_months']:.1f}")
            st.write(f"Vendite totali: {monthly_stats_show['total_sales']:,.0f}")
            st.markdown("---")
            st.markdown("#### 💼 Dettagli Ordine")
            st.write(f"Domanda Prevista ({product_lead_time + safety_stock_days} gg): {details['forecast_demand']:,.1f}")
            st.write(f"Domanda Prevista (centrale): {details['forecast_demand_central']:,.1f}")
            st.write(f"Stock Fisico: {details['current_stock']:,.0f}")
            st.write(f"Prenotate: {details['qty_reserved']:,.0f}")
            st.write(f"In Arrivo: {details['qty_incoming']:,.0f}")
            st.write(f"Stock Effettivo: {details['effective_stock']:,.1f}")
            st.write(f"Giorni copertura: {details['days_of_stock']:.1f}")
            st.write(f"Margine di sicurezza: {int(safety_margin*100)}%")
            st.write(f"MOQ: {product_moq}")

            if details['forecast_demand'] > 0:
                avg_daily_demand = details['forecast_demand'] / (product_lead_time + safety_stock_days)
                stockout_days = details['effective_stock'] / avg_daily_demand if avg_daily_demand > 0 else 999
                stockout_date = datetime.now() + timedelta(days=int(stockout_days))
                st.warning(f"⚠️ Previsto esaurimento scorte: {stockout_date.strftime('%Y-%m-%d')} (tra {int(stockout_days)} giorni)")

        st.markdown("---")
        # Grafici (settimanale / mensile)
        c1, c2 = st.columns(2)
        with c1:
            chart_w = create_forecast_chart(daily_sales_show, forecast_show, name_show, freq=freq)
            st.plotly_chart(chart_w, use_container_width=True)
        with c2:
            # Mostra anche split canale (sovrapposto): qui faccio storico B2B/B2C e forecast B2B/B2C come due linee
            hist_b2b, fc_b2b = _aggregate_series_for_display(daily_b2b_show, forecast_b2b_show, freq=freq)
            hist_b2c, fc_b2c = _aggregate_series_for_display(daily_b2c_show, forecast_b2c_show, freq=freq)
            fig_split = go.Figure()
            # Storici
            fig_split.add_trace(go.Scatter(x=hist_b2b.index, y=hist_b2b.values, mode='lines+markers',
                                        name=f'B2B Storico ({ "Settimana" if freq=="W" else "Mese" })',
                                        line=dict(width=2), marker=dict(size=6)))
            fig_split.add_trace(go.Scatter(x=hist_b2c.index, y=hist_b2c.values, mode='lines+markers',
                                        name=f'B2C Storico ({ "Settimana" if freq=="W" else "Mese" })',
                                        line=dict(width=2), marker=dict(size=6)))

            # Trend B2B
            try:
                if len(hist_b2b) >= 2 and np.isfinite(hist_b2b.values.astype(float)).all():
                    xb = np.arange(len(hist_b2b))
                    yb = hist_b2b.values.astype(float)
                    ab, bb = np.polyfit(xb, yb, 1)
                    trend_b2b = ab * xb + bb
                    fig_split.add_trace(go.Scatter(x=hist_b2b.index, y=trend_b2b, mode='lines',
                                                name='Trend B2B', line=dict(width=2, dash='dot')))
            except Exception:
                pass

            # Trend B2C
            try:
                if len(hist_b2c) >= 2 and np.isfinite(hist_b2c.values.astype(float)).all():
                    xc = np.arange(len(hist_b2c))
                    yc = hist_b2c.values.astype(float)
                    ac, bc = np.polyfit(xc, yc, 1)
                    trend_b2c = ac * xc + bc
                    fig_split.add_trace(go.Scatter(x=hist_b2c.index, y=trend_b2c, mode='lines',
                                                name='Trend B2C', line=dict(width=2, dash='dot')))
            except Exception:
                pass

            # Forecast
            fig_split.add_trace(go.Scatter(x=fc_b2b.index, y=fc_b2b.values, mode='lines',
                                        name=f'B2B Prev ({ "Settimana" if freq=="W" else "Mese" })',
                                        line=dict(width=2, dash='dash')))
            fig_split.add_trace(go.Scatter(x=fc_b2c.index, y=fc_b2c.values, mode='lines',
                                        name=f'B2C Prev ({ "Settimana" if freq=="W" else "Mese" })',
                                        line=dict(width=2, dash='dash')))

            fig_split.update_layout(
                title=f'{name_show} – Storico & Previsione per Canale ({ "Settimanale" if freq=="W" else "Mensile" })',
                xaxis_title='Periodo', yaxis_title='Unità',
                hovermode='x unified', height=480, showlegend=True
            )
            st.plotly_chart(fig_split, use_container_width=True)


        colh1, colh2 = st.columns(2)
        with colh1:
            st.subheader(translations['it']['historical_data'])
            # storico compatto: mostra ultimi 12 punti della serie aggregata scelta
            hist_for_table, _ = _aggregate_series_for_display(daily_sales_show, forecast_show, freq=freq)
            hist_df = hist_for_table.tail(12).reset_index()
            hist_df.columns = ['Periodo', 'Vendite']
            hist_df['Periodo'] = hist_df['Periodo'].dt.strftime('%Y-%m-%d')
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        with colh2:
            st.subheader(translations['it']['forecast'])
            _, fc_for_table = _aggregate_series_for_display(daily_sales_show, forecast_show, freq=freq)
            if fc_for_table is not None and len(fc_for_table) > 0:
                fc_df = fc_for_table.head(12).reset_index()
                fc_df.columns = ['Periodo', 'Previsione']
                fc_df['Previsione'] = fc_df['Previsione'].round().astype(int)
                fc_df['Periodo'] = fc_df['Periodo'].dt.strftime('%Y-%m-%d')
                st.dataframe(fc_df, use_container_width=True, hide_index=True)
            else:
                st.info("Dati insufficienti per la previsione")

# --- avvio ---
APP_NAME = os.getenv("APP_NAME", "SmallGiants App")
NAMES = [os.getenv("APP_USER", "SG")]
USERNAMES = [os.getenv("APP_USERNAME", "SG")]
PLAINTEXT_PASSWORD = os.getenv("APP_PASSWORD", "change-me-now")
COOKIE_NAME = os.getenv("COOKIE_NAME", "sg_cookie")
COOKIE_KEY = os.getenv("COOKIE_KEY", "set-a-long-random-key")

if __name__ == "__main__":
    if not st.session_state.get("_auth_ok", False):
        gate()
    else:
        main()
