import sys
print("USING PYTHON:", sys.executable)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import math
import calendar
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()  # carica .env se presente

## Prophet non viene più caricato

# Configurazione pagina
st.set_page_config(
    page_title="Small Giants - Inventory Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import streamlit as st

def gate():
    import os
    import streamlit as st

    expected_user = os.getenv("APP_USERNAME", "SG")
    expected_pwd  = os.getenv("APP_PASSWORD", "change-me")

    st.markdown("## 🔐 Login")

    # Form con key univoca e widget con key dedicate
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



# CSS personalizzato per migliorare l'aspetto
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
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
</style>
""", unsafe_allow_html=True)

def load_translations():
    """Definizioni multilingua"""
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

def find_column(df, possible_names):
    """
    Trova una colonna basandosi su una lista di possibili nomi (case insensitive)
    """
    for col in df.columns:
        for possible_name in possible_names:
            if possible_name.lower() in col.lower():
                return col
    return None

def process_stock_file(df):
    """
    Processa il file con le quantità attuali in magazzino
    """
    try:
        st.write("🔍 **Colonne stock trovate nel file:**", list(df.columns))
        
        # Pulisci i nomi delle colonne
        df.columns = df.columns.str.strip()

        # Se la prima colonna contiene il nome del file, le intestazioni sono nella prima riga
        if 'export' in str(df.columns[0]).lower() or 'unnamed' in str(df.columns[0]).lower():
            st.info("📋 Rilevate intestazioni nella prima riga del file stock. Riprocessando...")
            # La prima riga contiene i veri nomi delle colonne
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            df.columns = df.columns.str.strip()
            st.write("🔍 **Colonne stock corrette:**", list(df.columns))
            
        # Trova colonne magazzino per filtrare SMALLGIANTSAVAILABLE
        physical_warehouse_col = find_column(df, ['magazzino fisico', 'physical warehouse', 'warehouse physical', 'deposito fisico', 'wms', 'warehouse'])
        virtual_warehouse_col = find_column(df, ['magazzino virtuale', 'virtual warehouse', 'warehouse virtual', 'deposito virtuale', 'wms', 'warehouse'])
        
        # Filtra SOLO prodotti con ESATTAMENTE "SMALLGIANTSAVAILABLE"
        if physical_warehouse_col or virtual_warehouse_col:
            initial_rows = len(df)
            filter_mask = pd.Series([False] * len(df))
            
            if physical_warehouse_col:
                # Match esatto su "SMALLGIANTSAVAILABLE" (trim spazi ma case sensitive)
                physical_match = df[physical_warehouse_col].astype(str).str.strip() == 'SMALLGIANTSAVAILABLE'
                filter_mask = filter_mask | physical_match
            
            if virtual_warehouse_col:
                virtual_match = df[virtual_warehouse_col].astype(str).str.strip() == 'SMALLGIANTSAVAILABLE'
                filter_mask = filter_mask | virtual_match
            
            df = df[filter_mask].copy()
            filtered_count = initial_rows - len(df)
            
            if filtered_count > 0:
                st.info(f"🔍 Esclusi {filtered_count} prodotti non-SMALLGIANTSAVAILABLE (selezionati {len(df)} prodotti)")
            
            if len(df) == 0:
                st.warning("⚠️ Nessun prodotto con esattamente 'SMALLGIANTSAVAILABLE' trovato nel file stock")
                return pd.DataFrame()
            
            st.success(f"✅ Filtrati solo prodotti SMALLGIANTSAVAILABLE: {len(df)} prodotti")
        else:
            st.error("❌ Colonne 'Magazzino fisico' o 'Magazzino virtuale' non trovate. Impossibile filtrare.")
            return pd.DataFrame()
        
        # Trova le colonne chiave per il file stock
        sku_column = find_column(df, ['sku interno', 'internal sku', 'sku', 'codice', 'code'])
        available_column = find_column(df, ['disponibile', 'available', 'qtà disponibile', 'q.tà disponibile'])
        in_stock_column = find_column(df, ['in stock', 'stock', 'qtà in stock', 'q.tà in stock'])
        product_name_column = find_column(df, ['nome del prodotto', 'product name', 'nome', 'name', 'prodotto'])
        
        # Trova anche altre colonne utili
        reserved_column = find_column(df, ['prenotate', 'reserved', 'qtà prenotate', 'q.tà prenotate'])
        incoming_column = find_column(df, ['attesa di ricezione', 'incoming', 'in attesa', 'ricezione'])

        # Trova colonne magazzino per filtrare SMALLGIANTSAVAILABLE
        physical_warehouse_col = find_column(df, ['magazzino fisico', 'physical warehouse', 'warehouse physical', 'deposito fisico', 'wms', 'warehouse'])
        virtual_warehouse_col = find_column(df, ['magazzino virtuale', 'virtual warehouse', 'warehouse virtual', 'deposito virtuale', 'wms', 'warehouse'])
        
        # Filtra SOLO prodotti con ESATTAMENTE "SMALLGIANTSAVAILABLE"
        if physical_warehouse_col or virtual_warehouse_col:
            initial_rows = len(df)
            filter_mask = pd.Series([False] * len(df))
            
            if physical_warehouse_col:
                # Match esatto su "SMALLGIANTSAVAILABLE" (trim spazi ma case sensitive)
                physical_match = df[physical_warehouse_col].astype(str).str.strip() == 'SMALLGIANTSAVAILABLE'
                filter_mask = filter_mask | physical_match
                st.write(f"🔍 Prodotti con SMALLGIANTSAVAILABLE in magazzino fisico: {physical_match.sum()}")
            
            if virtual_warehouse_col:
                virtual_match = df[virtual_warehouse_col].astype(str).str.strip() == 'SMALLGIANTSAVAILABLE'
                filter_mask = filter_mask | virtual_match
                st.write(f"🔍 Prodotti con SMALLGIANTSAVAILABLE in magazzino virtuale: {virtual_match.sum()}")
            
            df = df[filter_mask].copy()
            filtered_count = initial_rows - len(df)
            
            if filtered_count > 0:
                st.info(f"🔍 Esclusi {filtered_count} prodotti non-SMALLGIANTSAVAILABLE (selezionati {len(df)} prodotti)")
            
            if len(df) == 0:
                st.error("❌ Nessun prodotto con esattamente 'SMALLGIANTSAVAILABLE' trovato nel file stock")
                return pd.DataFrame()
            
            st.success(f"✅ Filtrati solo prodotti SMALLGIANTSAVAILABLE: {len(df)} prodotti")
        else:
            st.error("❌ Colonne 'Magazzino fisico' o 'Magazzino virtuale' non trovate. Impossibile filtrare.")
            st.info("💡 Le colonne disponibili sono: " + str(list(df.columns)))
            return pd.DataFrame()
        
        # Verifica che almeno SKU e una colonna quantità siano presenti
        if sku_column is None:
            st.error("❌ Colonna SKU non trovata nel file stock")
            return pd.DataFrame()
        
        if available_column is None and in_stock_column is None:
            st.error("❌ Nessuna colonna quantità trovata nel file stock")
            return pd.DataFrame()
        
        st.success(f"✅ Colonne stock mappate:")
        st.success(f"   • SKU: '{sku_column}'")
        if available_column:
            st.success(f"   • Disponibile: '{available_column}'")
        if in_stock_column:
            st.success(f"   • In Stock: '{in_stock_column}'")
        if product_name_column:
            st.success(f"   • Nome Prodotto: '{product_name_column}'")
        if reserved_column:
            st.success(f"   • Prenotate: '{reserved_column}'")
        if incoming_column:
            st.success(f"   • In Arrivo: '{incoming_column}'")
        
        # Crea il mapping delle colonne
        column_mapping = {sku_column: 'sku'}
        
        if available_column:
            column_mapping[available_column] = 'qty_available'
        if in_stock_column:
            column_mapping[in_stock_column] = 'qty_in_stock'
        if product_name_column:
            column_mapping[product_name_column] = 'product_name'
        if reserved_column:
            column_mapping[reserved_column] = 'qty_reserved'
        if incoming_column:
            column_mapping[incoming_column] = 'qty_incoming'
        
        # Rinomina le colonne
        df = df.rename(columns=column_mapping)
        
        # Converti le quantità in numerico
        numeric_columns = ['qty_available', 'qty_in_stock', 'qty_reserved', 'qty_incoming']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calcola la quantità effettiva disponibile
        if 'qty_available' in df.columns:
            df['current_stock'] = df['qty_available']
        elif 'qty_in_stock' in df.columns:
            df['current_stock'] = df['qty_in_stock']
        else:
            df['current_stock'] = 0
        
        # Sottrai le quantità prenotate se disponibili
        if 'qty_reserved' in df.columns:
            df['current_stock'] = df['current_stock'] - df['qty_reserved']
        
        # Aggiungi le quantità in arrivo se disponibili
        if 'qty_incoming' in df.columns:
            df['current_stock'] = df['current_stock'] + df['qty_incoming']
        
        # Assicurati che non sia negativo
        df['current_stock'] = df['current_stock'].clip(lower=0)
        
        # Pulisci i dati
        df = df.dropna(subset=['sku'])
        df['sku'] = df['sku'].astype(str).str.strip()
        
        # NON aggregare duplicati - mantieni solo la prima occorrenza
        st.info("🔄 Rimuovendo prodotti duplicati (mantenendo prima occorrenza)...")
        
        df_before = len(df)
        df = df.drop_duplicates(subset=['sku'], keep='first')
        df_after = len(df)
        
        if df_before > df_after:
            st.success(f"✅ Rimossi {df_before - df_after} duplicati (da {df_before} a {df_after} righe)")
        
        
        st.success(f"✅ Stock processato: {len(df)} prodotti")
        
        return df[['sku', 'current_stock', 'product_name'] + [col for col in df.columns if col.startswith('qty_')]]
        
    except Exception as e:
        st.error(f"Errore nel processamento del file stock: {str(e)}")
        return pd.DataFrame()

def process_excel_data(df):
    """
    Processa il file Excel/CSV con le colonne specifiche per creare il formato richiesto
    """
    try:
        # Debug: mostra le colonne trovate nel file
        st.write("🔍 **Colonne trovate nel file:**", list(df.columns))
        
        # Pulisci i nomi delle colonne (rimuovi spazi extra)
        df.columns = df.columns.str.strip()
        
        # Se la prima colonna contiene il nome del file, probabilmente le intestazioni sono nella prima riga
        if 'export' in str(df.columns[0]).lower() or 'unnamed' in str(df.columns[0]).lower():
            st.info("📋 Rilevate intestazioni nella prima riga. Riprocessando...")
            # La prima riga contiene i veri nomi delle colonne
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            df.columns = df.columns.str.strip()
            st.write("🔍 **Colonne corrette:**", list(df.columns))
        
        # Trova le colonne con ricerca più flessibile
        date_column = find_column(df, ['date', 'data', 'datetime', 'time', 'giorno'])
        movement_column = find_column(df, ['movement', 'tipo', 'type', 'mov'])
        sku_column = find_column(df, ['sku', 'internal', 'codice', 'code', 'item'])
        quantity_column = find_column(df, ['quantity', 'quantit', 'qty', 'pezzi', 'qta'])
        name_column = find_column(df, ['name', 'nome', 'descrizione', 'description', 'product'])
        
        # Verifica che tutte le colonne necessarie siano state trovate
        missing_columns = []
        if date_column is None:
            missing_columns.append("data/date")
        if movement_column is None:
            missing_columns.append("tipo movimento/type")
        if sku_column is None:
            missing_columns.append("SKU/codice")
        if quantity_column is None:
            missing_columns.append("quantità/quantity")
        
        if missing_columns:
            st.error(f"❌ Colonne mancanti: {', '.join(missing_columns)}")
            st.error(f"Colonne disponibili: {list(df.columns)}")
            return pd.DataFrame()
        
        st.success(f"✅ Colonne mappate:")
        st.success(f"   • Data: '{date_column}'")
        st.success(f"   • Movimento: '{movement_column}'")
        st.success(f"   • SKU: '{sku_column}'")
        st.success(f"   • Quantità: '{quantity_column}'")
        if name_column:
            st.success(f"   • Nome: '{name_column}'")
        
        # Rinomina le colonne per standardizzare
        column_mapping = {
            date_column: 'Date',
            movement_column: 'Type',
            sku_column: 'Internal_SKU',
            quantity_column: 'Quantity'
        }
        
        if name_column:
            column_mapping[name_column] = 'Name'
        
        df = df.rename(columns=column_mapping)
        
        # Converti la data con più flessibilità
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        
        # Rimuovi righe con date non valide
        initial_rows = len(df)
        df = df.dropna(subset=['Date'])
        
        if len(df) == 0:
            st.error("❌ Nessuna data valida trovata nel file")
            return pd.DataFrame()
        
        if len(df) < initial_rows:
            st.warning(f"⚠️ {initial_rows - len(df)} righe con date non valide sono state rimosse")
        
        st.success(f"✅ Processate {len(df)} righe con date valide")
        
        # Converti quantità in numerico
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df = df.dropna(subset=['Quantity'])
        
        # Pulisci i valori della colonna Type
        df['Type'] = df['Type'].astype(str).str.lower().str.strip()
        
        # Debug: mostra i tipi di movimento unici
        st.write("🔍 **Tipi di movimento trovati:**", df['Type'].unique())
        
        # Filtra solo i movimenti di uscita (vendite) con più flessibilità
        sales_keywords = ['uscita', 'out', 'vendita', 'sale', 'sold', 'exit', 'ship']
        sales_condition = df['Type'].str.contains('|'.join(sales_keywords), na=False)
        sales_data = df[sales_condition]
        
        st.write(f"📊 **Dati di vendita trovati:** {len(sales_data)} righe")
        
        if len(sales_data) == 0:
            st.warning("⚠️ Nessun movimento di vendita trovato. Verifica i valori nella colonna Type.")
            st.write("Provo ad utilizzare tutti i movimenti...")
            sales_data = df.copy()
        
        # Calcola lo stock finale per ogni prodotto
        stock_data = []
        unique_skus = df['Internal_SKU'].dropna().unique()
        
        for sku in unique_skus:
            sku_data = df[df['Internal_SKU'] == sku].copy()
            sku_data = sku_data.sort_values('Date')
            
            # Calcola stock cumulativo
            stock_balance = 0
            
            for _, row in sku_data.iterrows():
                movement_type = str(row['Type']).lower()
                quantity = row['Quantity']
                
                # Identifica se è entrata o uscita
                if any(keyword in movement_type for keyword in ['entrata', 'in', 'receipt', 'receive', 'carico']):
                    stock_balance += quantity
                    units_sold = 0
                elif any(keyword in movement_type for keyword in sales_keywords):
                    stock_balance -= quantity
                    units_sold = quantity
                else:
                    # Default: considera come uscita se positivo, entrata se negativo
                    if quantity > 0:
                        stock_balance -= quantity
                        units_sold = quantity
                    else:
                        stock_balance += abs(quantity)
                        units_sold = 0
                
                # Ottieni il nome del prodotto
                product_name = row.get('Name', f'Prodotto {sku}')
                
                stock_data.append({
                    'date': row['Date'],
                    'sku': str(sku),
                    'product_name': product_name,
                    'units_sold': units_sold,
                    'on_hand_end': max(0, stock_balance)
                })
        
        # Converti in DataFrame
        result_df = pd.DataFrame(stock_data)
        
        if result_df.empty:
            st.error("❌ Nessun dato processabile trovato")
            return pd.DataFrame()
        
        # Raggruppa per data e SKU per evitare duplicati
        result_df = result_df.groupby(['date', 'sku', 'product_name']).agg({
            'units_sold': 'sum',
            'on_hand_end': 'last'
        }).reset_index()
        
        st.success(f"✅ Dati processati: {len(result_df)} record per {result_df['sku'].nunique()} prodotti")
        
        return result_df
        
    except Exception as e:
        st.error(f"Errore nel processamento dei dati: {str(e)}")
        st.error("Stack trace per debug:")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()

def simple_forecast(ts_data, periods):
    """
    Modello di forecasting semplificato se Prophet non è disponibile
    """
    if len(ts_data) < 2:
        # Se abbiamo pochi dati, usa la media
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
    
    # Calcola trend e stagionalità
    x = np.arange(len(ts_data))
    y = ts_data.values
    
    # Regressione lineare per il trend
    z = np.polyfit(x, y, 1)
    trend_slope = z[0]
    
    # Media mobile per ridurre il rumore
    if len(y) >= 7:
        y_smooth = pd.Series(y).rolling(window=7, min_periods=1).mean().values
    else:
        y_smooth = y
    
    last_value = y_smooth[-1]
    
    # Calcola stagionalità settimanale se abbiamo abbastanza dati
    weekly_pattern = np.ones(7)
    if len(ts_data) >= 14:
        for day in range(7):
            day_values = [y[i] for i in range(len(y)) if i % 7 == day]
            if day_values:
                weekly_pattern[day] = np.mean(day_values) / np.mean(y)
    
    # Genera previsioni
    forecast_data = []
    last_date = ts_data.index[-1]
    
    for i in range(1, periods + 1):
        forecast_date = last_date + timedelta(days=i)
        
        # Trend base
        base_value = last_value + trend_slope * i
        
        # Applica stagionalità settimanale
        day_of_week = (forecast_date.weekday()) % 7
        seasonal_factor = weekly_pattern[day_of_week]
        forecast_value = base_value * seasonal_factor
        
        # Assicurati che non sia negativo
        forecast_value = max(0, forecast_value)
        
        # Calcola intervalli di confidenza
        std_dev = np.std(y) if len(y) > 1 else forecast_value * 0.2
        
        forecast_data.append({
            'ds': forecast_date,
            'yhat': forecast_value,
            'yhat_lower': max(0, forecast_value - 1.96 * std_dev),
            'yhat_upper': forecast_value + 1.96 * std_dev
        })
    
    return pd.DataFrame(forecast_data)

def mark_out_of_stock_days(daily_series, min_run_days=14):
    """
    Restituisce una pd.Series booleana (stesso indice di daily_series)
    True se quel giorno è parte di una run di almeno `min_run_days`
    giorni consecutivi con vendite == 0.
    """
    s = daily_series.copy().sort_index()
    s = s.asfreq('D', fill_value=0)

    zero = (s == 0)
    # id di cambio run
    run_id = (zero != zero.shift()).cumsum()
    # lunghezza di ciascuna run
    run_length = zero.groupby(run_id).transform('sum')
    oos = zero & (run_length >= min_run_days)
    oos.index = s.index
    return oos

def calculate_monthly_average_excluding_oos(daily_series, min_run_days=14, min_weight=0.2):
    """
    Calcola la media mensile delle vendite con ponderazione per mesi con OOS parziale.
    APPROCCIO: prima aggrega le vendite per mese, poi calcola la media ponderata.
    """
    try:
        # Prepara la serie giornaliera
        s = daily_series.copy().sort_index()
        
        # Converti in frequenza giornaliera riempiendo i buchi
        s = s.asfreq('D', fill_value=0)
        
        # Identifica giorni OOS (run di almeno min_run_days con vendite=0)
        oos_days = mark_out_of_stock_days(s, min_run_days=min_run_days)
        
        # Raggruppa per anno-mese
        monthly_data = []
        
        # Per ogni mese nel periodo
        for month_period in s.index.to_period('M').unique():
            # Seleziona i giorni di questo mese
            mask = (s.index.to_period('M') == month_period)
            month_series = s[mask]
            month_oos_series = oos_days[mask]
            
            # Conta giorni
            total_days = len(month_series)
            oos_days_count = int(month_oos_series.sum())
            valid_days_count = total_days - oos_days_count
            
            # IMPORTANTE: somma le vendite di TUTTO il mese (inclusi giorni con 0 vendite non-OOS)
            # Questo è corretto perché vogliamo la performance mensile reale
            month_total_sales = float(month_series.sum())
            
            # Determina il peso del mese
            if oos_days_count >= total_days:
                # Mese completamente OOS: escludi completamente
                weight = 0.0
            elif oos_days_count == 0:
                # Nessun giorno OOS: peso pieno
                weight = 1.0
            else:
                # OOS parziale: peso proporzionale
                # min_weight assicura che mesi parzialmente OOS abbiano comunque un peso minimo
                weight = max(min_weight, valid_days_count / total_days)
            
            monthly_data.append({
                'month': month_period,
                'year': month_period.year,
                'month_num': month_period.month,
                'sales': month_total_sales,
                'weight': weight,
                'total_days': total_days,
                'valid_days': valid_days_count,
                'oos_days': oos_days_count
            })
        
        # Converti in DataFrame per analisi
        df_months = pd.DataFrame(monthly_data)
        
        # Filtra solo mesi con peso > 0 (escludi mesi completamente OOS)
        df_valid = df_months[df_months['weight'] > 0].copy()
        
        if len(df_valid) == 0:
            return {
                'monthly_avg': 0,
                'monthly_avg_simple': 0,
                'total_sales': 0,
                'valid_months': 0,
                'total_months': len(df_months),
                'oos_months': len(df_months[df_months['weight'] == 0]),
                'weighted_months': 0,
                'first_sale_date': None,
                'last_sale_date': None
            }
        
        # CALCOLO 1: Media ponderata (tiene conto dei mesi parzialmente OOS)
        total_weighted_sales = (df_valid['sales'] * df_valid['weight']).sum()
        total_weight = df_valid['weight'].sum()
        monthly_avg_weighted = total_weighted_sales / total_weight if total_weight > 0 else 0
        
        # CALCOLO 2: Media semplice (per confronto/debug)
        monthly_avg_simple = df_valid['sales'].mean()
        
        # Statistiche
        total_sales = float(df_valid['sales'].sum())
        valid_months = len(df_valid)
        total_months = len(df_months)
        oos_months = len(df_months[df_months['weight'] == 0])
        weighted_months = float(total_weight)
        
        # Date primo/ultimo movimento
        first_sale = s[s > 0].index.min() if (s > 0).any() else s.index.min()
        last_sale = s[s > 0].index.max() if (s > 0).any() else s.index.max()
        
        return {
            'monthly_avg': monthly_avg_weighted,
            'monthly_avg_simple': monthly_avg_simple,
            'total_sales': total_sales,
            'valid_months': valid_months,
            'total_months': total_months,
            'oos_months': oos_months,
            'weighted_months': weighted_months,
            'first_sale_date': first_sale,
            'last_sale_date': last_sale,
            'monthly_breakdown': df_months.to_dict('records')  # Per debug dettagliato
        }
        
    except Exception as e:
        import traceback
        return {
            'monthly_avg': 0,
            'monthly_avg_simple': 0,
            'total_sales': 0,
            'valid_months': 0,
            'total_months': 0,
            'oos_months': 0,
            'weighted_months': 0,
            'first_sale_date': None,
            'last_sale_date': None,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
def compute_oos_month_weights(daily_series, min_run_days=14, min_weight=0.2):
    """
    Ritorna un dict {Period('YYYY-MM'): weight} con il peso di ogni mese.
    - Se il mese è completamente OOS -> weight = 0 (escludi)
    - Se il mese ha OOS days ma non completamente -> weight = max(min_weight, 1 - O_days/total_days)
    - Se nessun OOS day -> weight = 1
    """
    s = daily_series.copy().asfreq('D', fill_value=0).fillna(0)
    oos_days_series = mark_out_of_stock_days(s, min_run_days=min_run_days)

    weights = {}
    months = s.index.to_period('M').unique()
    for m in months:
        mask = (s.index.to_period('M') == m)
        if mask.sum() == 0:
            continue
        total_days = int(mask.sum())
        oos_count = int(oos_days_series[mask].sum())
        if oos_count == 0:
            w = 1.0
        elif oos_count >= total_days:
            w = 0.0
        else:
            w = max(min_weight, 1.0 - (oos_count / total_days))
        weights[m] = float(w)
    return weights

def calculate_monthly_seasonality(all_products_data, daily_series_for_weights=None, min_run_days=14, min_weight=0.2):
    """
    Calcola fattori stagionali mensili {1..12: factor} usando media pesata dei mesi.
    - all_products_data: DataFrame con index datetime e colonna 'units_sold'
    - daily_series_for_weights: serie giornaliera del prodotto (opzionale) usata per calcolare pesi OOS
    """
    try:
        a = all_products_data.copy()
        # Assicuriamoci indice datetime
        if 'date' in a.columns:
            a = a.set_index('date')
        a.index = pd.to_datetime(a.index)

        # Calcola vendite mensili (per anno, mese)
        monthly_by_year = a.groupby([a.index.year, a.index.month])['units_sold'].sum().reset_index()
        monthly_by_year.columns = ['year', 'month', 'units_sold']
        if monthly_by_year.empty:
            return {i: 1.0 for i in range(1, 13)}

        # costruisci Period per ciascuna riga
        monthly_by_year['period'] = monthly_by_year.apply(
            lambda r: pd.Period(year=int(r['year']), month=int(r['month']), freq='M'), axis=1)

        # calcoliamo i pesi: preferiamo usare la serie del prodotto se fornita
        if daily_series_for_weights is not None:
            weights_map = compute_oos_month_weights(daily_series_for_weights, min_run_days=min_run_days, min_weight=min_weight)
        else:
            daily_from_all = a['units_sold'].groupby(a.index).sum()
            weights_map = compute_oos_month_weights(daily_from_all, min_run_days=min_run_days, min_weight=min_weight)

        # assegna il peso a ogni riga
        monthly_by_year['weight'] = monthly_by_year['period'].map(weights_map).fillna(1.0)

        # escludi i mesi con peso 0 (completamente OOS)
        monthly_by_year = monthly_by_year[monthly_by_year['weight'] > 0]

        if monthly_by_year.empty:
            return {i: 1.0 for i in range(1, 13)}

        # calcola media pesata per ogni mese numerico (1..12)
        def weighted_avg(group):
            w = group['weight']
            vals = group['units_sold']
            return (vals * w).sum() / w.sum()

        monthly_avg = monthly_by_year.groupby('month').apply(weighted_avg)
        overall_avg = monthly_avg.mean()

        seasonal_factors = (monthly_avg / overall_avg).to_dict()

        # assicura tutti i mesi
        for m in range(1, 13):
            if m not in seasonal_factors or not np.isfinite(seasonal_factors[m]):
                seasonal_factors[m] = 1.0

        return seasonal_factors
    except Exception:
        return {i: 1.0 for i in range(1, 13)}

def calculate_growth_rate(ts_data, months_to_compare=3):
    """
    Calcola il tasso di crescita confrontando gli ultimi mesi con gli stessi mesi dell'anno precedente
    """
    try:
        if len(ts_data) < 30:  # Almeno un mese di dati
            return 0.0
        
        data_df = ts_data.reset_index()
        data_df.columns = ['date', 'sales']
        data_df['month'] = data_df['date'].dt.month
        data_df['year'] = data_df['date'].dt.year
        
        # Raggruppa per anno e mese
        monthly_sales = data_df.groupby(['year', 'month'])['sales'].sum().reset_index()
        
        if len(monthly_sales) < 2:
            return 0.0
        
        # Ultimi mesi disponibili
        recent_months = monthly_sales.tail(months_to_compare)
        
        growth_rates = []
        for _, row in recent_months.iterrows():
            year = row['year']
            month = row['month']
            current_sales = row['sales']
            
            # Cerca stesso mese anno precedente
            previous_year = monthly_sales[(monthly_sales['year'] == year - 1) & 
                                         (monthly_sales['month'] == month)]
            
            if not previous_year.empty:
                prev_sales = previous_year['sales'].iloc[0]
                if prev_sales > 0:
                    growth = (current_sales - prev_sales) / prev_sales
                    growth_rates.append(growth)
        
        if growth_rates:
            return np.mean(growth_rates)
        return 0.0
    except:
        return 0.0

def forecast_with_monthly_seasonality(data, periods, all_products_data=None, current_sku=None):
    """
    Previsione basata su stagionalità mensile e crescita storica
    """
    if len(data) < 2:
        return simple_forecast(data, periods)
    
    try:
        data_df = data.reset_index()
        data_df.columns = ['date', 'sales']
        data_df['month'] = data_df['date'].dt.month
        data_df['year'] = data_df['date'].dt.year
        
        last_date = data_df['date'].max()
        
        # Calcola vendite mensili per questo prodotto
        monthly_sales = data_df.groupby(['year', 'month'])['sales'].sum().reset_index()
        
        # Calcola tasso di crescita
        growth_rate = calculate_growth_rate(data, months_to_compare=3)
        st.info(f"📈 Tasso di crescita rilevato: {growth_rate*100:.1f}%")
        
        # Ottieni stagionalità mensile
        if all_products_data is not None and len(all_products_data) > len(data) * 2:
            seasonal_factors = calculate_monthly_seasonality(all_products_data, current_sku)
            st.info("🌍 Usando stagionalità da tutti i prodotti")
        else:
            # Calcola stagionalità solo da questo prodotto
            if len(monthly_sales) >= 12:
                monthly_avg = monthly_sales.groupby('month')['sales'].mean()
                overall_avg = monthly_avg.mean()
                seasonal_factors = (monthly_avg / overall_avg).to_dict()
                st.info("📊 Usando stagionalità del prodotto specifico")
            else:
                seasonal_factors = {i: 1.0 for i in range(1, 13)}
                st.warning("⚠️ Dati insufficienti per stagionalità, uso andamento recente")

                # ---------------------------
        # CALCOLO PESI OOS E STAGIONALITA'
        # ---------------------------
        # Prepara la serie giornaliera 'data' per i pesi (daily_series)
        daily_series_for_weights = data.copy().sort_index().asfreq('D', fill_value=0).fillna(0)

        # Month weights per il prodotto (Period -> weight)
        month_weights = compute_oos_month_weights(daily_series_for_weights, min_run_days=14, min_weight=0.2)

        # Calcola i seasonal_factors usando i pesi (passiamo daily_series_for_weights così la funzione userà i pesi del prodotto)
        seasonal_factors = calculate_monthly_seasonality(all_products_data if all_products_data is not None else data.to_frame(name='units_sold'),
                                                         daily_series_for_weights=daily_series_for_weights,
                                                         min_run_days=14, min_weight=0.2)

        # Precalcola recent_avg (fallback) una volta per efficienza
        recent_avg = float(data.tail(30).mean()) if len(data) >= 30 else float(data.mean()) if len(data) > 0 else 0.0
        # ---------------------------

        
        # Genera previsioni
        forecast_data = []
        
        for i in range(1, periods + 1):
            forecast_date = last_date + timedelta(days=i)
            forecast_month = forecast_date.month
            forecast_year = forecast_date.year
            
            # Cerca vendite stesso mese anno precedente
            same_month_last_year = monthly_sales[
                (monthly_sales['year'] == forecast_year - 1) & 
                (monthly_sales['month'] == forecast_month)
            ]
            
                        # calcolo period per il mese dell'anno precedente
            period_prev_year = pd.Period(year=forecast_year-1, month=forecast_month, freq='M')

            if not same_month_last_year.empty:
                base_monthly_sales = float(same_month_last_year['sales'].iloc[0])
                adjusted_monthly_sales = base_monthly_sales * (1 + growth_rate)
                # giorni del mese forecast_year/forecast_month
                days_in_month = calendar.monthrange(forecast_year, forecast_month)[1]

                # recupera peso per il mese specifico (se non disponibile assume 1.0)
                month_weight = month_weights.get(period_prev_year, 1.0)

                if month_weight == 0.0:
                    # completamente OOS: fallback su recente con stagionalità
                    seasonal_factor = seasonal_factors.get(forecast_month, 1.0)
                    base_value = recent_avg * seasonal_factor * (1 + growth_rate * 0.5)
                elif month_weight < 1.0:
                    # mix tra valore anno precedente (pesato) e stima recente (meno affidabile)
                    val_from_lastyear = adjusted_monthly_sales / days_in_month
                    val_from_recent = recent_avg * seasonal_factors.get(forecast_month, 1.0) * (1 + growth_rate * 0.5)
                    base_value = month_weight * val_from_lastyear + (1.0 - month_weight) * val_from_recent
                else:
                    # peso pieno (1.0): uso il valore dell'anno precedente
                    base_value = adjusted_monthly_sales / days_in_month
            else:
                # non abbiamo dati per questo mese dell'anno scorso: usa recent_avg * seasonal
                seasonal_factor = seasonal_factors.get(forecast_month, 1.0)
                base_value = recent_avg * seasonal_factor * (1 + growth_rate * 0.5)
            
            # Assicurati non sia negativo
            forecast_value = max(0, base_value)
            
            # Calcola intervalli di confidenza
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
    """
    Restituisce una pd.Series con il valore 'centrale' per ogni riga del forecast:
    - se sono presenti 'yhat_lower' e 'yhat_upper' -> centro = (yhat_lower + yhat_upper) / 2
    - elif 'yhat' presente -> usa 'yhat'
    - else -> serie di zeri (fallback)
    L'indice ritornato sarà la colonna 'ds' convertita in datetime se presente, altrimenti l'indice del DataFrame.
    """
    if forecast_df is None or forecast_df.empty:
        return pd.Series(dtype=float)

    # preferiamo usare la colonna ds come indice se esiste
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
    """
    Effettua previsione usando stagionalità mensile come metodo principale
    """
    # Usa sempre il nuovo modello basato su stagionalità mensile
    return forecast_with_monthly_seasonality(data, periods, all_products_data, current_sku)

def calculate_order_recommendation(forecast_df, current_stock, safety_days, lead_time_days, 
                                 qty_incoming=0, qty_reserved=0, safety_margin=0.1, moq=1):
    """
    Calcola raccomandazioni di ordine con gestione avanzata
    
    Args:
        forecast_df: DataFrame con previsioni
        current_stock: Stock attualmente disponibile
        safety_days: Giorni di scorta di sicurezza
        lead_time_days: Giorni di lead time
        qty_incoming: Quantità già in arrivo
        qty_reserved: Quantità già prenotate/riservate
        safety_margin: Margine di sicurezza aggiuntivo (default 10%)
        moq: Minimum Order Quantity (quantità minima d'ordine)
    """
    if forecast_df is None or forecast_df.empty:
        return 0, "good", {}

    # Domanda prevista per il periodo di lead time + safety stock (usando valore centrale)
    total_days = lead_time_days + safety_days

    # Ottieni la serie centrale (media tra yhat_lower e yhat_upper se disponibili, altrimenti yhat)
    central_series = get_central_forecast_series(forecast_df)

    # Somma dei primi total_days valori centrali
    forecast_demand_central = central_series.head(total_days).sum()

    # Applica il margine di sicurezza sul valore centrale (non sul limite superiore)
    forecast_demand = float(forecast_demand_central) * (1 + float(safety_margin))

    # Calcola stock effettivo disponibile
    # Stock disponibile = stock attuale - prenotato + in arrivo
    effective_stock = current_stock - qty_reserved + qty_incoming

    # Calcola unità da ordinare
    units_needed = forecast_demand - effective_stock
    units_to_order = max(0, units_needed)

    # Arrotonda per eccesso per sicurezza
    units_to_order = math.ceil(units_to_order)

    # Applica MOQ (Minimum Order Quantity)
    if units_to_order > 0 and units_to_order < moq:
        units_to_order = moq

    # Arrotonda al multiplo di MOQ se necessario
    if units_to_order > 0 and moq > 1:
        units_to_order = math.ceil(units_to_order / moq) * moq

    # Determina stato
    if forecast_demand > 0:
        days_of_stock = effective_stock / (forecast_demand / total_days)
    else:
        # Se non c'è domanda prevista, lo stock è sufficiente
        days_of_stock = float('inf')

    if days_of_stock < lead_time_days:
        status = "critical"
    elif days_of_stock < lead_time_days + safety_days:
        status = "warning"
    else:
        status = "good"

    # Info dettagliate per debug
    details = {
        'forecast_demand': forecast_demand,  # centrale con margine
        'forecast_demand_central': float(central_series.head(total_days).sum()),  # centrale senza margine
        'current_stock': current_stock,
        'qty_reserved': qty_reserved,
        'qty_incoming': qty_incoming,
        'effective_stock': effective_stock,
        'days_of_stock': days_of_stock if days_of_stock != float('inf') else 999
    }

    return int(units_to_order), status, details

def create_forecast_chart(historical_data, forecast_data, product_name):
    """
    Crea grafico interattivo delle previsioni
    """
    fig = go.Figure()
    
    # Dati storici
    if len(historical_data) > 0:
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            mode='lines+markers',
            name='Vendite Storiche',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
    
    if not forecast_data.empty:
        # Previsione
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            mode='lines',
            name='Previsione',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Intervallo di confidenza
        fig.add_trace(go.Scatter(
            x=list(forecast_data['ds']) + list(forecast_data['ds'][::-1]),
            y=list(forecast_data['yhat_upper']) + list(forecast_data['yhat_lower'][::-1]),
            fill='toself',
            fillcolor='rgba(255,127,14,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervallo Confidenza',
            showlegend=True
        ))
    
    fig.update_layout(
        title=f'Analisi Vendite e Previsioni - {product_name}',
        xaxis_title='Data',
        yaxis_title='Unità Vendute',
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    return fig

def main():
    # Traduzioni
    translations = load_translations()
    
    # Sidebar per controlli
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/white?text=Small+Giants", width=200)
        
        # Selezione lingua
        language = st.selectbox("🌐 Language / Lingua", ["it", "en"], index=0)
        t = translations[language]
        
        st.markdown("---")
        
        # Upload file movimenti
        uploaded_file = st.file_uploader(
            "📊 " + t['upload_file'] + " (Movimenti)",
            type=['xlsx', 'xls', 'csv'],
            help="File con movimenti: Date, Type, Internal SKU, Quantity, ecc.",
            key="movements_file"
        )
        
        # Upload file stock attuale
        stock_file = st.file_uploader(
            "📦 Carica File Stock Attuale",
            type=['xlsx', 'xls', 'csv'],
            help="File con stock: SKU interno, Q.tà disponibile, Q.tà in stock, ecc.",
            key="stock_file"
        )
        
        # Parametri di forecasting
        st.markdown("### ⚙️ Parametri")
        forecast_days = st.slider(t['forecast_days'], 30, 365, 90)
        safety_stock_days = st.slider(t['safety_stock'], 7, 30, 14)
        lead_time_days = st.slider(t['lead_time'], 1, 30, 7)
        
        st.markdown("### 📦 Parametri Avanzati")
        safety_margin = st.slider("Margine di Sicurezza (%)", 0, 50, 10) / 100
        moq = st.number_input("MOQ - Quantità Minima d'Ordine", min_value=1, value=1, step=1)
    
    # Header principale
    st.markdown(f"<h1 class='main-header'>{t['title']}</h1>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Carica e processa i dati
            with st.spinner(t['processing']):
                # Leggi il file
                if uploaded_file.name.endswith('.csv'):
                    # Prova diversi separatori e encoding
                    try:
                        raw_df = pd.read_csv(uploaded_file, encoding='utf-8', sep=',')
                        if len(raw_df.columns) == 1:
                            uploaded_file.seek(0)
                            raw_df = pd.read_csv(uploaded_file, encoding='utf-8', sep=';')
                        if len(raw_df.columns) == 1:
                            uploaded_file.seek(0)
                            raw_df = pd.read_csv(uploaded_file, encoding='utf-8', sep='\t')
                    except:
                        uploaded_file.seek(0)
                        raw_df = pd.read_csv(uploaded_file, encoding='latin-1', sep=';')
                else:
                    raw_df = pd.read_excel(uploaded_file)
                
                st.write(f"📁 **File caricato:** {uploaded_file.name}")
                st.write(f"📊 **Righe totali:** {len(raw_df)}")
                
                # Processa i dati
                df = process_excel_data(raw_df)
                
                if df.empty:
                    st.error("❌ Nessun dato valido trovato nel file caricato.")
                    return
        
            # Selezione prodotto
            available_products = df['sku'].unique()
            product_options = []
            
            for sku in available_products:
                product_name = df[df['sku']==sku]['product_name'].iloc[0]
                product_options.append(f"{sku} - {product_name}")
            
            selected_product_display = st.selectbox(t['select_product'], product_options)
            selected_sku = selected_product_display.split(' - ')[0]
            
            # Filtra dati per prodotto selezionato
            product_data = df[df['sku'] == selected_sku].copy()
            
            if product_data.empty:
                st.warning(t['no_data'])
                return
            
            # Prepara dati per l'analisi
            product_data = product_data.sort_values('date')
            product_data.set_index('date', inplace=True)
            
            # Raggruppa per data (in caso di duplicati)
            daily_sales = product_data.groupby('date')['units_sold'].sum()
            
            # Ottieni stock attuale dal file stock se disponibile
            current_stock = 0
            qty_reserved = 0
            qty_incoming = 0
            
            if stock_file is not None:
                try:
                    # Leggi il file stock
                    if stock_file.name.endswith('.csv'):
                        try:
                            stock_raw_df = pd.read_csv(stock_file, encoding='utf-8', sep=',')
                            if len(stock_raw_df.columns) == 1:
                                stock_file.seek(0)
                                stock_raw_df = pd.read_csv(stock_file, encoding='utf-8', sep=';')
                        except:
                            stock_file.seek(0)
                            stock_raw_df = pd.read_csv(stock_file, encoding='latin-1', sep=';')
                    else:
                        stock_raw_df = pd.read_excel(stock_file)
                    
                    # Processa il file stock
                    stock_df = process_stock_file(stock_raw_df)
                    
                    if not stock_df.empty:
                        # Cerca il prodotto corrente
                        product_stock = stock_df[stock_df['sku'].astype(str).str.strip() == str(selected_sku).strip()]
                        
                        if not product_stock.empty:
                            current_stock = float(product_stock['current_stock'].iloc[0])
                            if 'qty_reserved' in product_stock.columns:
                                qty_reserved = float(product_stock['qty_reserved'].iloc[0])
                            if 'qty_incoming' in product_stock.columns:
                                qty_incoming = float(product_stock['qty_incoming'].iloc[0])
                            
                            st.success(f"✅ Stock trovato nel file: {current_stock:,.0f} unità")
                            if qty_reserved > 0:
                                st.info(f"📋 Quantità prenotate: {qty_reserved:,.0f}")
                            if qty_incoming > 0:
                                st.info(f"📦 Quantità in arrivo: {qty_incoming:,.0f}")
                        else:
                            st.warning(f"⚠️ SKU {selected_sku} non trovato nel file stock. Uso l'ultimo valore dai movimenti.")
                            current_stock = product_data['on_hand_end'].iloc[-1] if len(product_data) > 0 else 0
                except Exception as e:
                    st.warning(f"⚠️ Errore lettura file stock: {str(e)}. Uso l'ultimo valore dai movimenti.")
                    current_stock = product_data['on_hand_end'].iloc[-1] if len(product_data) > 0 else 0
            else:
                # Se non c'è file stock, usa l'ultimo valore dai movimenti
                current_stock = product_data['on_hand_end'].iloc[-1] if len(product_data) > 0 else 0
                st.info("💡 File stock non caricato. Usando l'ultimo valore dai movimenti di magazzino.")
            

            # Effettua previsione
            with st.spinner("Generando previsioni..."):
                # Prepara dati di tutti i prodotti per stagionalità
                all_products_sales = df.copy()
                all_products_sales = all_products_sales.set_index('date')
                
                forecast = forecast_with_prophet(daily_sales, forecast_days, 
                                                 all_products_data=all_products_sales,
                                                 current_sku=selected_sku)
                
            # Calcola media mensile storica escludendo OOS
            monthly_stats = calculate_monthly_average_excluding_oos(daily_sales, min_run_days=14)
            
            # Layout a colonne - aggiungiamo una quarta colonna
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label=t['current_stock'],
                    value=f"{current_stock:,}",
                    delta=None
                )
            
            with col2:
                if not forecast.empty:
                    # Usa la serie centrale per la metrica mostrata (coerente con il calcolo degli ordini)
                    central_series_ui = get_central_forecast_series(forecast)
                    predicted_demand = int(central_series_ui.head(lead_time_days + safety_stock_days).sum())
                    st.metric(
                        label=t['predicted_demand'],
                        value=f"{predicted_demand:,}",
                        delta=None
                    )
                else:
                    predicted_demand = 0
            with col3:
                st.metric(
                    label="📊 Media Mensile Storica",
                    value=f"{monthly_stats['monthly_avg']:,.0f}",
                    delta=None,
                    help=f"Calcolata su {monthly_stats['valid_months']} mesi validi (pesati: {monthly_stats['weighted_months']:.1f}), esclusi {monthly_stats['oos_months']} mesi completamente OOS"
                )
                
            with col4:
                units_to_order, status, details = calculate_order_recommendation(
                    forecast, current_stock, safety_stock_days, lead_time_days,
                    qty_incoming=qty_incoming, qty_reserved=qty_reserved,
                    safety_margin=safety_margin, moq=moq
                )
                
                status_colors = {
                    "critical": "#ff4444",
                    "warning": "#ffaa00", 
                    "good": "#00aa44"
                }
                
                st.metric(
                    label=t['units_to_order'],
                    value=f"{units_to_order:,}",
                    delta=None
                )
                
                # Stato con colore
                status_text = t[status]
                st.markdown(f"<p style='color: {status_colors[status]}; font-weight: bold; text-align: center;'>"
                           f"Status: {status_text}</p>", unsafe_allow_html=True)
            
            # Mostra dettagli calcolo in un expander
            with st.expander("🔍 Dettagli Calcolo Ordine"):
                st.markdown("#### 📊 Statistiche Storiche")
                st.write(f"**Media Mensile (ponderata):** {monthly_stats['monthly_avg']:,.1f} unità/mese")
                st.write(f"**Periodo analizzato:** da {monthly_stats['first_sale_date'].strftime('%Y-%m-%d') if monthly_stats['first_sale_date'] else 'N/A'} "
                        f"a {monthly_stats['last_sale_date'].strftime('%Y-%m-%d') if monthly_stats['last_sale_date'] else 'N/A'}")
                st.write(f"**Mesi totali nel periodo:** {monthly_stats['total_months']}")
                st.write(f"**Mesi validi (con vendite):** {monthly_stats['valid_months']}")
                st.write(f"**Mesi completamente OOS (esclusi):** {monthly_stats['oos_months']}")
                st.write(f"**Mesi ponderati equivalenti:** {monthly_stats['weighted_months']:.1f}")
                st.write(f"**Vendite totali (mesi validi):** {monthly_stats['total_sales']:,.0f} unità")
                st.write(f"**Media semplice (no pesi):** {monthly_stats.get('monthly_avg_simple', 0):,.1f} unità/mese")
                
                # Debug dettagliato mesi
                if st.checkbox("🔍 Mostra dettaglio per mese"):
                    if 'monthly_breakdown' in monthly_stats:
                        breakdown_df = pd.DataFrame(monthly_stats['monthly_breakdown'])
                        breakdown_df['month_label'] = breakdown_df.apply(
                            lambda x: f"{x['year']}-{x['month_num']:02d}", axis=1
                        )
                        breakdown_df = breakdown_df[['month_label', 'sales', 'weight', 'oos_days', 'valid_days', 'total_days']]
                        breakdown_df.columns = ['Mese', 'Vendite', 'Peso', 'Giorni OOS', 'Giorni Validi', 'Giorni Totali']
                        st.dataframe(breakdown_df, use_container_width=True)
                        
                        st.write(f"**Verifica calcolo:**")
                        st.write(f"Somma vendite mesi validi: {monthly_stats['total_sales']:,.0f}")
                        st.write(f"Numero mesi validi: {monthly_stats['valid_months']}")
                        st.write(f"Media se dividessi semplicemente: {monthly_stats['total_sales']/monthly_stats['valid_months']:,.0f}")
                st.markdown("---")
                st.markdown("#### 💼 Dettagli Ordine")
                st.write(f"**Domanda Prevista ({lead_time_days + safety_stock_days} giorni):** {details['forecast_demand']:,.1f} unità")
                # Mostra anche il valore centrale senza margine
                if 'forecast_demand_central' in details:
                    st.write(f"**Domanda Prevista (centrale, senza margine):** {details['forecast_demand_central']:,.1f} unità")
                st.write(f"**Stock Fisico:** {details['current_stock']:,.0f} unità")
                st.write(f"**Quantità Prenotate:** {details['qty_reserved']:,.0f} unità")
                st.write(f"**Quantità in Arrivo:** {details['qty_incoming']:,.0f} unità")
                st.write(f"**Stock Effettivo Disponibile:** {details['effective_stock']:,.1f} unità")
                st.write(f"**Giorni di Copertura:** {details['days_of_stock']:.1f} giorni")
                st.write(f"**Margine di Sicurezza Applicato:** {safety_margin*100:.0f}%")
                st.write(f"**MOQ Applicato:** {moq} unità")
                
                # Calcola quando finirà lo stock
                if details['forecast_demand'] > 0:
                    avg_daily_demand = details['forecast_demand'] / (lead_time_days + safety_stock_days)
                    stockout_days = details['effective_stock'] / avg_daily_demand if avg_daily_demand > 0 else 999
                    stockout_date = datetime.now() + timedelta(days=int(stockout_days))
                    st.warning(f"⚠️ **Previsto esaurimento scorte:** {stockout_date.strftime('%Y-%m-%d')} (tra {int(stockout_days)} giorni)")
            
            # Grafico principale
            st.markdown("---")
            
            if len(daily_sales) > 0:
                product_name = product_data['product_name'].iloc[0] if 'product_name' in product_data.columns else selected_sku
                chart = create_forecast_chart(daily_sales, forecast, product_name)
                st.plotly_chart(chart, use_container_width=True)
            
            # Tabelle dettagliate
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(t['historical_data'])
                if len(daily_sales) > 0:
                    hist_df = daily_sales.tail(10).reset_index()
                    hist_df['date'] = hist_df['date'].dt.strftime('%Y-%m-%d')
                    hist_df.columns = ['Data', 'Vendite']
                    st.dataframe(hist_df, use_container_width=True)
                else:
                    st.info(t['no_data'])
            
            with col2:
                st.subheader(t['forecast'])
                if not forecast.empty:
                    forecast_display = forecast.head(10).copy()
                    forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
                    forecast_display['yhat'] = forecast_display['yhat'].round().astype(int)
                    forecast_display = forecast_display[['ds', 'yhat']].copy()
                    forecast_display.columns = ['Data', 'Previsione']
                    st.dataframe(forecast_display, use_container_width=True)
                else:
                    st.info("Dati insufficienti per la previsione")
        
        except Exception as e:
            st.error(f"Errore nell'elaborazione: {str(e)}")
            import traceback
            st.error("Stack trace completo:")
            st.code(traceback.format_exc())
    
    else:
        # Pagina di benvenuto
        st.markdown("""
        ## 👋 Benvenuto nel Sistema di Previsione Inventario
        
        ### 📁 Come utilizzare l'app:
        1. **Carica il tuo file Excel/CSV** nella barra laterale (movimenti magazzino)
        2. **Carica il file stock attuale** (opzionale ma consigliato)
        3. **Seleziona un prodotto** dalla lista
        4. **Regola i parametri** di previsione e ordine
        5. **Analizza** le raccomandazioni di riordino
        
        ### 📊 Cosa ottieni:
        - **Previsioni** delle vendite future basate su AI
        - **Raccomandazioni** di riordino personalizzate con MOQ
        - **Gestione avanzata** stock: prenotati, in arrivo, disponibili
        - **Grafici interattivi** delle vendite e previsioni
        - **Analisi** dello stato dell'inventario
        - **Data prevista** esaurimento scorte
        
        ### 📋 Formato file movimenti richiesto:
        Il file deve contenere le colonne:
        - `Date` (o simile) - Data del movimento
        - `Type` (o simile) - Tipo movimento (entrata/uscita)
        - `Internal SKU` (o simile) - Codice prodotto
        - `Quantity` (o simile) - Quantità
        - `Name` (opzionale) - Nome del prodotto
        
        ### 📦 Formato file stock attuale (opzionale):
        Il file deve contenere le colonne:
        - `SKU interno` (o simile) - Codice prodotto
        - `Q.tà disponibile` o `Q.tà in stock` - Quantità attuale
        - `Q.tà prenotate` (opzionale) - Quantità riservate
        - `In attesa di ricezione` (opzionale) - Quantità in arrivo
        
        **Nota:** L'app riconoscerà automaticamente le colonne anche con nomi simili!
        
        ### 🎯 Funzionalità Avanzate:
        - **Arrotondamento per eccesso**: ordini sempre sufficienti
        - **MOQ**: rispetta quantità minime d'ordine
        - **Margine di sicurezza**: buffer aggiuntivo configurabile
        - **Stock effettivo**: considera prenotati e in arrivo
        - **Previsioni conservative**: usa limiti superiori intervallo confidenza
        """)
# --- AUTH ---
import os

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

