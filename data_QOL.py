import requests
import pandas as pd
import xlwings as xw
import regex as re

def get_yellow_rows(file_path, sheet_name=None):
    """
    Extrae filas completas donde la primera celda (columna A) tiene fondo amarillo.
    """
    app = xw.App(visible=False)
    wb = None
    try:
        wb = xw.Book(file_path, encoding='utf-8')
        sheet = wb.sheets[sheet_name] if sheet_name else wb.sheets[0]
        
        yellow_rows = []
        yellow_rgb = (255, 255, 0)
        
        last_row = sheet.used_range.last_cell.row
        last_col = sheet.used_range.last_cell.column
        
        print(f"Procesando {last_row} filas...")
        
        for row in range(1, last_row + 1):
            try:
                cell = sheet.range(f'A{row}')
                
                if hasattr(cell, 'color'):
                    if cell.color == yellow_rgb:
                        row_values = sheet.range((row, 1), (row, last_col)).value
                        clean_row = [value for value in row_values if value is not None]
                        if clean_row:
                            yellow_rows.append(clean_row)
                            print(f"Fila {row}: {len(clean_row)} columnas")
            except Exception as e:
                continue
                
        print(f"Total de filas amarillas encontradas: {len(yellow_rows)}")
        return yellow_rows
        
    except Exception as e:
        print(f"Error grave al procesar el archivo: {str(e)}")
        return []
        
    finally:
        if wb is not None:
            wb.close()
        app.quit()

# Extraer datos del Excel
celdas_amarillas = get_yellow_rows('code_book/code_book/Libro de codigos_actualizado.xlsx.xlsx')
xyr = pd.DataFrame(celdas_amarillas)
print(f"Shape del DataFrame de filas amarillas: {xyr.shape}")

# ============ CORRECCIÓN CRÍTICA ============
# Extraer SOLO los nombres de variables REDCap (los que están en la PRIMERA COLUMNA)
# NO extraer etiquetas como nombres de variables

redcap_field_names = []

for row in celdas_amarillas:
    if len(row) > 0:
        # Tomar SOLO la primera columna que contiene el nombre REDCap
        var_name = str(row[0]).strip()
        
        # Limpiar corchetes y espacios
        var_name = var_name.replace("[", "").replace("]", "").strip()
        
        # Ignorar si es vacío o si parece una etiqueta (tiene espacios, números, etc)
        if var_name and len(var_name) > 1:
            # Verificar que parece un nombre de variable REDCap (solo letras, números, guiones bajos)
            if re.match(r'^[a-zA-Z0-9_]+$', var_name):
                if var_name not in redcap_field_names:
                    redcap_field_names.append(var_name)
                    print(f"  Variable REDCap encontrada: [{var_name}]")

print(f"\n=== TOTAL DE VARIABLES REDCAP ENCONTRADAS: {len(redcap_field_names)} ===")
print("Primeras 20 variables:", redcap_field_names[:20])

# Crear mapping SOLO de nombres REDCap a etiquetas (para renombrar después)
name_mapping = {}
for row in celdas_amarillas:
    if len(row) >= 2:
        redcap_name = str(row[0]).replace("[", "").replace("]", "").strip()
        if len(row) > 1 and row[1] and redcap_name:
            label = str(row[1]).strip()
            # Solo mapear si el nombre REDCap es válido
            if re.match(r'^[a-zA-Z0-9_]+$', redcap_name):
                name_mapping[redcap_name] = label
                print(f"Mapping: {redcap_name} -> {label}")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json',
    'Content-Type': 'application/x-www-form-urlencoded',
}

# Obtener los nombres de campos reales de REDCap
data_fields = {
    'token': '3ECC702833A265FFDBFB3DD4E1911E7A',
    'content': 'exportFieldNames',
    'format': 'json',
    'field': '',
    'returnFormat': 'json'
}

r1 = requests.post('https://redcap.solcaquito.org.ec/api/',
                   data=data_fields,
                   headers=headers,
                   verify=False)

print("\n*******************************************************\n")
d1 = r1.json()
code_book = pd.DataFrame(d1)

with pd.ExcelWriter("./code_book/Libro de codigos redcap.xlsx", engine='openpyxl') as writer:
    code_book.to_excel(writer, sheet_name="code_book_redcap", index=False)

print("\nLISTA DE VARIABLES REALES DE REDCAP:")
redcap_vars = code_book['original_field_name'].tolist()
print(f"Total en REDCap: {len(redcap_vars)}")

# Filtrar solo las variables que existen en REDCap
fields = {}
f = 0
missing_vars = []

for var in redcap_field_names:
    if var in redcap_vars:
        fields[f'fields[{f}]'] = var
        f += 1
    else:
        missing_vars.append(var)

print(f"\n=== VARIABLES A SOLICITAR A REDCAP: {len(fields)} ===")
print("Primeras 30 variables solicitadas:")
for i, (key, value) in enumerate(list(fields.items())[:30]):
    print(f"  {key}: {value}")

if missing_vars:
    print(f"\nVariables NO encontradas en REDCap ({len(missing_vars)}):")
    print(missing_vars[:20])

# Configuración para exportar registros
data_export = {
    'token': '3ECC702833A265FFDBFB3DD4E1911E7A',
    'content': 'record',
    'action': 'export',
    'format': 'json',
    'type': 'eav',
    'csvDelimiter': '',
    'rawOrLabel': 'label',
    'rawOrLabelHeaders': 'label',
    'exportCheckboxLabel': 'false',
    'exportSurveyFields': 'false',
    'exportDataAccessGroups': 'false',
    'returnFormat': 'json'
}

# Agregar formularios
form_idx = 0
forms_list = [
    'formulario_1_caratersticas_base',
    'formulario_2a_post_tiroidectomia',
    'formulario_2b',
    'formulario_2c',
    'formulario_3_seguimiento'
]

for form in forms_list:
    data_export[f'forms[{form_idx}]'] = form
    form_idx += 1

# Agregar TODAS las variables encontradas
data_export.update(fields)

print(f"\nTotal de campos solicitados a REDCap: {len(fields)}")
print(f"Total de formularios solicitados: {form_idx}")

try:
    r2 = requests.post(
        'https://redcap.solcaquito.org.ec/api/',
        data=data_export,
        headers=headers,
        verify=False,
        timeout=120
    )
    
    print(f"\nStatus Code: {r2.status_code}")
    
    if r2.status_code == 200:
        r2_j = r2.json()
        print(f"Registros recibidos (formato EAV): {len(r2_j)}")
        
        if len(r2_j) > 0:
            df2 = pd.DataFrame(r2_j)
            
            if 'redcap_event_name' in df2.columns:
                df2 = df2.drop(columns='redcap_event_name')
            
            # Transformar de formato EAV a ancho
            print("Transformando datos de formato EAV a ancho...")
            transposed_df = df2.pivot_table(
                index='record', 
                columns='field_name', 
                values='value',
                aggfunc='first'
            ).reset_index()
            
            print(f"DataFrame transformado: {transposed_df.shape[0]} filas, {transposed_df.shape[1]} columnas")
            
            # Verificar que datodem04 está en los datos
            if 'datodem04' in transposed_df.columns:
                print(f"✅ Variable datodem04 (AGE_AT_DIAGNOSIS) ENCONTRADA en los datos")
            else:
                print(f"❌ Variable datodem04 NO encontrada en los datos")
            
            # Renombrar columnas según el mapping
            columns_to_rename = {}
            for col in transposed_df.columns:
                if col in name_mapping:
                    columns_to_rename[col] = name_mapping[col]
            
            if columns_to_rename:
                transposed_df = transposed_df.rename(columns=columns_to_rename)
                print(f"Renombradas {len(columns_to_rename)} columnas")
            
            print(f"\n=== COLUMNAS FINALES EN DATAFRAME: {len(transposed_df.columns)} ===")
            
            # Verificar si AGE_AT_DIAGNOSIS está después del renombrado
            if 'Edad al diagnóstico' in transposed_df.columns:
                print(f"✅ AGE_AT_DIAGNOSIS ENCONTRADA después del renombrado")
            else:
                print(f"❌ AGE_AT_DIAGNOSIS NO encontrada después del renombrado")
            
            # # Guardar CSV
            # transposed_df.to_csv("./data_extracted/todos_los_datos_completos.csv", 
            #                    encoding='utf-8-sig', index=False)
            # print("Archivo CSV guardado con todos los datos")
            
            # ============ EXPORTACIÓN A EXCEL CON HOJAS DE DATOS Y ESTADÍSTICAS ============
            hospitals_code = {
                'HEE': '9-', 'SOLCA-UIO': '10-', 'ITECC': '24-', 'SOLCA-GYE': '11-',
                'SOLCA-CUE': '12-', 'HOSP-AMBATO': '13-', 'HOSP-JOSE-GONZ-MEX': '23-', 
                'CLIN-ESPEC-MEX': '16-', 'HOSP-EDGARDO-PERU': '21-', 'HOSP-MANUEL-QUINT-URUGUAY': '22-'
            }
            
            def export_to_excel_with_sheets(basedf, hospitals_code, output_file="./data_extracted/Registers_catalina_part11.6_wforms3.xlsx"):
                """
                Exporta a Excel con:
                - Por cada hospital: Hoja "[hospital]_data" con sus registros
                - Por cada hospital: Hoja "[hospital]_stats" con estadísticas descriptivas
                """
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    for hospital_name, prefix in hospitals_code.items():
                        if 'codigo_id' in basedf.columns:
                            filtered_df = basedf[basedf['codigo_id'].str.startswith(prefix, na=False)]
                            
                            if not filtered_df.empty:
                                # Hoja de datos del hospital
                                filtered_df.to_excel(
                                    writer, 
                                    sheet_name=f"{hospital_name}_data", 
                                    index=False
                                )
                                print(f"  ✅ {hospital_name}_data: {len(filtered_df)} registros")
                                
                                # Hoja de estadísticas del hospital
                                summary_df = filtered_df.describe(include='all')
                                summary_df.to_excel(
                                    writer, 
                                    sheet_name=f"{hospital_name}_stats", 
                                    index=True
                                )
                                print(f"  ✅ {hospital_name}_stats: estadísticas calculadas")
                            else:
                                print(f"  ⚠️ {hospital_name}: Sin registros")
                        else:
                            print(f"  ❌ Columna 'CODIGO ID' no encontrada en el dataframe")
                            break
                
                print(f"\n✅ Archivo Excel guardado: {output_file}")
            
            # Ejecutar la exportación
            export_to_excel_with_sheets(transposed_df, hospitals_code)
            
        else:
            print("No se recibieron datos de REDCap")
    else:
        print(f"Error en la solicitud: {r2.status_code}")
        print(r2.text[:500])
        
except Exception as e:
    print(f"Error al procesar la respuesta de REDCap: {e}")
    import traceback
    traceback.print_exc()