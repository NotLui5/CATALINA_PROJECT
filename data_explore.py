import requests
import pandas as pd
import xlwings as xw
import regex as re

def get_yellow_rows(file_path, sheet_name=None):
    """
    Extrae filas completas donde la primera celda (columna A) tiene fondo amarillo.
    
    Args:
        file_path (str): Ruta al archivo Excel
        sheet_name (str, optional): Nombre de la hoja. Defaults to active sheet.
    
    Returns:
        list: Lista de filas donde la primera celda es amarilla
    """
    app = xw.App(visible=False)
    wb = None
    try:
        wb = xw.Book(file_path, encoding = 'utf-8')
        sheet = wb.sheets[sheet_name] if sheet_name else wb.sheets[0]
        
        yellow_rows = []
        yellow_rgb = (255, 255, 0)  # Amarillo estándar
        
        # Determinar el rango con datos
        last_row = sheet.used_range.last_cell.row
        last_col = sheet.used_range.last_cell.column
        
        print(f"Procesando {last_row} filas...")
        
        for row in range(1, last_row + 1):
            try:
                cell = sheet.range(f'A{row}')
                
                # Verificar si la celda tiene color y si es amarillo
                if hasattr(cell, 'color'):
                    if cell.color == yellow_rgb:
                        # Obtener toda la fila y filtrar valores None
                        row_values = sheet.range((row, 1), (row, last_col)).value
                        clean_row = [value for value in row_values if value is not None]
                        if clean_row:  # Solo añadir si hay valores
                            yellow_rows.append(clean_row)
            except Exception as e:
                # Continuar con la siguiente fila si hay error
                continue
                
        return yellow_rows
        
    except Exception as e:
        print(f"Error grave al procesar el archivo: {str(e)}")
        return []
        
    finally:
        # Limpieza segura
        if wb is not None:
            wb.close()
        app.quit()

def get_yellow_cells(file_path, sheet_name=None):
    """
    Extrae solo las celdas de la columna A que tienen fondo amarillo.
    
    Args:
        file_path (str): Ruta al archivo Excel
        sheet_name (str, optional): Nombre de la hoja. Si es None, usa la primera hoja.
    
    Returns:
        list: Lista de valores de celdas amarillas en la columna A
    """
    app = xw.App(visible=False)
    wb = None
    yellow_cells = []
    
    try:
        wb = xw.Book(file_path)
        sheet = wb.sheets[sheet_name] if sheet_name else wb.sheets[0]
        
        # Amarillo estándar (RGB)
        yellow_rgb = (255, 255, 0)  
        
        # Obtener el último renglón con datos
        last_row = sheet.range('A' + str(sheet.cells.last_cell.row)).end('up').row
        
        for row in range(1, last_row + 1):
            cell = sheet.range(f'A{row}')
            try:
                # Verificar si la celda tiene color amarillo
                if hasattr(cell, 'color') and cell.color == yellow_rgb:
                    yellow_cells.append(cell.value)
            except:
                # Si hay error, continuar con la siguiente celda
                continue
                
        return yellow_cells
        
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        return []
        
    finally:
        # Cerrar todo correctamente
        if wb is not None:
            wb.close()
        app.quit()

# Ejemplo de uso
celdas_amarillas = get_yellow_rows('./code_book/Libro de codigos.xlsx') ############
xyr = pd.DataFrame(celdas_amarillas)
print(celdas_amarillas)
xyr[0] = [x.replace("[", "").replace("]", "").replace(" ", "") for x in xyr[0]]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json',
    'Content-Type': 'application/x-www-form-urlencoded',
}

data = {
    'token': '3ECC702833A265FFDBFB3DD4E1911E7A',
    'content': 'exportFieldNames',
    'format': 'json',
    'field': '', # Get fields name
    'returnFormat': 'json'
}
r1 = requests.post('https://redcap.solcaquito.org.ec/api/',
                  data=data,
                  headers=headers,
                  verify=False) # Disable SSL verification if needed
print("*******************************************************\n")
d1 = r1.json() # Get the fields name
code_book = pd.DataFrame(d1)
with pd.ExcelWriter("./code_book/Libro de codigos redcap.xlsx", engine='openpyxl') as writer:
    code_book.to_excel(writer, sheet_name="code_book_redcap", index=False)

f = 0
fields = {}
for var in xyr[0]:
    if any(var in item.values() for item in d1):
    # if not any(var in item.values() for item in d1):
        fields[f'fields[{f}]'] = var
        f += 1  
print(fields)

data = {
    'token': '3ECC702833A265FFDBFB3DD4E1911E7A', 
    'content': 'record',
    'action': 'export',
    'format': 'json',
    'type': 'eav',
    'csvDelimiter': '',
    # 'forms[0]': 'formulario_1_caratersticas_base',############
    # 'forms[0]': 'formulario_2_tratamientos',############
    # 'forms[0]': 'formulario_3_seguimiento', ############
    'rawOrLabel': 'label',
    'rawOrLabelHeaders': 'label',
    'exportCheckboxLabel': 'false',
    'exportSurveyFields': 'false',
    'exportDataAccessGroups': 'false',
    'returnFormat': 'json'
}
data.update(fields)


r2 = requests.post(
    'https://redcap.solcaquito.org.ec/api/',
    data=data,
    headers=headers,
    verify=False  # Disable SSL verification if needed
)

# # print('HTTP Status:', r.status_code)
print("*******************************************************\n")
# print(r.text)
# print(r.headers)
# print(r.content) # not in utf
# print(r.apparent_encoding) # is in ascii
r2_j = r2.json()
df2 = pd.DataFrame(r2_j)

df2 = df2.drop(columns='redcap_event_name')
transposed_df = df2.pivot_table(
    index='record', 
    columns='field_name', 
    values='value',
    aggfunc='first' # Takes the first value if duplicates exist
).reset_index()
name_mapping = dict(zip(xyr[0], xyr[1]))
transposed_df = transposed_df.rename(columns=name_mapping)
transposed_df = transposed_df.drop(columns='record')
# transposed_df.to_csv("Registers_catalina_part2.csv", encoding='utf-8-sig')
hospitals_code = {'HEE': '9-', 'SOLCA-UIO': '10-', 'ITECC':'24-', 'SOLCA-GYE':'11-',
                  'SOLCA-CUE':'12-', 'HOSP-AMBATO': '13-', 'HOSP-JOSE-GONZ-MEX':'23-', 
                  'CLIN-ESPEC-MEX': '16-', 'HOSP-EDGARDO-PERU':'21-', 'HOSP-MANUEL-QUINT-URUGUAY': '22-'}

df_ord = pd.read_excel("./gold_order/Orden de variables 3.xlsx") ############
df_ord2 = pd.read_excel("./gold_order/Orden de variables form 3.xlsx") ############
# seen = set()
# col_order = [
#     re.sub(r'\.\d+$', '', col) 
#     for col in df_ord.columns 
#     if not str(col).startswith('Unnamed:') 
#     and not (re.sub(r'\.\d+$', '', col) in seen or seen.add(re.sub(r'\.\d+$', '', col)))
# ]
# col_order.remove('record')
# col_order.remove('ETNNIA_TEXT')
# col_order.remove('TiRADS_DESCRIBE')
# col_order_2 = [col for col in transposed_df.columns if col not in col_order]
col_order1  = list(df_ord.columns) ############
col_order2  = list(df_ord2.columns) ############
col_order3 = col_order1 + col_order2  
col_order = [col for col in col_order3 if col in transposed_df.columns]
df_ord_def = pd.read_excel("./gold_order/Total 2.xlsx") ############
# col_order = [col for col in df_ord_def if col in transposed_df.columns]
col_order = [col for col in transposed_df.columns if col not in df_ord_def.columns]
with open("./data_extracted/var_no_included.txt", "w", encoding="utf-8") as file:
    for col in col_order:
        file.write(f"{col} \n")

# var_order = [col for col in df_ord_def if col not in transposed_df.columns]
# with open("./data_extracted/var_no_included.txt", "w", encoding="utf-8") as file:
#     for col in var_order:
#         file.write(f"{col} \n")
transposed_df = transposed_df[col_order]
# transposed_df = transposed_df[col_order + extra_cols] ############

# def export_to_excel_with_sheets(basedf, hospitals_code, output_file="./data_extracted/Registers_catalina_part11_wforms3.xlsx"): ##########
#     with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
#         for hospital_name, prefix in hospitals_code.items():
#             filtered_df = basedf[basedf['CODIGO ID'].str.startswith(prefix, na=False)]
#             # filtered_df = basedf[basedf['ID paciente '].str.startswith(prefix, na=False)]
#             filtered_df.to_excel(
#                 writer, 
#                 sheet_name=f"{hospital_name}_data", 
#                 index=False
#             )
                        
#             summary_df = filtered_df.describe(include='all')
#             summary_df.to_excel(
#                 writer, 
#                 sheet_name=f"{hospital_name}_stats", 
#                 index=True
#             )


# export_to_excel_with_sheets(transposed_df, hospitals_code)
