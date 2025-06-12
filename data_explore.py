import requests
import pandas as pd
import xlwings as xw

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
celdas_amarillas = get_yellow_rows('Libro de codigos.xlsx')
xyr = pd.DataFrame(celdas_amarillas)
# print(celdas_amarillas)

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

f = 0
fields = {}
for var in xyr[0]:
    if any(var in item.values() for item in d1):
    # if not any(var in item.values() for item in d1):
        fields[f'fields[{f}]'] = var
        f += 1  
fields1 = {}
# print(fields)

data = {
    'token': '3ECC702833A265FFDBFB3DD4E1911E7A',
    'content': 'record',
    'action': 'export',
    'format': 'json',
    'type': 'eav',
    'csvDelimiter': '',
    # 'forms[0]': 'formulario_1_caratersticas_base',
    # 'forms[1]': 'formulario_2_tratamientos',
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

transposed_df.to_csv("Registers_catalina_part1.csv", encoding='utf-8-sig')
# print(transposed_df)
print("*******************************************************\n")
desc = transposed_df.describe()
desc.to_csv("summary_register1.csv", encoding='utf-8-sig')
