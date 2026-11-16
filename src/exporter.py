import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

def export_cells_to_excel(cells, output_path: str):
    wb = Workbook()
    ws = wb.active
    ws.title = "Extracted_Table"

    thin_border = Border(
        left=Side(style='thin', color='A0A0A0'),
        right=Side(style='thin', color='A0A0A0'),
        top=Side(style='thin', color='A0A0A0'),
        bottom=Side(style='thin', color='A0A0A0')
    )
    header_fill = PatternFill(start_color="EAECEE", end_color="EAECEE", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True)
    body_font = Font(name="Calibri", size=11, bold=False)

    for cell in cells:
        r = int(cell.get("row", 1))
        c = int(cell.get("col", 1))
        text = str(cell.get("text", "")).strip()

        excel_cell = ws.cell(row=r, column=c, value=text)
        excel_cell.border = thin_border

        if r == 1:
            excel_cell.fill = header_fill
            excel_cell.font = header_font
            excel_cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        else:
            excel_cell.font = body_font
            excel_cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = col[0].column_letter
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    wb.save(output_path)