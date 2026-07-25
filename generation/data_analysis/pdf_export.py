from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import cm
import io
import base64

def generate_pdf(report, summary, charts):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()

    # custom styles
    title_style = ParagraphStyle('title',
        fontSize=24, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#00ff88'),
        spaceAfter=6)

    heading_style = ParagraphStyle('heading',
        fontSize=12, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#00aaff'),
        spaceBefore=16, spaceAfter=6)

    body_style = ParagraphStyle('body',
        fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#cccccc'),
        spaceAfter=4)

    elements = []

    # title
    elements.append(Paragraph('SAINYX', title_style))
    elements.append(Paragraph('Data Analysis Report', ParagraphStyle('sub',
        fontSize=14, fontName='Helvetica',
        textColor=colors.HexColor('#444444'),
        spaceAfter=20)))

    # summary
    elements.append(Paragraph('SUMMARY', heading_style))
    elements.append(Paragraph(summary, body_style))
    elements.append(Spacer(1, 12))

    # overview table
    elements.append(Paragraph('OVERVIEW', heading_style))
    overview_data = [
        ['Rows', 'Columns', 'Missing Values', 'Numeric Columns'],
        [
            str(report['rows']),
            str(report['columns']),
            str(sum(report['missing'].values())),
            str(sum(1 for t in report['dtypes'].values() if 'int' in t or 'float' in t))
        ]
    ]
    overview_table = Table(overview_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0a0a1a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#00ff88')),
        ('TEXTCOLOR', (0,1), (-1,1), colors.HexColor('#ffffff')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#1a1a2e')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#050510'), colors.HexColor('#0a0a1a')]),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(overview_table)
    elements.append(Spacer(1, 12))

    # column analysis
    elements.append(Paragraph('COLUMN ANALYSIS', heading_style))
    col_data = [['Column', 'Type', 'Missing']]
    for col in report['col_names']:
        missing = report['missing'].get(col, 0)
        col_data.append([col, report['dtypes'][col], str(missing) + ' missing' if missing > 0 else '✓ complete'])

    col_table = Table(col_data, colWidths=[6*cm, 4*cm, 6*cm])
    col_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0a0a1a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#00ff88')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#cccccc')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#1a1a2e')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#050510'), colors.HexColor('#0a0a1a')]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(col_table)
    elements.append(Spacer(1, 12))

    # stats
    if report.get('stats'):
        elements.append(Paragraph('STATISTICS', heading_style))
        stats_data = [['Column', 'Mean', 'Min', 'Max', 'Std']]
        for col, stats in report['stats'].items():
            stats_data.append([
                col,
                f"{stats.get('mean', 0):.2f}",
                f"{stats.get('min', 0):.2f}",
                f"{stats.get('max', 0):.2f}",
                f"{stats.get('std', 0):.2f}",
            ])
        stats_table = Table(stats_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm, 3*cm])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0a0a1a')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#00ff88')),
            ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#cccccc')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#1a1a2e')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#050510'), colors.HexColor('#0a0a1a')]),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 12))

    # charts
    if charts:
        elements.append(Paragraph('VISUALIZATIONS', heading_style))
        for chart in charts:
            img_data = base64.b64decode(chart['data'])
            img_buffer = io.BytesIO(img_data)
            img = Image(img_buffer, width=16*cm, height=8*cm)
            elements.append(Paragraph(chart['title'], ParagraphStyle('chart_title',
                fontSize=9, fontName='Helvetica',
                textColor=colors.HexColor('#444444'),
                spaceAfter=4)))
            elements.append(img)
            elements.append(Spacer(1, 8))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()