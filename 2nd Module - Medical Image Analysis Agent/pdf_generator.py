from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(text):
    """
    Generates a PDF from the analysis text.
    """
    if text is None:
        text = "No Text"  # Ensure text is a valid string

    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

    # Get built-in styles
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    title_style = styles["Title"]
    bullet_style = styles["BodyText"]

    content = []

    # Add Title
    content.append(Paragraph("📋 Medical Image Analysis Report", title_style))
    content.append(Spacer(1, 12))  # Space after title

    # Split content into sections using markdown headers ("###")
    sections = text.split("### ") if text else []
    
    for section in sections:
        if section.strip():  # Ignore empty sections
            lines = section.split("\n")  # Split section into lines
            section_title = lines[0]  # First line is the title
            
            # Add section title
            content.append(Paragraph(f"🔹 {section_title}", title_style))
            content.append(Spacer(1, 8))  # Space after section title

            # Process remaining lines as bullet points
            bullet_points = []
            for line in lines[1:]:  # Skip title, process content
                if line.strip():
                    bullet_points.append(ListItem(Paragraph(line.strip(), bullet_style)))
            
            if bullet_points:
                content.append(ListFlowable(bullet_points, bulletType="bullet"))
                content.append(Spacer(1, 10))  # Space after bullet list

    # Build the PDF
    doc.build(content)
    pdf_buffer.seek(0)
    return pdf_buffer
