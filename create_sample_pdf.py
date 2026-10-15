"""
create_sample_pdf.py
--------------------
Generates a realistic sample "corporate handbook" PDF for testing the RAG pipeline.
Supports fpdf2, reportlab, or pymupdf (fitz).
"""

import sys
import os

SAMPLE_TEXT = """ACME Corporation - Employee Handbook (Sample)

Chapter 1: Leave Policy
Employees are entitled to 20 days of paid annual leave per calendar year.
Leave must be applied for at least 5 working days in advance.
Sick leave is limited to 10 days per year and requires a medical certificate
for absences exceeding 3 consecutive days.

Chapter 2: Remote Work Policy
Employees may work remotely up to 3 days per week with prior approval from
their line manager. Remote employees must be reachable during core hours of
10 AM to 4 PM in their local timezone. A stable internet connection of at
least 10 Mbps is required.

Chapter 3: Code of Conduct
All employees must maintain respectful communication in the workplace.
Harassment, discrimination, or bullying of any kind will result in immediate
disciplinary action. Employees are expected to protect confidential company
information and must sign an NDA on their first day.

Chapter 4: IT & Security
Passwords must be at least 12 characters and changed every 90 days.
All company laptops must have full-disk encryption enabled. Employees must
not install unauthorised software. Suspected security incidents should be
reported to security@acme.com within 1 hour of discovery.

Chapter 5: Expense Reimbursement
Business expenses up to $500 can be self-approved with receipts submitted
within 30 days. Expenses above $500 require manager pre-approval. Travel
bookings must be made through the official travel portal. No alcohol expenses
are reimbursable except at company-sponsored events.

Chapter 6: Performance Reviews
Performance reviews are conducted twice a year - in June and December.
Employees who receive a rating below expectations for two consecutive cycles
may be placed on a Performance Improvement Plan (PIP). All goals must be
set using the SMART framework and uploaded to the HR portal by end of January.
"""

def generate_handbook(output_path: str = "handbook.pdf") -> str:
    """Generates the sample handbook PDF using available PDF libraries."""
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(0, 10, "ACME Corporation - Employee Handbook", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Helvetica", size=11)
        
        # Write body lines
        for paragraph in SAMPLE_TEXT.strip().split("\n\n"):
            lines = paragraph.split("\n")
            title = lines[0]
            body = " ".join(lines[1:]) if len(lines) > 1 else ""
            
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(0, 8, title, ln=True)
            if body:
                pdf.set_font("Helvetica", size=10)
                pdf.multi_cell(0, 6, body)
            pdf.ln(3)

        pdf.output(output_path)
        print(f"Sample PDF successfully created with fpdf2: {output_path}")
        return output_path
    except ImportError:
        pass

    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), SAMPLE_TEXT, fontsize=11, fontname="helv")
        doc.save(output_path)
        doc.close()
        print(f"Sample PDF successfully created with pymupdf: {output_path}")
        return output_path
    except ImportError:
        pass

    raise RuntimeError("No PDF generation library found. Please install fpdf2: pip install fpdf2")

if __name__ == "__main__":
    generate_handbook("handbook.pdf")
