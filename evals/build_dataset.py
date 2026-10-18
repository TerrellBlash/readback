"""
Build the labeled evaluation dataset.

Writes evals/data/<id>.txt or .pdf plus evals/data/labels.json. Everything
is synthetic and seeded, so the same command always writes the same files:
fictional companies and people, example.com addresses and 555 phone
numbers.

Main set, 44 documents:
- 12 invoices, 9 resumes, 9 contracts, and 3 documents of no supported
  type (a recipe, meeting notes, a building notice)
- each type split evenly across plain text, text-layer PDF and scanned PDF
- the 11 scanned documents a second time as poor scans (low resolution,
  heavy blur), with the same labels, so any drop in accuracy between the
  two is down to scan quality alone

Plus a held-out set of 12 new documents (4 of each type), poor scans only,
labeled split "holdout". They were added after the second-read check was
chosen on the poor scans above, so they measure that check on documents
it was never tuned on. Everything else is split "main".
- 3 invoices, 2 resumes and 2 contracts leave out one required field, so
  the pipeline should send them to human review instead of storing them
- contracts are the types that carry payment terms; NDAs are left out
  because the pipeline requires payment terms of every contract (see Known
  Limitations in the README)

Run: python -m evals.build_dataset
"""

import json
import random
import textwrap
import zlib
from datetime import date, timedelta
from pathlib import Path

from evals.pdfgen import OFFICE_SCAN, POOR_SCAN, scanned_pdf, text_pdf

DATA_DIR = Path(__file__).parent / "data"
SEED = 2026
FORMATS = ("text", "pdf_text", "pdf_scan")
POOR_SCAN_FORMAT = "pdf_scan_poor"

VENDORS = [
    ("Brightline Logistics LLC", "4100 Commerce Pkwy", "Memphis, TN 38118"),
    ("Northwind Analytics Inc.", "88 Market Street, Suite 400", "Denver, CO 80202"),
    ("Cobalt Peak Software", "2200 Ridge Road", "Boise, ID 83702"),
    ("Harbor & Finch Supply Co.", "15 Wharf Lane", "Portland, ME 04101"),
    ("Meridian Facility Services", "702 Industrial Blvd", "Columbus, OH 43215"),
    ("Quartzline Printing", "39 Press Avenue", "Raleigh, NC 27601"),
    ("Redwood Data Systems", "1450 Pine Street", "Sacramento, CA 95814"),
    ("Silverleaf Catering LLC", "620 Orchard Way", "Savannah, GA 31401"),
    ("Tidewater Electrical Inc.", "77 Harbor Drive", "Norfolk, VA 23510"),
    ("Vantage Point Consulting", "500 Lake Shore Dr", "Chicago, IL 60611"),
    ("Juniper Labs GmbH", "Lindenstrasse 12", "10969 Berlin, Germany"),
    ("Kestrel Office Partners Ltd", "8 Bishopsgate", "London EC2N 4BQ, UK"),
]
CUSTOMERS = [
    "Acme Corporation", "Bluebird Health Partners", "Granite State Credit Union",
    "Orchard Hill Schools", "Pinecrest Hotel Group", "Summit Ridge Manufacturing",
]
LINE_ITEMS = [
    ("Freight - LTL shipment", (1, 4), (180, 950)),
    ("Warehouse storage (pallet-month)", (5, 40), (12, 25)),
    ("Software subscription - annual seat", (1, 25), (120, 480)),
    ("Implementation services (hours)", (4, 40), (95, 185)),
    ("Printer toner cartridge", (2, 12), (45, 120)),
    ("Catering - working lunch for 20", (1, 3), (320, 640)),
    ("Electrical inspection", (1, 2), (250, 600)),
    ("Cloud storage (TB-month)", (1, 20), (18, 35)),
    ("On-site support visit", (1, 5), (150, 400)),
    ("Ergonomic office chair", (2, 10), (180, 420)),
]
PEOPLE = [
    ("Maya Okafor", "Atlanta", "GA"), ("Daniel Reyes", "Austin", "TX"),
    ("Priya Natarajan", "Seattle", "WA"), ("Samuel Brooks", "Raleigh", "NC"),
    ("Elena Petrova", "Denver", "CO"), ("Jordan Kim", "Chicago", "IL"),
    ("Aisha Rahman", "Charlotte", "NC"), ("Marcus Bell", "Phoenix", "AZ"),
    ("Hannah Lindqvist", "Minneapolis", "MN"),
]
TITLES = ["Data Analyst", "Software Engineer", "Machine Learning Engineer", "Operations Manager", "Product Analyst"]
EMPLOYERS = [
    "Lumen Freight", "Bayside Mutual", "Crescent Health Systems", "Ironwood Retail Group",
    "Polaris Payments", "Westfield Labs", "Cardinal Insurance Co.", "Riverstone Bank",
]
SCHOOLS = [
    ("Georgia State University", "B.S.", "Computer Science"),
    ("University of Texas at Austin", "B.A.", "Economics"),
    ("North Carolina State University", "M.S.", "Statistics"),
    ("University of Colorado Boulder", "B.S.", "Applied Mathematics"),
    ("Arizona State University", "B.S.", "Information Systems"),
]
SKILLS = [
    "Python", "SQL", "Pandas", "scikit-learn", "PyTorch", "Docker", "Kubernetes", "AWS",
    "Tableau", "FastAPI", "Airflow", "Spark", "Git", "Terraform", "LangChain", "PostgreSQL",
]
CONTRACT_TYPES = {
    "Master Services Agreement": ("Provider", "Client"),
    "Consulting Agreement": ("Consultant", "Company"),
    "Software License Agreement": ("Licensor", "Licensee"),
    "Commercial Lease Agreement": ("Landlord", "Tenant"),
    "Supply Agreement": ("Supplier", "Buyer"),
}
STATES = ["Delaware", "Georgia", "Texas", "New York", "California", "Illinois", "Colorado", "Ohio"]
SIGNERS = ["Laura Chen", "Robert Diaz", "Fatima Noor", "Kevin O'Neill", "Grace Park", "Thomas Webb"]


def _date_text(rng: random.Random, value: date) -> str:
    styles = [
        f"{value:%B} {value.day}, {value.year}",
        value.isoformat(),
        f"{value.day:02d} {value:%b} {value.year}",
        f"{value:%b} {value.day}, {value.year}",
    ]
    return rng.choice(styles)


def _money(amount: float, currency: str | None) -> str:
    if currency == "USD":
        return f"${amount:,.2f}"
    if currency is None:
        return f"{amount:,.2f}"
    return f"{currency} {amount:,.2f}"


def _wrap(text: str) -> str:
    return "\n".join(textwrap.fill(p, 90) if p else "" for p in text.split("\n"))


def make_invoice(rng: random.Random, index: int, absent: str | None) -> tuple[str, dict]:
    vendor, street, city = VENDORS[index % len(VENDORS)]
    currency = {"Juniper Labs GmbH": "EUR", "Kestrel Office Partners Ltd": "GBP"}.get(vendor, "USD")
    if absent == "currency":
        currency = None
    issued = date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))
    terms = rng.choice([15, 30, 45, 60])
    due = issued + timedelta(days=terms)
    number = rng.choice([f"INV-{issued.year}-{rng.randint(100, 9999):04d}", f"{rng.randint(10000, 99999)}",
                         f"{vendor[:2].upper()}-{rng.randint(10000, 99999)}"])
    items = []
    for name, (qmin, qmax), (pmin, pmax) in rng.sample(LINE_ITEMS, rng.randint(1, 5)):
        quantity = rng.randint(qmin, qmax)
        unit = round(rng.uniform(pmin, pmax), 2)
        items.append({"description": name, "quantity": quantity, "unit_price": unit,
                      "total": round(quantity * unit, 2)})
    subtotal = round(sum(item["total"] for item in items), 2)
    rate = 0.2 if currency in ("EUR", "GBP") else rng.choice([0.05, 0.0625, 0.0725, 0.0825])
    tax = round(subtotal * rate, 2)
    total = round(subtotal + tax, 2)
    customer = rng.choice(CUSTOMERS)

    if index % 2 == 0:
        lines = [vendor, street, city, "", "INVOICE", ""]
        if absent != "invoice_number":
            lines.append(f"Invoice Number: {number}")
        lines.append(f"Invoice Date: {_date_text(rng, issued)}")
        if absent != "due_date":
            lines += [f"Due Date: {_date_text(rng, due)}", f"Payment Terms: Net {terms}"]
        lines += ["", "Bill To:", customer, "", f"{'Description':<38}{'Qty':>5}{'Unit Price':>14}{'Amount':>14}"]
        lines += [f"{i['description']:<38}{i['quantity']:>5}{_money(i['unit_price'], currency):>14}"
                  f"{_money(i['total'], currency):>14}" for i in items]
        lines += ["", f"Subtotal: {_money(subtotal, currency)}",
                  f"Tax ({rate * 100:g}%): {_money(tax, currency)}",
                  f"Total Due: {_money(total, currency)}", "",
                  f"Thank you for your business. Questions? billing@{vendor.split()[0].lower()}.example.com"]
    else:
        lines = ["INVOICE", "-" * 40, f"From: {vendor}, {street}, {city}", f"To: {customer}"]
        if absent != "invoice_number":
            lines.append(f"Invoice #: {number}")
        lines.append(f"Issued: {_date_text(rng, issued)}")
        if absent != "due_date":
            lines.append(f"Payment due: {_date_text(rng, due)}")
        lines += ["", "Items:"]
        lines += [f"{n}. {i['description']} - {i['quantity']} x {_money(i['unit_price'], currency)}"
                  f" = {_money(i['total'], currency)}" for n, i in enumerate(items, 1)]
        lines += ["", f"Subtotal ........ {_money(subtotal, currency)}",
                  f"Tax ............. {_money(tax, currency)}",
                  f"TOTAL ........... {_money(total, currency)}"]

    fields = {
        "vendor_name": vendor, "invoice_number": number, "invoice_date": issued.isoformat(),
        "due_date": due.isoformat(), "line_items": items, "subtotal": subtotal, "tax": tax,
        "total_amount": total, "currency": currency,
    }
    if absent:
        fields[absent] = None
    return "\n".join(lines), fields


def make_resume(rng: random.Random, index: int, absent: str | None) -> tuple[str, dict]:
    name, city, state = PEOPLE[index % len(PEOPLE)]
    first, last = name.lower().split()
    email = f"{first}.{last}@example.com"
    phone = f"({rng.choice([404, 512, 206, 919, 303, 312])}) 555-01{rng.randint(10, 99)}"
    title = rng.choice(TITLES)
    skills = rng.sample(SKILLS, rng.randint(6, 9))
    years = rng.randint(4, 12)
    summary = (f"{title} with {years} years of experience turning messy operational data into "
               f"reliable reporting and automation. Comfortable owning projects end to end.")
    jobs, end_year = [], 2026
    for employer in rng.sample(EMPLOYERS, rng.randint(2, 3)):
        start_year = end_year - rng.randint(2, 4)
        jobs.append((employer, rng.choice(TITLES), start_year, "Present" if end_year == 2026 else str(end_year)))
        end_year = start_year
    schools = rng.sample(SCHOOLS, rng.randint(1, 2))

    contact = [f"{city}, {state}", email]
    if absent != "phone":
        contact.append(phone)
    lines = [name.upper(), title, " | ".join(contact), ""]
    if absent != "summary":
        lines += ["SUMMARY", summary, ""]
    lines += ["SKILLS", ", ".join(skills), "", "EXPERIENCE"]
    for employer, job_title, start, end in jobs:
        lines += [f"{job_title} - {employer} ({start} - {end})",
                  "- Built dashboards and pipelines used by finance and operations teams",
                  "- Cut manual reporting time by automating weekly data pulls", ""]
    lines.append("EDUCATION")
    lines += [f"{degree} in {field_} - {school} ({rng.randint(2008, 2018)})" for school, degree, field_ in schools]

    fields = {
        "full_name": name, "email": email, "phone": phone, "location": city, "summary": summary,
        "skills": skills, "experience": [job[0] for job in jobs], "education": [s[0] for s in schools],
    }
    if absent:
        fields[absent] = None
    return _wrap("\n".join(lines)), fields


def make_contract(rng: random.Random, index: int, absent: str | None) -> tuple[str, dict]:
    contract_type = list(CONTRACT_TYPES)[index % len(CONTRACT_TYPES)]
    role_a, role_b = CONTRACT_TYPES[contract_type]
    party_a, party_b = (v[0] for v in rng.sample(VENDORS[:10], 2))
    state_a, state_b, law = rng.sample(STATES, 3)
    effective = date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))
    expiration = effective.replace(year=effective.year + rng.choice([1, 2, 3]))
    notice, cure = rng.choice([30, 60, 90]), rng.choice([15, 30])
    payment = {
        "Master Services Agreement": f"{role_b} shall pay {role_a} the fees set out in each Statement of Work. "
                                     f"{role_a} will invoice monthly, and {role_b} shall pay each invoice within "
                                     f"thirty (30) days of receipt.",
        "Consulting Agreement": f"{role_b} shall pay {role_a} ${rng.randint(95, 220)} per hour, invoiced monthly "
                                f"and payable within fifteen (15) days of the invoice date.",
        "Software License Agreement": f"{role_b} shall pay an annual license fee of ${rng.randint(12, 90) * 1000:,}, "
                                      f"due within thirty (30) days of the Effective Date and of each anniversary.",
        "Commercial Lease Agreement": f"{role_b} shall pay monthly rent of ${rng.randint(28, 95) * 100:,}, due on "
                                      f"the first day of each month, and a security deposit equal to two months' "
                                      f"rent on signing.",
        "Supply Agreement": f"{role_b} shall pay for each delivery at the unit prices in Schedule A within "
                            f"forty-five (45) days of the invoice date.",
    }[contract_type]
    obligations = {
        "Master Services Agreement": f"{role_a} shall perform the services described in each Statement of Work "
                                     f"signed by both parties, with the skill and care of a qualified professional.",
        "Consulting Agreement": f"{role_a} shall provide data and reporting advisory services as requested by "
                                f"{role_b} and shall not subcontract any work without written consent.",
        "Software License Agreement": f"{role_a} grants {role_b} a non-exclusive, non-transferable license to use "
                                      f"the software for internal business purposes on up to 250 named users.",
        "Commercial Lease Agreement": f"{role_a} leases to {role_b} approximately 4,800 square feet of office space "
                                      f"for general office use, with access to shared parking.",
        "Supply Agreement": f"{role_a} shall deliver the products listed in Schedule A within ten (10) business "
                            f"days of each purchase order, packaged to prevent damage in transit.",
    }[contract_type]
    term = (f"This Agreement begins on the Effective Date and ends on {_date_text(rng, expiration)}, unless "
            f"terminated earlier under Section 6." if absent != "expiration_date" else
            "This Agreement begins on the Effective Date and continues until terminated under Section 6.")
    sections = [
        ("SCOPE", obligations),
        ("TERM", term),
        ("PAYMENT", payment),
        ("CONFIDENTIALITY", "Each party shall keep the other party's confidential information secret, use it only "
                            "to perform this Agreement, and return or destroy it when this Agreement ends."),
        ("WARRANTIES", "Each party warrants that it has the authority to enter into this Agreement and that "
                       "performing it will not breach any other agreement it is bound by."),
        ("TERMINATION", f"Either party may terminate this Agreement for convenience on {notice} days' written "
                        f"notice, or immediately if the other party materially breaches this Agreement and does "
                        f"not cure the breach within {cure} days of written notice."),
        ("LIMITATION OF LIABILITY", "Neither party is liable for indirect or consequential damages. Each party's "
                                    "total liability is capped at the fees paid in the twelve months "
                                    "before the claim."),
    ]
    if absent != "governing_law":
        sections.append(("GOVERNING LAW", f"This Agreement is governed by the laws of the State of {law}, without "
                                          f"regard to its conflict of laws rules."))
    sections.append(("ENTIRE AGREEMENT", "This Agreement is the parties' entire agreement on its subject and "
                                         "replaces all earlier proposals and understandings."))
    signer_a, signer_b = rng.sample(SIGNERS, 2)
    lines = [contract_type.upper(), "",
             f'This {contract_type} (the "Agreement") is made as of {_date_text(rng, effective)} (the "Effective '
             f'Date") by and between {party_a}, a {state_a} company ("{role_a}"), and {party_b}, a {state_b} '
             f'company ("{role_b}").', ""]
    for number, (heading, body) in enumerate(sections, 1):
        lines += [f"{number}. {heading}. {body}", ""]
    lines += ["IN WITNESS WHEREOF, the parties have signed this Agreement as of the Effective Date.", "",
              f"{party_a}", f"By: {signer_a}", "", f"{party_b}", f"By: {signer_b}"]

    fields = {
        "contract_type": contract_type, "parties": [party_a, party_b], "effective_date": effective.isoformat(),
        "expiration_date": expiration.isoformat(), "key_terms": obligations, "payment_terms": payment,
        "termination_clause": sections[5][1], "governing_law": law,
    }
    if absent:
        fields[absent] = None
    return _wrap("\n".join(lines)), fields


OTHER_DOCUMENTS = [
    "LEMON BARS\nServes 16\n\nIngredients\n- 1 cup butter, softened\n- 1/2 cup powdered sugar\n- 2 cups flour\n"
    "- 4 eggs\n- 1 1/2 cups sugar\n- 1/3 cup lemon juice\n\nSteps\n1. Heat the oven to 350 F.\n2. Mix butter, "
    "powdered sugar and flour; press into a 9x13 pan and bake 20 minutes.\n3. Whisk eggs, sugar and lemon juice; "
    "pour over the crust.\n4. Bake 25 minutes more, cool, and dust with powdered sugar.",
    "Q3 PLANNING OFFSITE - NOTES\nAttendees: Laura, Kevin, Fatima, Grace\n\nDecisions\n- Move the weekly sync to "
    "Tuesdays at 10:00.\n- Pilot the new onboarding checklist with the October hires.\n\nAction items\n- Kevin: draft "
    "the onboarding checklist by Friday.\n- Grace: book the room for the November review.\n- Fatima: collect "
    "feedback from the support team.",
    "BUILDING NOTICE\nTo all tenants of 500 Lake Shore Drive\n\nThe north elevator will be out of service for "
    "scheduled maintenance from Monday through Wednesday next week. The south elevator and both stairwells remain "
    "open. Deliveries should use the loading dock entrance on Grand Avenue. Thank you for your patience.\n\n"
    "Property Management Office",
]

# (document type, how many, {index: field to leave out})
PLAN = [
    ("invoice", 12, {0: "due_date", 4: "invoice_number", 8: "currency"}),
    ("resume", 9, {1: "phone", 5: "summary"}),
    ("contract", 9, {0: "expiration_date", 7: "governing_law"}),
]
# (document type, first index, how many): continues each type's numbering,
# so every held-out document has its own id and its own random draws.
HOLDOUT_PLAN = [("invoice", 12, 4), ("resume", 9, 4), ("contract", 9, 4)]
MAKERS = {"invoice": make_invoice, "resume": make_resume, "contract": make_contract}
PREFIX = {"invoice": "inv", "resume": "res", "contract": "con", "unknown": "unk"}


def _write(doc_id: str, text: str, fmt: str) -> str:
    if fmt == "text":
        path = DATA_DIR / f"{doc_id}.txt"
        path.write_text(text + "\n")
        return path.name
    path = DATA_DIR / f"{doc_id}.pdf"
    if fmt == "pdf_text":
        path.write_bytes(text_pdf(text))
    else:
        quality = POOR_SCAN if fmt == POOR_SCAN_FORMAT else OFFICE_SCAN
        path.write_bytes(scanned_pdf(text, seed=zlib.crc32(doc_id.encode()), quality=quality))
    return path.name


def build() -> list[dict]:
    """
    Write the dataset files and labels.

    - Returns: the label records written to labels.json.
    """
    DATA_DIR.mkdir(exist_ok=True)
    for old in DATA_DIR.iterdir():
        old.unlink()
    labels: list[dict] = []
    scans: list[tuple[dict, str]] = []

    def add(doc_id: str, text: str, fmt: str, doc_type: str, absent: str | None, fields: dict,
            split: str = "main") -> None:
        label = {
            "id": doc_id, "file": _write(doc_id, text, fmt), "format": fmt, "doc_type": doc_type,
            "absent_field": absent, "expected_route": "review" if absent or doc_type == "unknown" else "auto",
            "split": split, "fields": fields,
        }
        labels.append(label)
        if fmt == "pdf_scan":
            scans.append((label, text))

    for doc_type, count, absent_by_index in PLAN:
        for index in range(count):
            doc_id = f"{PREFIX[doc_type]}-{index + 1:02d}"
            rng = random.Random(f"{SEED}-{doc_id}")
            absent = absent_by_index.get(index)
            text, fields = MAKERS[doc_type](rng, index, absent)
            add(doc_id, text, FORMATS[index % 3], doc_type, absent, fields)
    for index, text in enumerate(OTHER_DOCUMENTS):
        add(f"unk-{index + 1:02d}", text, FORMATS[index % 3], "unknown", None, {})
    for label, text in scans:
        poor_id = f"{label['id']}-poor"
        labels.append({**label, "id": poor_id, "file": _write(poor_id, text, POOR_SCAN_FORMAT),
                       "format": POOR_SCAN_FORMAT})
    for doc_type, first, count in HOLDOUT_PLAN:
        for index in range(first, first + count):
            doc_id = f"{PREFIX[doc_type]}-{index + 1:02d}"
            rng = random.Random(f"{SEED}-{doc_id}")
            text, fields = MAKERS[doc_type](rng, index, None)
            add(doc_id, text, POOR_SCAN_FORMAT, doc_type, None, fields, split="holdout")
    (DATA_DIR / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    return labels


if __name__ == "__main__":
    written = build()
    print(f"Wrote {len(written)} documents and labels.json to {DATA_DIR}")
