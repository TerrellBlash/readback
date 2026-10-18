"""
The vendor_lookup tool for enrich_extraction.

The model decides whether to call it; invoices with a named vendor usually
get a lookup, resumes and contracts don't. The vendor table below is a
fixed stand-in so runs are reproducible; a deployment would replace it with
its vendor master data.
"""

from langchain_core.tools import tool

# Keys are lowercased vendor names.
_VENDORS = {
    "acme corp": {
        "address": "100 Industrial Way, Springfield, IL 62701",
        "tax_id": "12-3456789",
        "payment_history": "on_time",
    },
    "globex llc": {
        "address": "42 Galaxy Blvd, Austin, TX 78701",
        "tax_id": "98-7654321",
        "payment_history": "late_2_invoices",
    },
    "initech": {
        "address": "7 Cubicle Lane, Dallas, TX 75201",
        "tax_id": "55-5555555",
        "payment_history": "on_time",
    },
}


@tool
def vendor_lookup(vendor_name: str) -> dict:
    """Look up a vendor's address, tax ID, and payment history by name.

    Use this when processing an invoice or similar document where a
    vendor or supplier is named and additional vendor details would
    enrich the extracted data. Do not use it for documents that have
    no vendor, such as resumes.

    Args:
        vendor_name: The vendor or company name to look up.

    Returns:
        A dict with address, tax_id, and payment_history. If the vendor
        is unknown, returns a record marked as not found.
    """
    key = vendor_name.strip().lower()
    record = _VENDORS.get(key)
    if record is None:
        return {
            "address": None,
            "tax_id": None,
            "payment_history": "unknown",
            "found": False,
        }
    return {**record, "found": True}
