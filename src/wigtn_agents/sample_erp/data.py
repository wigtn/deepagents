"""Seed data of the sample ERP: two company codes, one month (October 2026) of operations.

Amounts are KRW, net of VAT. The data is made up and small enough to check by hand; every
exception an agent is expected to find is listed in ``tests/test_sample_erp.py``.

Company 1000 (manufacturing) carries the cases:

* purchasing: one clean three-way match, a price variance over tolerance, a partial receipt
  invoiced in full, a possible duplicate invoice, and two receipts not yet invoiced (GR/IR);
* inventory: two materials below their reorder point once open orders are counted;
* receivables: one customer over its credit limit, items in every aging bucket;
* close: depreciation and a freight accrual not yet posted for 2026-10.

Company 2000 (distribution) has a little of everything, so the scope guard has something to keep apart.
"""

from __future__ import annotations

from typing import Any

BUSINESS_DATE = "2026-10-31"
PRICE_TOLERANCE_PCT = 2.0  # invoice price above the PO price by more than this is a variance

COMPANIES: dict[str, dict[str, Any]] = {
    "1000": {"name": "Sample Manufacturing Co.", "currency": "KRW"},
    "2000": {"name": "Sample Distribution Co.", "currency": "KRW"},
}

PERIODS: dict[str, dict[str, str]] = {
    "1000": {"2026-08": "closed", "2026-09": "closed", "2026-10": "open", "2026-11": "future"},
    "2000": {"2026-09": "closed", "2026-10": "open", "2026-11": "future"},
}

VENDORS: dict[str, dict[str, dict[str, Any]]] = {
    "1000": {
        "V-100": {"name": "Hangyeol Materials", "terms_days": 30},
        "V-200": {"name": "Daemyung Bearing", "terms_days": 45},
        "V-300": {"name": "Woojin Packaging", "terms_days": 30},
        "V-400": {"name": "Seohae Oil", "terms_days": 30},
        "V-500": {"name": "Hanbit Logistics", "terms_days": 30},
    },
    "2000": {"V-900": {"name": "Coastal Supply", "terms_days": 30}},
}

CUSTOMERS: dict[str, dict[str, dict[str, Any]]] = {
    "1000": {
        "C-500": {"name": "Donghae Machinery", "credit_limit": 50_000_000},
        "C-510": {"name": "Namsan Electronics", "credit_limit": 30_000_000},
        "C-520": {"name": "Seoul Precision", "credit_limit": 10_000_000},
    },
    "2000": {"C-900": {"name": "Harbor Retail", "credit_limit": 20_000_000}},
}

MATERIALS: dict[str, dict[str, dict[str, Any]]] = {
    "1000": {
        "M-1001": {
            "description": "Steel plate 2t",
            "uom": "EA",
            "on_hand": 120,
            "reserved": 140,
            "reorder_point": 150,
            "safety_stock": 60,
            "min_order_qty": 50,
            "lead_time_days": 14,
            "unit_cost": 85_000,
            "vendor": "V-100",
        },
        "M-1002": {
            "description": "Bearing 6204",
            "uom": "EA",
            "on_hand": 1_900,
            "reserved": 300,
            "reorder_point": 500,
            "safety_stock": 200,
            "min_order_qty": 500,
            "lead_time_days": 7,
            "unit_cost": 3_200,
            "vendor": "V-200",
        },
        "M-1003": {
            "description": "Packing box, large",
            "uom": "EA",
            "on_hand": 300,
            "reserved": 280,
            "reorder_point": 600,
            "safety_stock": 150,
            "min_order_qty": 500,
            "lead_time_days": 3,
            "unit_cost": 1_100,
            "vendor": "V-300",
        },
        "M-1004": {
            "description": "Hydraulic oil 20L",
            "uom": "CAN",
            "on_hand": 35,
            "reserved": 0,
            "reorder_point": 30,
            "safety_stock": 10,
            "min_order_qty": 10,
            "lead_time_days": 10,
            "unit_cost": 64_000,
            "vendor": "V-400",
        },
        "M-1005": {
            "description": "Bolt M10",
            "uom": "EA",
            "on_hand": 5_000,
            "reserved": 1_200,
            "reorder_point": 2_000,
            "safety_stock": 800,
            "min_order_qty": 1_000,
            "lead_time_days": 5,
            "unit_cost": 45,
            "vendor": "V-100",
        },
    },
    "2000": {
        "M-9001": {
            "description": "Pallet wrap",
            "uom": "ROLL",
            "on_hand": 40,
            "reserved": 35,
            "reorder_point": 30,
            "safety_stock": 10,
            "min_order_qty": 20,
            "lead_time_days": 4,
            "unit_cost": 12_000,
            "vendor": "V-900",
        },
    },
}

# Purchase orders: one line each keeps the matching readable.
PURCHASE_ORDERS: dict[str, dict[str, dict[str, Any]]] = {
    "1000": {
        "PO-4500000101": {"vendor": "V-100", "material": "M-1001", "qty": 100, "price": 85_000, "date": "2026-10-02"},
        "PO-4500000102": {"vendor": "V-200", "material": "M-1002", "qty": 1_000, "price": 3_200, "date": "2026-10-05"},
        "PO-4500000103": {"vendor": "V-300", "material": "M-1003", "qty": 2_000, "price": 1_100, "date": "2026-10-08"},
        "PO-4500000104": {"vendor": "V-400", "material": "M-1004", "qty": 20, "price": 64_000, "date": "2026-10-15"},
        "PO-4500000105": {"vendor": "V-100", "material": "M-1001", "qty": 80, "price": 86_000, "date": "2026-10-25"},
        "PO-4500000106": {"vendor": "V-100", "material": "M-1001", "qty": 50, "price": 85_000, "date": "2026-10-20"},
    },
    "2000": {
        "PO-4600000001": {"vendor": "V-900", "material": "M-9001", "qty": 30, "price": 12_000, "date": "2026-10-10"},
    },
}

GOODS_RECEIPTS: dict[str, list[dict[str, Any]]] = {
    "1000": [
        {"gr": "GR-5000000201", "po": "PO-4500000101", "qty": 100, "date": "2026-10-12"},
        {"gr": "GR-5000000202", "po": "PO-4500000102", "qty": 1_000, "date": "2026-10-14"},
        {"gr": "GR-5000000203", "po": "PO-4500000103", "qty": 1_500, "date": "2026-10-16"},
        {"gr": "GR-5000000204", "po": "PO-4500000104", "qty": 20, "date": "2026-10-29"},
        {"gr": "GR-5000000205", "po": "PO-4500000106", "qty": 50, "date": "2026-10-30"},
    ],
    "2000": [{"gr": "GR-5100000001", "po": "PO-4600000001", "qty": 30, "date": "2026-10-15"}],
}

VENDOR_INVOICES: dict[str, dict[str, dict[str, Any]]] = {
    "1000": {
        "INV-V100-7781": {"vendor": "V-100", "po": "PO-4500000101", "qty": 100, "price": 85_000, "date": "2026-10-13"},
        "INV-V200-0915": {"vendor": "V-200", "po": "PO-4500000102", "qty": 1_000, "price": 3_350, "date": "2026-10-16"},
        "INV-V300-1102": {"vendor": "V-300", "po": "PO-4500000103", "qty": 2_000, "price": 1_100, "date": "2026-10-18"},
        "INV-V200-0917": {"vendor": "V-200", "po": "PO-4500000102", "qty": 1_000, "price": 3_350, "date": "2026-10-27"},
    },
    "2000": {
        "INV-V900-0042": {"vendor": "V-900", "po": "PO-4600000001", "qty": 30, "price": 12_000, "date": "2026-10-16"},
    },
}

CUSTOMER_INVOICES: dict[str, dict[str, dict[str, Any]]] = {
    "1000": {
        "AR-9001": {"customer": "C-500", "amount": 12_000_000, "paid": 0, "due": "2026-10-15"},
        "AR-9002": {"customer": "C-500", "amount": 8_500_000, "paid": 0, "due": "2026-11-20"},
        "AR-9010": {"customer": "C-510", "amount": 18_000_000, "paid": 0, "due": "2026-08-20"},
        "AR-9011": {"customer": "C-510", "amount": 9_000_000, "paid": 0, "due": "2026-09-25"},
        "AR-9012": {"customer": "C-510", "amount": 6_000_000, "paid": 0, "due": "2026-11-10"},
        "AR-9020": {"customer": "C-520", "amount": 4_000_000, "paid": 1_000_000, "due": "2026-06-30"},
    },
    "2000": {"AR-9901": {"customer": "C-900", "amount": 7_000_000, "paid": 0, "due": "2026-10-20"}},
}

ACCOUNTS: dict[str, dict[str, str]] = {
    "1100": {"name": "Cash and deposits", "type": "asset"},
    "1200": {"name": "Accounts receivable", "type": "asset"},
    "1300": {"name": "Inventory", "type": "asset"},
    "1500": {"name": "Machinery and equipment", "type": "asset"},
    "1590": {"name": "Accumulated depreciation", "type": "asset"},
    "2100": {"name": "Accounts payable", "type": "liability"},
    "2150": {"name": "GR/IR clearing", "type": "liability"},
    "2200": {"name": "Accrued expenses", "type": "liability"},
    "3100": {"name": "Share capital", "type": "equity"},
    "3200": {"name": "Retained earnings", "type": "equity"},
    "4100": {"name": "Sales", "type": "revenue"},
    "5100": {"name": "Cost of goods sold", "type": "expense"},
    "5200": {"name": "Freight", "type": "expense"},
    "6100": {"name": "Salaries", "type": "expense"},
    "6200": {"name": "Rent", "type": "expense"},
    "6300": {"name": "Depreciation", "type": "expense"},
}

# Balances at the start of 2026-10 (debit positive, credit negative); each set sums to zero.
OPENING_BALANCES: dict[str, dict[str, int]] = {
    "1000": {
        "1100": 272_000_000,
        "1200": 18_000_000,
        "1300": 95_000_000,
        "1500": 288_000_000,
        "1590": -96_000_000,
        "2100": -38_000_000,
        "2150": 0,
        "2200": -4_000_000,
        "3100": -300_000_000,
        "3200": -235_000_000,
    },
    "2000": {"1100": 65_000_000, "1200": 0, "1300": 15_000_000, "2100": -10_000_000, "3100": -70_000_000},
}

# October postings other than the purchasing flow (which the ledger derives from receipts and invoices).
JOURNAL_ENTRIES: dict[str, list[dict[str, Any]]] = {
    "1000": [
        {
            "doc": "JE-1000-2026-10-001",
            "date": "2026-10-01",
            "text": "October rent",
            "lines": [{"account": "6200", "debit": 3_000_000}, {"account": "1100", "credit": 3_000_000}],
        },
        {
            "doc": "JE-1000-2026-10-002",
            "date": "2026-10-25",
            "text": "October salaries",
            "lines": [{"account": "6100", "debit": 42_000_000}, {"account": "1100", "credit": 42_000_000}],
        },
        {
            "doc": "JE-1000-2026-10-003",
            "date": "2026-10-28",
            "text": "October sales (summary)",
            "lines": [
                {"account": "1200", "debit": 38_500_000},
                {"account": "4100", "credit": 38_500_000},
                {"account": "5100", "debit": 24_000_000},
                {"account": "1300", "credit": 24_000_000},
            ],
        },
    ],
    "2000": [
        {
            "doc": "JE-2000-2026-10-001",
            "date": "2026-10-20",
            "text": "October sales (summary)",
            "lines": [{"account": "1200", "debit": 7_000_000}, {"account": "4100", "credit": 7_000_000}],
        },
    ],
}

# Month-end items the close checklist looks for. ``match`` says which posting settles the item.
RECURRING_CLOSE_ITEMS: dict[str, list[dict[str, Any]]] = {
    "1000": [
        {
            "id": "rent",
            "text": "Monthly rent",
            "amount": 3_000_000,
            "match": {"debit": "6200", "credit": "1100"},
            "proposal": None,
        },
        {
            "id": "depreciation",
            "text": "Monthly depreciation, machinery (straight line, 288,000,000 over 120 months)",
            "amount": 2_400_000,
            "match": {"debit": "6300", "credit": "1590"},
            "proposal": {"debit": "6300", "credit": "1590"},
        },
        {
            "id": "freight-accrual",
            "text": "Freight for October under the V-500 contract (invoice arrives next month)",
            "amount": 1_800_000,
            "match": {"debit": "5200", "credit": "2200"},
            "proposal": {"debit": "5200", "credit": "2200"},
        },
    ],
    "2000": [],
}
