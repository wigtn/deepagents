"""A small in-memory ERP the shipped catalog's agents work against.

It covers purchasing, inventory, receivables and the general ledger. See ``data`` for the seed
and ``system`` for the rules.
"""

from wigtn_agents.sample_erp.routes import erp_routes
from wigtn_agents.sample_erp.system import ErpError, SampleErp, shared_erp

__all__ = ["ErpError", "SampleErp", "erp_routes", "shared_erp"]
