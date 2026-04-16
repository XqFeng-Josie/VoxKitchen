"""Schema version migrations.

When ``SCHEMA_VERSION`` in ``voxkitchen.schema.io`` is bumped, add a
migration function here that upgrades a manifest written under the old
version to the new one. ``read_cuts`` / ``read_header`` will route
version-mismatched reads through this package.

Plan 1 ships with ``SCHEMA_VERSION = "0.1"`` and no migrations. This module
is a stub placeholder so Plan 2+ knows where to put them.
"""
