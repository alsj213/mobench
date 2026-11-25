"""Framework adapters — the only place engine knowledge lives.

Add a new engine by adding ``mobench/adapters/<engine>.py`` exposing an
adapter class whose ``engine`` id matches the module name; nothing in the
core/platform/analysis needs to change (spec: benchmark-core 框架适配器).
"""
