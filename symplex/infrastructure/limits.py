"""Host-owned request limits, separate from persistent monetary entitlements."""

# A complete semantic representation may require more than the ordinary memory
# envelope. Bytes remain a conservative token reservation, not a free expansion.
MAX_REQUEST_BYTES = 96000
DEFAULT_ROLE_REQUEST_BYTES = 58000
