import uuid

def uuid_to_oid_decimal(uuid_str):
    # Remove hyphens and convert to integer
    uuid_int = int(uuid_str.replace('-', ''), 16)
    # Convert to 128-bit binary string
    binary_uuid = bin(uuid_int)[2:].zfill(128)

    # Split into 32-bit (4-byte) segments for better readability, though
    # the standard is often interpreted as 16 bytes directly
    # For OID, it's typically treated as segments of the 128-bit number

    # Common interpretation is to just convert the whole 128-bit number into
    # multiple decimal components, splitting every 32 bits (4 bytes)
    # or every 8 bits (1 byte) for the OID arc.
    # A simple approach for OID generation is to split the 128-bit value
    # into 8 16-bit values, or 4 32-bit values.
    # Let's go with the more common and robust approach of treating the entire
    # UUID as a sequence of bytes.

    # A common method is to convert the UUID to 16 bytes, then represent
    # each byte as a decimal component.
    byte_values = []
    for i in range(0, 16):
        byte_values.append(int(uuid_str.replace('-', '')[i*2:(i*2)+2], 16))

    return ".".join(map(str, byte_values))

# Example:
my_uuid = str(uuid.uuid4()) # Generate a new random UUID
# my_uuid = 'f81d4fae-7dec-11d0-a765-00a0c91e6bf6' # Example fixed UUID
uuid_oid_segment = uuid_to_oid_decimal(my_uuid)

full_oid = f"2.25.{uuid_oid_segment}"
print(f"Generated UUID: {my_uuid}")
print(f"Corresponding OID under 2.25: {full_oid}")