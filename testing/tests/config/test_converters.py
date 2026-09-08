import pytest
from loguru import logger
from quartical.config.converters import as_antenna_index


# Antenna names which cannot be confused with antenna indices.
DISTINCT_NAMES = ["m000", "m001", "m002"]

# Antenna names which are themselves integers, offset from their indices.
AMBIGUOUS_NAMES = ["1", "2", "3"]


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("1", 1),
        ("index:1", 1),
        ("m001", 1),
        ("name:m001", 1),
    ]
)
def test_distinct_names(arg, expected):
    assert as_antenna_index(arg, DISTINCT_NAMES) == expected


def test_ambiguous_defaults_to_index():
    assert as_antenna_index("1", AMBIGUOUS_NAMES) == 1


def test_ambiguous_warns():
    messages = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        as_antenna_index("1", AMBIGUOUS_NAMES)
    finally:
        logger.remove(sink_id)
    assert any("ambiguous" in message for message in messages)


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("index:1", 1),
        ("name:1", 0),
    ]
)
def test_ambiguous_prefixes(arg, expected):
    assert as_antenna_index(arg, AMBIGUOUS_NAMES) == expected


def test_name_only_valid_as_name():
    # "3" is a name in AMBIGUOUS_NAMES but not a valid index for three
    # antennas, so it must resolve by name without a warning.
    assert as_antenna_index("3", AMBIGUOUS_NAMES) == 2


@pytest.mark.parametrize(
    "arg",
    ["m003", "name:m003", "index:3", "index:m000", "3", "-1"]
)
def test_invalid_selections_raise(arg):
    with pytest.raises(ValueError):
        as_antenna_index(arg, DISTINCT_NAMES)
