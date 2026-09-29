import pytest
from loguru import logger
from quartical.config.converters import as_antenna_index


# Antenna names which cannot be confused with antenna indices.
DISTINCT_NAMES = ["m000", "m001", "m002"]

# Antenna names which are themselves integers, offset from their indices.
AMBIGUOUS_NAMES = ["1", "2", "3"]


def warnings_from(arg, ant_names):
    """Return the warnings emitted while resolving arg."""

    messages = []
    sink_id = logger.add(messages.append, level="WARNING")

    try:
        as_antenna_index(arg, ant_names)
    finally:
        logger.remove(sink_id)

    return messages


@pytest.mark.parametrize("names", [DISTINCT_NAMES, AMBIGUOUS_NAMES])
def test_int_is_an_index(names):
    assert as_antenna_index(1, names) == 1


def test_str_is_a_name():
    assert as_antenna_index("m001", DISTINCT_NAMES) == 1


def test_int_and_str_disagree_when_names_are_integers():
    # The index and the name are distinct selections and neither shadows the
    # other - this is the whole point of typing the option as Union[int, str].
    assert as_antenna_index(1, AMBIGUOUS_NAMES) == 1
    assert as_antenna_index("1", AMBIGUOUS_NAMES) == 0


def test_int_which_is_also_a_name_warns():
    messages = warnings_from(1, AMBIGUOUS_NAMES)
    assert any("both a valid antenna index and the name" in m
               for m in messages)


def test_int_which_is_not_a_name_is_silent():
    assert not warnings_from(1, DISTINCT_NAMES)


def test_str_is_never_ambiguous():
    assert not warnings_from("1", AMBIGUOUS_NAMES)


@pytest.mark.parametrize("arg", ["m003", "1", ""])
def test_unknown_names_raise(arg):
    with pytest.raises(ValueError, match="not found in the antenna table"):
        as_antenna_index(arg, DISTINCT_NAMES)


@pytest.mark.parametrize("arg", [3, -1])
def test_out_of_range_indices_raise(arg):
    with pytest.raises(ValueError, match="out of range"):
        as_antenna_index(arg, DISTINCT_NAMES)


def test_out_of_range_index_which_is_a_name_hints_at_quoting():
    # Names 1-3 occupy indices 0-2, so index 3 is out of range even though
    # "3" is a valid name. The error should say so.
    with pytest.raises(ValueError, match="Quote the value"):
        as_antenna_index(3, AMBIGUOUS_NAMES)
