from gdevsim.contacts import parse_contact_interface

import pytest

def test_parse_contact_interface_standard():
    # Test case with expected output
    contact_interface = "regionA@port1___regionB"
    expected = ("regionA@port1", "regionB", "port1")
    assert parse_contact_interface(contact_interface) == expected

def test_parse_contact_interface_reversed_regions():
    # Test case with reversed regions
    contact_interface = "regionB___regionA@port1"
    expected = ("regionA@port1", "regionB", "port1")
    assert parse_contact_interface(contact_interface) == expected

def test_parse_contact_interface_different_delimiters():
    # Test case with different delimiters
    contact_interface = "regionC#port2---regionD"
    expected = ("regionC#port2", "regionD", "port2")
    assert parse_contact_interface(contact_interface, interface_delimiter="---", contact_delimiter="#") == expected

def test_parse_contact_interface_missing_delimiters():
    # Test case with missing delimiters should raise an error
    contact_interface = "regionEregionF"
    with pytest.raises(ValueError):
        parse_contact_interface(contact_interface)


