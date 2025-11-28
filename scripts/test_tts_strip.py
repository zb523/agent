#!/usr/bin/env python3
"""
Test cases for strip_tts_tags() function.
Run: uv run python scripts/test_tts_strip.py
"""
import sys
sys.path.insert(0, "src")

from agent import strip_tts_tags


def test_strip_tts_tags():
    """Test various tag stripping scenarios."""
    tests = [
        # (input, expected_output, description)
        (
            '[QURAN ref="1:1"]In the name of Allah[/QURAN]',
            'In the name of Allah',
            "Simple QURAN tag"
        ),
        (
            'Allah says: [QURAN ref="1:1"]In the name of Allah, the Entirely Merciful, the Especially Merciful.[/QURAN]',
            'Allah says: In the name of Allah, the Entirely Merciful, the Especially Merciful.',
            "QURAN tag with prefix"
        ),
        (
            '[All] praise is [due] to Allah',
            'All praise is due to Allah',
            "Bracket words only"
        ),
        (
            '[QURAN ref="1:1"][All] praise is [due] to Allāh, Lord of the worlds -[/QURAN]',
            'All praise is due to Allāh, Lord of the worlds -',
            "QURAN tag with bracket words inside"
        ),
        (
            'Allah says: [QURAN ref="1:1"]In the name of Allāh[/QURAN] [QURAN ref="1:2"][All] praise is [due] to Allāh[/QURAN]',
            'Allah says: In the name of Allāh All praise is due to Allāh',
            "Multiple QURAN tags"
        ),
        (
            '[QURAN ref="2:255-256"]Allah - there is no deity except Him[/QURAN]',
            'Allah - there is no deity except Him',
            "Range reference"
        ),
        (
            'This is regular text without any tags.',
            'This is regular text without any tags.',
            "No tags at all"
        ),
        (
            '',
            '',
            "Empty string"
        ),
        (
            '   [QURAN ref="1:1"]text[/QURAN]   ',
            'text',
            "Whitespace handling"
        ),
        (
            '[QURAN ref="1:1"]First[/QURAN] middle [QURAN ref="1:2"]Second[/QURAN]',
            'First middle Second',
            "Tags with text in between"
        ),
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("Testing strip_tts_tags()")
    print("=" * 60)
    
    for input_text, expected, description in tests:
        result = strip_tts_tags(input_text)
        if result == expected:
            print(f"✓ PASS: {description}")
            passed += 1
        else:
            print(f"✗ FAIL: {description}")
            print(f"  Input:    {repr(input_text)}")
            print(f"  Expected: {repr(expected)}")
            print(f"  Got:      {repr(result)}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = test_strip_tts_tags()
    sys.exit(0 if success else 1)

