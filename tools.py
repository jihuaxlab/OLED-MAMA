import re
from collections import defaultdict


def build_core_patterns_from_examples(examples):
    signature_to_examples = defaultdict(list)
    for s in examples:
        sig = _get_structure_signature(s)
        signature_to_examples[sig].append(s)

    patterns = []
    for sig, group in signature_to_examples.items():
        pat = _signature_to_core_regex(sig, group)
        if pat:
            patterns.append(re.compile(pat))
    return patterns


def _get_structure_signature(s):
    # Rule: if it's LETTERS followed by digits/S, treat suffix as 'D'
    if re.fullmatch(r'[A-Za-z]+[0-9S]*', s):
        match = re.match(r'^([A-Za-z]+)([0-9S]*)$', s)
        if match:
            letters, suffix = match.groups()
            if suffix:  # non-empty suffix
                return ('L', 'D')
            else:
                return ('L',)

    # Fallback: general tokenization
    tokens = re.findall(r'[A-Za-z]+|[0-9]+|[\[\]\-]', s)
    sig = []
    for t in tokens:
        if t.isalpha():
            sig.append('L')
        elif t.isdigit():
            sig.append('D')
        elif t in '[]-':
            sig.append(t)
        else:
            sig.append(t)
    return tuple(sig)


def _signature_to_core_regex(sig, examples):
    text = ''.join(examples)
    has_5 = '5' in text or '6' in text
    has_2 = '2' in text

    parts = []
    for token in sig:
        if token == 'L':
            parts.append(r'[A-Za-z]+')
        elif token == 'D':
            if has_5 and not has_2:
                parts.append(r'[0-9SlOo]+')
            elif has_2 and not has_5:
                parts.append(r'[0-9ZzlOo]+')
            elif has_2 and has_5:
                parts.append(r'[0-9SZzlOo]+')
            else:
                parts.append(r'[0-9lOo]+')
        elif token == '[':
            parts.append(r'[Il\[]')
        elif token == ']':
            parts.append(r'[IJ\]]')
        elif token == '-':
            parts.append(r'-')
        else:
            parts.append(re.escape(token))

    base = ''.join(parts)
    # return f'(?:{base}|{其他})'
    return base


# Rest of the code (is_candidate_valid, etc.) remains the same


def is_candidate_valid(candidate, core_patterns, max_extra=3):
    """
    Check whether the candidate contains a substring that matches the core_patterns, and the total number of additional characters before and after this substring is less than or equal to max_extra.
    """
    n = len(candidate)
    for pat in core_patterns:
        for match in pat.finditer(candidate):
            prefix_len = match.start()
            suffix_len = n - match.end()
            if prefix_len + suffix_len <= max_extra:
                return True, [candidate]
    if 'R' in candidate and '=' in candidate:
        return True, [candidate]
    if 'R' in candidate and ':' in candidate:
        return True, [candidate]
    if 'X' in candidate and '=' in candidate:
        return True, [candidate]
    if 'X' in candidate and ':' in candidate:
        return True, [candidate]
    if 'x' in candidate and '=' in candidate:
        return True, [candidate]
    if 'D' in candidate and '=' in candidate:
        return True, [candidate]
    if 'A' in candidate and '=' in candidate:
        return True, [candidate]
    if 'B' in candidate and '=' in candidate:
        return True, [candidate]
    # 
    matched = []
    if '-' in candidate or '=' in candidate:
        sub_candidates = [part.strip() for part in re.split(r'[-=\s]+', candidate) if part.strip()]
        for sub_candidate in sub_candidates:
            n = len(sub_candidate)
            for pat in core_patterns:
                for match in pat.finditer(sub_candidate):
                    prefix_len = match.start()
                    suffix_len = n - match.end()
                    if prefix_len + suffix_len <= max_extra:
                        matched.append(sub_candidate)
    matched = list(set(matched))
    return len(matched)>0, matched

# ===== Examples =====
if __name__ == "__main__":
    examples = ['Qx59', 'tBuTPA-BN', 'BN1', 'BN5', 'BB12[15]', 'Cat7', 'Molecule-[23]', 'X', '5TzPm-PXZ']
    examples = ['Qx51', '5TzPm-PXZ']
    examples = ['Qx52', 'Molecule-[25]']
    patterns = build_core_patterns_from_examples(examples)

    candidates = ['BNS', 'BB-I15J', 'QxSZ', 'tx199', 'tBuTPA-BN', '3.16', 'Molecule-I23]', ' 6Tz-PXZ', 'sfsd-fd']

    print("Patterns:")
    for p in patterns:
        print('  ', p)

    print("\nMatching candidates:")
    for c in candidates:
        print(c, is_candidate_valid(c, patterns, max_extra=3))