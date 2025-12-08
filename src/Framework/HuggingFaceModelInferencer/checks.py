def checkIfHalfOfFileContentsIsReversed(questionsFileContents: str):
    """
    Checks whether exactly half of the lines are reversed sentence-pairs.
    If not, triggers a warning (input("Warning")).
    """
    lines = [line.strip() for line in questionsFileContents.splitlines() if line.strip()]

    straight_count = 0
    reversed_count = 0

    # Fast string-identification patterns
    straight_prefix = 'Does the word "'
    straight_mid = '" mean the same thing in sentences "'
    straight_sep = '" and "'
    straight_end = '"?'

    for line in lines:
        if line.startswith(straight_prefix) and straight_mid in line and line.endswith(straight_end):
            # Parse parts
            try:
                # Extract the two sentences in order
                # Format:
                # Does the word "{word}" mean the same thing in sentences "{A}" and "{B}"?
                rest = line[len(straight_prefix):]
                # rest = {word}" mean the same thing in sentences "{A}" and "{B}"?
                first_quote_end = rest.find('"')
                rest = rest[first_quote_end + 1:]  # after closing of word"
                # rest =  mean the same thing in sentences "{A}" and "{B}"?
                mid_idx = rest.find(straight_mid)
                if mid_idx == -1:
                    continue

                parts = rest.split(straight_mid)
                if len(parts) != 2:
                    continue
                right = parts[1]

                # Extract A and B
                # right = "{A}" and "{B}"?
                if straight_sep in right:
                    A, B = right.split(straight_sep)
                    # A = "{A}"
                    # B = "{B}"?
                    A = A.strip().strip('"')
                    B = B.strip().rstrip('?"').strip('"')

                    # Now detect reversed by checking if reversed pattern matches
                    # A straight pair has A then B
                    # A reversed pair has B then A
                    # So we detect reversed by looking at the order in the actual string

                    # We could detect reversed simply by checking the original string:
                    #   ... sentences "{sentence_b}" and "{sentence_a}"?
                    # So:
                    if f'sentences "{B}" and "{A}"?' in line:
                        reversed_count += 1
                    else:
                        straight_count += 1
            except Exception:
                # Ignore malformed lines
                continue

    # Optimal check: counts must match
    if straight_count != reversed_count:
        key = input("Warning: the number of straight and reversed sentences is not exactly the same. Do you wish to continue? (y/n)")
        if key == 'n':
            exit(0)
