DATA = [
    ("The Eiffel Tower is in Paris. It was completed in 1889.",
     [("The tower is in Lyon.", False), ("It was completed in 1889.", True)]),
    ("Mercury is the closest planet to the Sun. Venus has a longer day than its year.",
     [("Mercury is the closest planet to the Sun.", True), ("Venus has a shorter day than its year.", False)]),
]
def to_prompt(passage, claims):
    lines = [f"Paragraph: '{passage}'", "For each claim, answer only True or False and cite the exact sentence."]
    for i,(c,_) in enumerate(claims,1): lines.append(f"{i}) {c}")
    return "\n".join(lines)
