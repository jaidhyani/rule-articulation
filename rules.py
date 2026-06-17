"""Rule definitions for the pilot: programmatic label fn (where checkable),
human articulation, and generator instructions for natural-text positives/negatives."""

ANIMALS = {
    "ant","ape","bat","bear","bee","bird","boar","bug","cat","cow","crab","crow","deer","dog",
    "dove","duck","eel","elk","emu","ewe","fish","fly","fox","frog","goat","hare","hawk","hen",
    "hog","horse","kiwi","lamb","lion","lynx","mole","moth","mouse","mule","newt","owl","ox",
    "panda","pig","pony","pug","ram","rat","ray","seal","shark","sheep","slug","snail","snake",
    "sow","swan","tiger","toad","trout","tuna","wasp","wolf","worm","wren","yak","zebra",
    "cattle","camel","koala","otter","robin","goose","moose","whale","sloth","gecko","llama",
}

def _words(s):
    import re
    return re.findall(r"[A-Za-z]+", s)

def _alliterates(s):
    w = _words(s)
    return len(w) >= 2 and len({x[0].lower() for x in w}) == 1

def _acrostic_animal(s):
    w = _words(s)
    return len(w) >= 2 and "".join(x[0].lower() for x in w) in ANIMALS

class Rule:
    def __init__(self, rid, articulation, gen_pos, gen_neg, check=None):
        self.id = rid
        self.articulation = articulation      # ground-truth human statement
        self.gen_pos = gen_pos                 # generator instruction for True examples
        self.gen_neg = gen_neg                 # generator instruction for False examples
        self.check = check                     # label fn, or None for semantic-trust

RULES = {r.id: r for r in [
    Rule("lowercase",
         "The sentence is written entirely in lowercase.",
         "Every sentence is written entirely in lowercase letters, with NO capital letters anywhere. "
         "Vary the ending punctuation: about half should end with a period, question mark, or "
         "exclamation point, and about half with no terminal punctuation at all.",
         "Every sentence contains at least one capital letter somewhere. Vary the form so the capital "
         "is not always first: roughly half should start with a capital letter, and roughly half should "
         "begin with a lowercase word and place the capital later (e.g. a name or place mid-sentence, "
         "like 'i ran into Maria downtown'). Also vary ending punctuation: some with a period, some none.",
         check=lambda s: any(c.isalpha() for c in s) and s == s.lower()),
    Rule("has_digit",
         "The sentence contains at least one digit.",
         "Every sentence contains at least one numeric digit such as 3, 7, or 2024.",
         "No sentence contains any digit; write out any numbers as words.",
         check=lambda s: any(c.isdigit() for c in s)),
    Rule("question",
         "The sentence is a question (ends with '?').",
         "Every sentence is a question ending with a question mark.",
         "Every sentence is a plain statement ending with a period, never a question.",
         check=lambda s: s.strip().endswith("?")),
    Rule("animal",
         "The sentence mentions an animal.",
         "Every sentence mentions at least one animal (for example a dog, an owl, a salmon).",
         "No sentence mentions any animal whatsoever.",
         check=None),  # semantic — trust generator (optionally judge)
    Rule("alliteration",
         "Every word in the sentence starts with the same letter.",
         "Every word in the sentence starts with the same letter, e.g. 'Silly snakes slither slowly.' "
         "Keep it a natural-sounding short sentence.",
         "Write a normal short sentence whose words begin with several different letters.",
         check=_alliterates),
    Rule("acrostic_animal",
         "The first letters of the words, read in order, spell the name of an animal.",
         "Write a short natural sentence where the FIRST LETTERS of the words, read in order, spell "
         "the name of an animal. Example: 'Cats are tame' -> C,A,T -> CAT. Vary the animal.",
         "Write a short natural sentence whose words' first letters do NOT spell any animal name.",
         check=_acrostic_animal),
]}

# Pairs for the two-rule (correlated decoy) pilot.
PAIRS = [
    ("lowercase", "has_digit"),
    ("question", "animal"),
    ("lowercase", "animal"),
    ("has_digit", "alliteration"),
]
