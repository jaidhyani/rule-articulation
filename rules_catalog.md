# Rule catalog

Each rule is a programmatic label function over a controlled input space, with a
known human articulation (ground truth for grading) and a scaling knob.

Two knob types:
- **execution knob** — turning it up degrades no-CoT *classification*, articulation stays easy
  ("easy to say, hard to do"). Use to map the >90% frontier.
- **description knob** — turning it up keeps execution a cheap local scan (classification
  survives) but degrades *articulation* (must enumerate more). These target the core question.

Input spaces:
- `rand_lower(L)` — random lowercase letters, length L
- `rand_mixed(L)` — letters + digits + punct + (optionally) unicode lookalikes
- `brackets(L, types, depth)` — bracket strings with filler
- `words(n)` — n random English words, space-joined
- `sentence()` — LLM-generated sentence (Opus 4.8 thinking-medium), labeled by judge or wordlist

---

## Family 1 — char-detection (membership)  [control end + a description knob]

| id | rule | knob | articulation |
|----|------|------|--------------|
| det-contains-digit | contains ≥1 digit | — | "contains a digit" |
| det-all-lower | every char is a lowercase letter | — | "is all lowercase" |
| det-forbidden-k | contains none of a fixed random set S, \|S\|=k | **description: k** | "contains none of: {S}" |
| det-required-k | every char is in alphabet A, \|A\|=k | **description: k** | "uses only the letters {A}" |

`det-forbidden-k` is the clean description-knob: k ∈ {1,2,3,5,8}. Execution stays a scan;
articulation must recover all k chars. Expect classification flat-high, articulation falling in k.

## Family 2 — char-arithmetic (single left-to-right scan → bool)

### 2a count-modulus  [execution knob: m]
Input `rand_lower(L)`. Rule: (#vowels) ≡ 0 (mod m). m ∈ {2,3,4,5}.
Articulation: "the number of vowels is divisible by m" ("even" for m=2).
Also sweep L ∈ {6,12,20} — execution degrades in both m and L.

### 2b count-threshold  [execution knob: T]
#vowels ≥ T, T ∈ {1,3,5,7}. Artic: "has at least T vowels."

### 2c consecutive-run  [execution knob: w]
contains w consecutive vowels (or consonants). w ∈ {2,3,4,5}. Artic: "has w vowels in a row."

### 2d positional-single  [execution knob: i]
char at position i is a vowel. i ∈ {1,3,7,12}. Artic: "the i-th character is a vowel."
(larger i = harder to locate no-CoT)

### 2e positional-multi  [description knob: number of positions]
chars at positions P (e.g. {2,5,9,...}) are all vowels; \|P\| ∈ {1,2,3,5}.
Artic: "the characters at positions {P} are all vowels." Execution local, articulation enumerates.

### 2f ends-palindrome  [mixed knob: k]
first k chars == reverse of last k chars. k ∈ {1,2,3}. k=1 → "first and last char match."
Articulation and execution both scale with k.

### 2g balanced-brackets  [execution knob: depth d / #types]
brackets balanced; variants: max nesting depth ≤ d (d ∈ {1,2,3}), or #bracket types
∈ {1,2,3} (1=counter, ≥2=stack). Artic: "the brackets are balanced."

### 2h periodic-case  [execution knob: period p]
case pattern repeats with period p. p=2 → alternating Aa; p∈{2,3,4}. Artic: "case alternates / repeats every p."

### 2i digit-sum-mod  [execution knob: m]
Input `rand_mixed`. sum of digits ≡ 0 (mod m). m ∈ {2,3,5}. Artic: "the digits sum to a multiple of m."

## Family 3 — semantic-detection  [control: articulation expected easy; knob = #conjuncts]

| id | rule | knob | articulation |
|----|------|------|--------------|
| sem-animal | mentions an animal | — | "mentions an animal" |
| sem-indoors | set indoors | — | "takes place indoors" |
| sem-neg-emotion | expresses a negative emotion | — | "expresses a negative emotion" |
| sem-conj-c | conjunction of c semantic conditions | **description: c** | "is X and Y and Z" |

Abstraction ladder (separate knob): dog → animal → living-thing.
Semantic is the "hard task, easy to name" control — shows dissociation is about *structure*, not difficulty.

## Family 4 — puzzle (multi-step composition)  [articulation-hard end]

| id | rule | knob | articulation |
|----|------|------|--------------|
| puz-acrostic-cat | first letter of each word spells a word in category C | C breadth | "first letters spell an animal" |
| puz-skip-k | every k-th character spells a word | skip k | "every k-th letter spells a word" |
| puz-wordlen-mono | word lengths strictly increasing | — | "each word is longer than the last" |
| puz-lastletter-chain | last letter of word i == first letter of word i+1 | — | "word-chain: each word starts with the previous word's last letter" |

Most puzzles likely fail Step-1 no-CoT classification — that's informative. Survivors are the
strongest dissociation candidates (executable as a local pattern, hard to name).

---

## Step-1 filter plan
Run classification (no-CoT) across all rules × knob values × a length sweep. Keep >90%
held-out survivors. Report survivor yield per family/knob — that table is itself a result.
Only survivors proceed to articulation (Step 2) and faithfulness (Step 3).
