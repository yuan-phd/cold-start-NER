"""Offset helper — auto-calculate entity offsets for hand-written gold test samples.

Usage:
    python scripts/offset_helper.py

Reads sample definitions from a simple format and outputs valid JSON
with correct start/end offsets. Validates all offsets before saving.
"""

import json
from pathlib import Path
from typing import Any

from src.data.validate import validate_sample


def create_sample(
    text: str,
    entities: list[tuple[str, str]],
    noisy: bool = False,
) -> dict[str, Any] | None:
    """Create a validated NER sample with auto-calculated offsets.

    Args:
        text: The full text string.
        entities: List of (entity_string, label) tuples.
            entity_string must appear exactly in text.
        noisy: If True, marks this sample as part of the noisy subset.

    Returns:
        Valid sample dict, or None if any entity string not found in text.
    """
    sample_entities = []
    search_from = 0

    for entity_text, label in entities:
        idx = text.find(entity_text, search_from)
        if idx == -1:
            # Try from beginning — but warn, this may indicate ordering issue
            idx = text.find(entity_text)
            if idx != -1:
                print(f"  WARNING: '{entity_text}' found at position {idx} (before search_from={search_from}) — verify this is the correct occurrence")
        if idx == -1:
            print(f"  ERROR: '{entity_text}' not found in text")
            print(f"  Text: '{text}'")
            return None

        # Check for duplicate occurrences
        second_idx = text.find(entity_text, idx + 1)
        if second_idx != -1:
            print(f"  WARNING: '{entity_text}' appears multiple times in text (positions {idx} and {second_idx}). Using first occurrence. Verify manually.")

        sample_entities.append({
            "text": entity_text,
            "label": label,
            "start": idx,
            "end": idx + len(entity_text),
        })

    sample: dict[str, Any] = {
        "text": text,
        "entities": sample_entities,
    }
    if noisy:
        sample["noisy"] = True

    # Validate
    is_valid, issues = validate_sample(sample)
    if not is_valid:
        print(f"  VALIDATION FAILED: {issues}")
        return None

    return sample


def build_gold_set() -> list[dict[str, Any]]:
    """Build the complete gold test set.

    Returns:
        List of validated NER samples.
    """
    # --- ANNOTATION CONVENTIONS ---
    # 1. SPELLING-OUT: Full continuous span including spelling-out portion.
    #    BIO/BIOES cannot represent discontinuous entities.
    # 2. FILLERS INSIDE ENTITIES: Include filler in span (continuous token stream).
    # 3. SELF-CORRECTIONS: Only the corrected (final) entity is annotated.
    # 4. STANDALONE NAME COMPONENTS: Non-adjacent name parts are separate entities
    #    or unlabeled. Only contiguous portions are annotated.
    # See ANNOTATION.md for full documentation.
    samples = []

    def add(text, entities, noisy=False):
        s = create_sample(text, entities, noisy)
        if s:
            samples.append(s)
        else:
            print(f"  SKIPPED sample #{len(samples)+1}")

    # =================================================================
    # ORIGINAL 20 hand-written samples
    # All samples defined in code — single source of truth, no file loading
    # =================================================================
    add(
        "yeah hi uh my name is bob martinez and i got this order number CT-55123 but the thing arrived broken",
        [("bob martinez", "NAME"), ("CT-55123", "CONTRACT_ID")],
    )
    add(
        "um so i bought the Samsung Galaxy S24 like three days ago and its already not working",
        [("Samsung Galaxy S24", "PRODUCT"), ("three days ago", "ISSUE_DATE")],
    )
    add(
        "hi this is sarah johnson calling about my account my email is sarah dot j at yahoo dot com",
        [("sarah johnson", "NAME"), ("sarah dot j at yahoo dot com", "EMAIL")],
    )
    add(
        "can you transfer me to someone else please i've been waiting for twenty minutes",
        [],
    )
    add(
        "ok so the order is ORD-2024-8891 and i need to return the Dyson V15 vacuum it was delivered on February 3rd",
        [("ORD-2024-8891", "CONTRACT_ID"), ("Dyson V15 vacuum", "PRODUCT"), ("February 3rd", "ISSUE_DATE")],
    )
    add(
        "my name is uh david no wait daniel allen and my contract id is HD-2025-03312",
        [("daniel allen", "NAME"), ("HD-2025-03312", "CONTRACT_ID")],
    )
    add(
        "i ordered a playstation 5 last tuesday and it still hasnt shipped yet my reference number is REF-44821",
        [("playstation 5", "PRODUCT"), ("last tuesday", "ISSUE_DATE"), ("REF-44821", "CONTRACT_ID")],
    )
    add(
        "yeah its priya patel p r i y a patel and you can reach me at priya.patel@outlook.com",
        [("priya patel", "NAME"), ("priya.patel@outlook.com", "EMAIL")],
    )
    add(
        "hold on let me find it um the the number is uh A1B2C3 i think thats right",
        [("A1B2C3", "CONTRACT_ID")],
    )
    add(
        "we bought the premium subscription plan about a week ago and want to cancel",
        [("premium subscription plan", "PRODUCT"), ("about a week ago", "ISSUE_DATE")],
    )
    add(
        "im calling because my iphone 15 pro screen cracked the day before yesterday",
        [("iphone 15 pro", "PRODUCT"), ("the day before yesterday", "ISSUE_DATE")],
    )
    add(
        "no i dont have the order number but my name is mohammed al rashid and my email is m.alrashid at gmail dot com",
        [("mohammed al rashid", "NAME"), ("m.alrashid at gmail dot com", "EMAIL")],
    )
    add(
        "yeah thank you for holding i appreciate your patience",
        [],
    )
    add(
        "the fiber optic 500 mbps plan has been having issues since end of last month and my account is SUB-33018",
        [("fiber optic 500 mbps plan", "PRODUCT"), ("end of last month", "ISSUE_DATE"), ("SUB-33018", "CONTRACT_ID")],
    )
    add(
        "hi um im liam obrien thats l i a m obrien and i need help with order TKT-90274 for the ring doorbell i got on 2024-09-15",
        [("liam obrien", "NAME"), ("TKT-90274", "CONTRACT_ID"), ("ring doorbell", "PRODUCT"), ("2024-09-15", "ISSUE_DATE")],
    )
    add(
        "its for the extended warranty package the one i got earlier this week",
        [("extended warranty package", "PRODUCT"), ("earlier this week", "ISSUE_DATE")],
    )
    add(
        "you can email me at carlos dot rivera at protonmail dot com my name is carlos rivera",
        [("carlos dot rivera at protonmail dot com", "EMAIL"), ("carlos rivera", "NAME")],
    )
    add(
        "i dont remember exactly when maybe sometime last week but the macbook air m3 just stopped turning on",
        [("sometime last week", "ISSUE_DATE"), ("macbook air m3", "PRODUCT")],
    )
    add(
        "please just transfer me already this is the third time ive called",
        [],
    )
    add(
        "my wifes name is ananya sharma and she placed the order its under her email ananya.sharma at gmail dot com for the nespresso vertuo on january 15th with order id CUS-88710",
        [("ananya sharma", "NAME"), ("ananya.sharma at gmail dot com", "EMAIL"), ("nespresso vertuo", "PRODUCT"), ("january 15th", "ISSUE_DATE"), ("CUS-88710", "CONTRACT_ID")],
    )

    # =================================================================
    # NEW CLEAN SAMPLES — targeting known gaps from EDA
    # =================================================================

    # --- Gap: Entity at text start (B6: only 4 entities at start) ---
    add(
        "john smith here i need help with my order",
        [("john smith", "NAME")],
    )
    add(
        "ORD-2024-7721 thats my order number and i want to know the status",
        [("ORD-2024-7721", "CONTRACT_ID")],
    )
    add(
        "samsung galaxy s24 thats what i ordered and it hasnt arrived",
        [("samsung galaxy s24", "PRODUCT")],
    )

    # --- Gap: NAME at text end (B5: 75% of NAMEs at start) ---
    add(
        "can you look up the account the name on it is maria garcia",
        [("maria garcia", "NAME")],
    )
    add(
        "the order was placed by my husband his name is william jackson",
        [("william jackson", "NAME")],
    )

    # --- Gap: CONTRACT_ID at text start ---
    add(
        "TKT-90274 is the ticket number i was given when i called last time about the ipad air",
        [("TKT-90274", "CONTRACT_ID"), ("ipad air", "PRODUCT")],
    )

    # --- Gap: All lowercase names (B7: 680 capitalized, 9 lowercase) ---
    add(
        "hi yeah its alice chen calling about my subscription",
        [("alice chen", "NAME")],
    )
    add(
        "this is carlos rivera and i need to update my email to carlos dot rivera at protonmail dot com",
        [("carlos rivera", "NAME"), ("carlos dot rivera at protonmail dot com", "EMAIL")],
    )

    # --- Gap: Adjacent entities (gap < 5 chars) ---
    add(
        "my name is linda white email linda.w@yahoo.com about order WO-67893",
        [("linda white", "NAME"), ("linda.w@yahoo.com", "EMAIL"), ("WO-67893", "CONTRACT_ID")],
    )
    add(
        "the fitbit charge 6 order REF-44821 from last friday",
        [("fitbit charge 6", "PRODUCT"), ("REF-44821", "CONTRACT_ID"), ("last friday", "ISSUE_DATE")],
    )

    # --- Gap: Long text (200+ chars) ---
    add(
        "ok so let me explain what happened i bought the dell xps 15 laptop about three weeks ago and it was working fine at first but then last monday morning the screen started flickering and now it wont turn on at all and i really need it for work so i need to either get a replacement or a refund my order number is PRJ-10284",
        [
            ("dell xps 15", "PRODUCT"),
            ("about three weeks ago", "ISSUE_DATE"),
            ("last monday morning", "ISSUE_DATE"),
            ("PRJ-10284", "CONTRACT_ID"),
        ],
    )

    # --- More diverse entity combinations ---
    add(
        "i signed up for the cloud storage 2tb plan on february 3rd and my account number is LIC-55901",
        [("cloud storage 2tb plan", "PRODUCT"), ("february 3rd", "ISSUE_DATE"), ("LIC-55901", "CONTRACT_ID")],
    )
    add(
        "yeah my email is d.allen@email.com and i want to cancel the annual maintenance contract",
        [("d.allen@email.com", "EMAIL"), ("annual maintenance contract", "PRODUCT")],
    )
    add(
        "i placed an order two weeks ago for a nintendo switch oled under the name fatima osei",
        [("two weeks ago", "ISSUE_DATE"), ("nintendo switch oled", "PRODUCT"), ("fatima osei", "NAME")],
    )

    # --- Negative samples (no entities) ---
    add(
        "yeah im still here just waiting can you check again please",
        [],
    )
    add(
        "ok thanks for letting me know is there anything else i should do",
        [],
    )

    # --- Ambiguous / tricky ---
    add(
        "the product is called ring doorbell and my name is also ring wei can you believe that",
        [("ring doorbell", "PRODUCT"), ("ring wei", "NAME")],
    )
    add(
        "i got it on the fifteenth no wait the sixteenth of march",
        [("the sixteenth of march", "ISSUE_DATE")],
    )

    # --- Gap: EMAIL underrepresented (12% vs 20%+ for other types) ---
    add(
        "you can reach me at karen.robinson@gmail.com if there are any updates on the delivery",
        [("karen.robinson@gmail.com", "EMAIL")],
    )
    add(
        "my email address is richard dot harris at protonmail dot com and the order is INV-2023-887",
        [("richard dot harris at protonmail dot com", "EMAIL"), ("INV-2023-887", "CONTRACT_ID")],
    )
    add(
        "w.jackson@email.com thats my email can someone get back to me about my refund",
        [("w.jackson@email.com", "EMAIL")],
    )
    add(
        "um yeah so my emale is like jess dot m at gmial dot com i always mess it up",
        [("jess dot m at gmial dot com", "EMAIL")],
        noisy=True,
    )
    add(
        "send the confirmation to to nhall at companey dot org thats n h a l l",
        [("nhall at companey dot org", "EMAIL")],
        noisy=True,
    )

    # =================================================================
    # NEW NOISY SAMPLES — written directly in degraded form
    # Marked as noisy=True for separate evaluation reporting
    # =================================================================

    # --- Character-level ASR errors ---
    add(
        "hi my naem is jennifer taylr and i need help with my ordr",
        [("jennifer taylr", "NAME")],
        noisy=True,
    )
    add(
        "the ordor numbr is O R D dash 2024 dash 5591 i thnk",
        [("O R D dash 2024 dash 5591", "CONTRACT_ID")],
        noisy=True,
    )
    add(
        "my emial is sarahj at yaho dot com",
        [("sarahj at yaho dot com", "EMAIL")],
        noisy=True,
    )

    # --- Filler-heavy + lowercase ---
    add(
        "um yeah so like my name is uh robert wilson and um i bought the uh sony wh one thousand xm five headphones",
        [("robert wilson", "NAME"), ("sony wh one thousand xm five headphones", "PRODUCT")],
        noisy=True,
    )
    add(
        "so basically i think it was uh like about a week ago maybe uh ten days ago i dont really remember",
        [("about a week ago", "ISSUE_DATE")],
        noisy=True,
    )

    # --- Word merging + stuttering ---
    add(
        "myname is is david anderson and myemail is david dot a at company dot com",
        [("david anderson", "NAME"), ("david dot a at company dot com", "EMAIL")],
        noisy=True,
    )
    add(
        "the the contract number is is SUB dash three three zero one eight",
        [("SUB dash three three zero one eight", "CONTRACT_ID")],
        noisy=True,
    )

    # --- Heavy noise: multiple error types combined ---
    add(
        "hi um its uh prya patel thats p r i y a and i orderd the the kindle paperwite like um three days ago",
        [("prya patel", "NAME"), ("kindle paperwite", "PRODUCT"), ("three days ago", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "yeah so my email is um mbrown fourty two at outlok dot com and the refrence number is ref dash four four eight two one",
        [("mbrown fourty two at outlok dot com", "EMAIL"), ("ref dash four four eight two one", "CONTRACT_ID")],
        noisy=True,
    )
    add(
        "i wanna return the the lg oled tv sixtyfive inch i got it on on uh novmber thirtieth",
        [("lg oled tv sixtyfive inch", "PRODUCT"), ("novmber thirtieth", "ISSUE_DATE")],
        noisy=True,
    )

    # --- Mixed clean and noisy in same sample ---
    add(
        "yeah so um this is betty young and i neeed to return the uh airpods pro i got them on like jan fifteenth my order is ACC-00192",
        [("betty young", "NAME"), ("airpods pro", "PRODUCT"), ("jan fifteenth", "ISSUE_DATE"), ("ACC-00192", "CONTRACT_ID")],
        noisy=True,
    )

    # SELF-CORRECTION (10 samples)

    # Self-correction: name
    add(
        "my name is uh david no wait daniel allen and my contract is ORD-2024-7712",
        [("daniel allen", "NAME"), ("ORD-2024-7712", "CONTRACT_ID")],
        noisy=True,
    )
    # Self-correction: email
    add(
        "my email is james at gmail no sorry james dot r at gmail dot com",
        [("james dot r at gmail dot com", "EMAIL")],
        noisy=True,
    )
    # Self-correction: contract ID
    add(
        "the order number is ORD dash 2024 dash um no wait its ORD dash 2025 dash 0091",
        [("ORD dash 2025 dash 0091", "CONTRACT_ID")],
        noisy=True,
    )
    # Self-correction: product
    add(
        "i bought the um samsung galaxy no actually it was the iphone fifteen pro max",
        [("iphone fifteen pro max", "PRODUCT")],
        noisy=True,
    )
    # Self-correction: date
    add(
        "it was on monday i think no wait it was tuesday the eighteenth",
        [("tuesday the eighteenth", "ISSUE_DATE")],
        noisy=True,
    )
    # Self-correction: name with typo in corrected version
    add(
        "its sarah no i mean lisa thompsen and i need help with my order",
        [("lisa thompsen", "NAME")],
        noisy=True,
    )
    # Self-correction: email with surrounding noise
    add(
        "yeah so um send it to like mjones at hotmal dot com no actually mjones at gmail dot com",
        [("mjones at gmail dot com", "EMAIL")],
        noisy=True,
    )
    # Self-correction: mid-sentence correction
    add(
        "i ordered a dyson v fifteen no sorry a dyson v twelve detect about two weeks ago",
        [("dyson v twelve detect", "PRODUCT"), ("two weeks ago", "ISSUE_DATE")],
        noisy=True,
    )
    # Self-correction: double correction
    add(
        "my name is john no wait james no sorry jason park and my email is jpark at yahoo dot com",
        [("jason park", "NAME"), ("jpark at yahoo dot com", "EMAIL")],
        noisy=True,
    )
    # Self-correction: implicit correction with "I mean"
    add(
        "the reference is SUB dash 44012 i mean SUB dash 44021 yeah thats right",
        [("SUB dash 44021", "CONTRACT_ID")],
        noisy=True,
    )

    # CHARACTER-LEVEL ERRORS (6 samples)

    add(
        "hi my name is micahel chen and i neeed to cancl my subscrption",
        [("micahel chen", "NAME")],
        noisy=True,
    )
    add(
        "the prodct i orderd was the nintendo swtich oled bundel",
        [("nintendo swtich oled bundel", "PRODUCT")],
        noisy=True,
    )
    add(
        "my emial adress is k dot wong at outllok dot com can you chek that",
        [("k dot wong at outllok dot com", "EMAIL")],
        noisy=True,
    )
    add(
        "i placeed the ordr on febuary twentyith i think",
        [("febuary twentyith", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "the refrence numbr is TKT dash nine oh two seven fohr",
        [("TKT dash nine oh two seven fohr", "CONTRACT_ID")],
        noisy=True,
    )
    add(
        "yeah its amanada garcia and i bougth a bose quietcomfrt headphnes",
        [("amanada garcia", "NAME"), ("bose quietcomfrt headphnes", "PRODUCT")],
        noisy=True,
    )

    # FILLER-HEAVY (6 samples)

    add(
        "so like um i was like wondering if you could like check on my order you know the um premium wireless charger",
        [("premium wireless charger", "PRODUCT")],
        noisy=True,
    )
    add(
        "um so basically uh my name is like uh tom brady you know and uh i need to uh return something",
        [("tom brady", "NAME")],
        noisy=True,
    )
    add(
        "uh yeah so um the email is uh rjohnson at uh proton mail dot com you know",
        [("rjohnson at uh proton mail dot com", "EMAIL")],
        noisy=True,
    )
    add(
        "mhm yeah so like it was um about uh three weeks ago i think yeah mhm",
        [("about uh three weeks ago", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "so um like the contract thing is um REQ dash uh 2025 dash like 1147 right",
        [("REQ dash uh 2025 dash like 1147", "CONTRACT_ID")],
        noisy=True,
    )
    add(
        "well um so like i got the uh apple watch ultra right and um it was like delivered on um march third i think",
        [("apple watch ultra", "PRODUCT"), ("march third", "ISSUE_DATE")],
        noisy=True,
    )

    # WORD MERGING / STUTTERING (5 samples)

    add(
        "myname is is rachel kim and myorder is is overdue",
        [("rachel kim", "NAME")],
        noisy=True,
    )
    add(
        "the the product was a a logitech mx master three mouse and it it broke",
        [("logitech mx master three mouse", "PRODUCT")],
        noisy=True,
    )
    add(
        "itslike the email is is jlee at at fastmail dot com",
        [("jlee at at fastmail dot com", "EMAIL")],
        noisy=True,
    )
    add(
        "i i ordered it on on the the fifteenth of of january",
        [("fifteenth of of january", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "the the order number number is RET dash 2025 dash dash 0087",
        [("RET dash 2025 dash dash 0087", "CONTRACT_ID")],
        noisy=True,
    )

    # SPELLING OUT (5 samples)
    # Convention: full continuous span including spelling-out (BIO limitation)

    # Standalone last name — first name spelled out separately, non-adjacent
    add(
        "my name is nguyen thats n g u y e n and first name is thi t h i",
        [("nguyen", "NAME")],
        noisy=True,
    )
    # Full span including spelling-out (convention: contiguous span required)
    add(
        "the email is asmith thats a s m i t h at gmail dot com",
        [("asmith thats a s m i t h at gmail dot com", "EMAIL")],
        noisy=True,
    )
    # Phonetic alphabet spelling for contract ID
    add(
        "order number is charlie tango dash 5 5 1 2 3 thats CT dash 55123",
        [("CT dash 55123", "CONTRACT_ID")],
        noisy=True,
    )
    # Full name with interleaved spelling — full continuous span
    add(
        "yeah its maria m a r i a santos s a n t o s calling about my order",
        [("maria m a r i a santos s a n t o s", "NAME")],
        noisy=True,
    )
    # Email with mid-span clarification
    add(
        "my email is p dot chen thats p as in peter dot c h e n at outlook dot com",
        [("p dot chen thats p as in peter dot c h e n at outlook dot com", "EMAIL")],
        noisy=True,
    )

    # ALL LOWERCASE (5 samples)

    add(
        "hi this is kevin martinez and i need to check on order ord-2024-6634",
        [("kevin martinez", "NAME"), ("ord-2024-6634", "CONTRACT_ID")],
        noisy=True,
    )
    add(
        "my email is slee at company dot org and i bought the playstation five digital edition",
        [("slee at company dot org", "EMAIL"), ("playstation five digital edition", "PRODUCT")],
        noisy=True,
    )
    add(
        "um yeah so its carlos ruiz and the order was placed on december tenth",
        [("carlos ruiz", "NAME"), ("december tenth", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "the contract number is sub dash 99281 and the product is a samsung galaxy tab s9",
        [("sub dash 99281", "CONTRACT_ID"), ("samsung galaxy tab s9", "PRODUCT")],
        noisy=True,
    )
    add(
        "i need to return the item my name is wei zhang email is wzhang at mail dot com order ref dash 10042",
        [("wei zhang", "NAME"), ("wzhang at mail dot com", "EMAIL"), ("ref dash 10042", "CONTRACT_ID")],
        noisy=True,
    )

    # ORAL FORMAT DATES (6 samples)

    add(
        "i bought it on like last monday morning and it stopped working by wednesday",
        [("last monday morning", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "it was delivered um maybe two or three weeks ago im not totally sure",
        [("two or three weeks ago", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "the purchase was on the twenty first of feb this year",
        [("twenty first of feb", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "uh i think it was like around christmas time maybe december twenty third or twenty fourth",
        [("december twenty third or twenty fourth", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "so i orderd it like the day before yesterday and its already broken can you believe that",
        [("the day before yesterday", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "it was sometime in early january like maybe the second or third i dont remember exactly",
        [("early january", "ISSUE_DATE")],
        noisy=True,
    )

    # ADJACENT ENTITIES UNDER NOISE (5 samples)

    add(
        "um this is anna lee alee at inbox dot com order TKT-2025-0003",
        [("anna lee", "NAME"), ("alee at inbox dot com", "EMAIL"), ("TKT-2025-0003", "CONTRACT_ID")],
        noisy=True,
    )
    add(
        "hi im omar hassan email ohassan at gmail dot com i bought the sony xm5 earbuds on feb third",
        [("omar hassan", "NAME"), ("ohassan at gmail dot com", "EMAIL"), ("sony xm5 earbuds", "PRODUCT"), ("feb third", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "james wu jameswu at tech dot org REQ-2024-8891 iphone sixteen pro march twelfth",
        [("james wu", "NAME"), ("jameswu at tech dot org", "EMAIL"), ("REQ-2024-8891", "CONTRACT_ID"), ("iphone sixteen pro", "PRODUCT"), ("march twelfth", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "yeah its uh nina patel nina dot p at yahoo dot com and the order is ORD dash 2025 dash 0412",
        [("nina patel", "NAME"), ("nina dot p at yahoo dot com", "EMAIL"), ("ORD dash 2025 dash 0412", "CONTRACT_ID")],
        noisy=True,
    )
    add(
        "so the prodct is a dell xps fifteen laptop order number RMA dash 70219 purchsed on jan ninth",
        [("dell xps fifteen laptop", "PRODUCT"), ("RMA dash 70219", "CONTRACT_ID"), ("jan ninth", "ISSUE_DATE")],
        noisy=True,
    )

    # LONG TEXTS WITH HEAVY NOISE (4 samples)

    add(
        "ok so um like hi my name is uh stephanie chen and like i bought this um samsung galaxy s24 ultra from your store like about two weeks ago and um the screen like started flickering and now it wont turn on and like i tried everything and my email is steph dot chen at gmail dot com and the order was um ORD dash 2025 dash 1234 i think",
        [("stephanie chen", "NAME"), ("samsung galaxy s24 ultra", "PRODUCT"), ("about two weeks ago", "ISSUE_DATE"), ("steph dot chen at gmail dot com", "EMAIL"), ("ORD dash 2025 dash 1234", "CONTRACT_ID")],
        noisy=True,
    )
    add(
        "yeah so basically um i called yesterday and nobody helped me right so my name is like marcus johnson and uh i ordered a dyson airwrap complete on um january fifteenth and the the order number is SUB dash 2025 dash 0088 and my email is mjohnson at protonmail dot com and i want a refund because it stopped working after like two days",
        [("marcus johnson", "NAME"), ("dyson airwrap complete", "PRODUCT"), ("january fifteenth", "ISSUE_DATE"), ("SUB dash 2025 dash 0088", "CONTRACT_ID"), ("mjohnson at protonmail dot com", "EMAIL")],
        noisy=True,
    )
    add(
        "hi um so uh this is really frustrating but um my name is aisha khan thats a i s h a and um i need to return the the bose quietcomfort ultra headphones i got on like march first from your website the order ref is TKT dash 2025 dash 0771 and my email is akhan at company dot co dot uk",
        [("aisha khan", "NAME"), ("bose quietcomfort ultra headphones", "PRODUCT"), ("march first", "ISSUE_DATE"), ("TKT dash 2025 dash 0771", "CONTRACT_ID"), ("akhan at company dot co dot uk", "EMAIL")],
        noisy=True,
    )
    add(
        "so like um ok i know this is complicated but bear with me right so i orderd two things a logitech mx keys keyboard and a logitech mx master mouse on uh feb twenty second and only the keyboard arrived and my name is pat rivera and email is privera at outlook dot com order is ORD dash 2025 dash 4421",
        [("pat rivera", "NAME"), ("logitech mx keys keyboard", "PRODUCT"), ("feb twenty second", "ISSUE_DATE"), ("privera at outlook dot com", "EMAIL"), ("ORD dash 2025 dash 4421", "CONTRACT_ID")],
        noisy=True,
    )

    # NEGATIVE SAMPLES UNDER NOISE (3 samples)

    add(
        "um yeah so like i was just wondering if you guys have like any sales going on right now you know",
        [],
        noisy=True,
    )
    add(
        "hi uh can you tell me what your return policy is i just wanna know before i buy anything",
        [],
        noisy=True,
    )
    add(
        "so like basically i just wanted to say that the customer service last time was really great you know mhm yeah thanks",
        [],
        noisy=True,
    )

    # --- Non-"is" NAME contexts (testing pattern generalization) ---
    add(
        "yeah so uh my colleague sarah chen she placed the order last week",
        [("sarah chen", "NAME"), ("last week", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "the account holder michael rivera called yesterday about the refund",
        [("michael rivera", "NAME"), ("yesterday", "ISSUE_DATE")],
        noisy=True,
    )
    add(
        "um could you look up the order for diana ross she needs to return the airpods max",
        [("diana ross", "NAME"), ("airpods max", "PRODUCT")],
        noisy=True,
    )
    add(
        "talking to you from the account of nina patel order REF dash 2025 dash 0093",
        [("nina patel", "NAME"), ("REF dash 2025 dash 0093", "CONTRACT_ID")],
        noisy=True,
    )
    add(
        "so the person who orderd it was um kwame osei and it was a sony wh one thousand xm five",
        [("kwame osei", "NAME"), ("sony wh one thousand xm five", "PRODUCT")],
        noisy=True,
    )
    add(
        "kwesi mensah here um i need to check on a return i started about three days ago",
        [("kwesi mensah", "NAME"), ("about three days ago", "ISSUE_DATE")],
        noisy=True,
    )

    return samples


def main() -> None:
    samples = build_gold_set()

    # Count stats
    clean = [s for s in samples if not s.get("noisy", False)]
    noisy = [s for s in samples if s.get("noisy", False)]

    print(f"\nGold test set: {len(samples)} total ({len(clean)} clean, {len(noisy)} noisy)")

    # Entity type coverage
    from collections import Counter
    all_entities = [e for s in samples for e in s.get("entities", [])]
    type_counts = Counter(e["label"] for e in all_entities)
    print(f"Entity coverage: {dict(type_counts)}")
    print(f"Negative samples: {sum(1 for s in samples if len(s.get('entities', [])) == 0)}")

    # Save
    output_path = Path("data/eval/gold_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved to {output_path}")

    # Final validation
    from src.data.validate import validate_file, print_report
    report = validate_file(output_path)
    print_report(report)


if __name__ == "__main__":
    main()