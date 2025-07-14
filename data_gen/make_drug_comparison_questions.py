"""
python -m ipdb data_gen/make_drug_comparison_questions.py

The mesh hierarchy xml in `desc2025.xml` was downloaded from https://www.nlm.nih.gov/databases/download/mesh.html  and specifically from https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/?_gl=1*1y4y2dp*_ga*NjIwOTc1NzMxLjE3NDI3OTkxMzA.*_ga_7147EPK006*czE3NTIzNjMwNjAkbzkkZzEkdDE3NTIzNjQwMTEkajUxJGwwJGgw*_ga_P1FPTH9PL4*czE3NTIzNjMwNjAkbzkkZzEkdDE3NTIzNjQwMTEkajUxJGwwJGgw 
"""
import random
import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
from data_gen.api import call_llm_batch
from pathlib import Path
from data_gen.allMesh_to_parquet import return_indexer
import tqdm
import pickle
import ipdb
import xml.etree.ElementTree as ET
import pickle
from functools import lru_cache


def parse_json_from_llm_response(response: str):
    """
    Parse JSON from LLM response that might be wrapped in markdown code blocks.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Parsed JSON object/array or None if parsing fails
    """
    try:
        # Try to parse as-is first (in case it's clean JSON)
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from markdown-wrapped response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to extract JSON array from markdown-wrapped response
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # If all parsing attempts fail, return None
    return None


def sample_drug_abstracts(indexer,
                          drug_classes: List[str],
                          n_samples: int = 1000,
                          seed: int = 42) -> List[Dict]:
    """
    Sample abstracts that likely contain drug-related content based on MeSH terms.

    Note: I hardcoded a list of candidate drugs from an LLM bc we just needed a random range of drugs. 
    A nicer way to do this would be to use the drug classes

    Args:
        indexer: ArrowPMIDIndexer object
        n_samples: Number of abstracts to sample
        seed: Random seed for reproducibility

    Returns:
        List of sampled abstracts
    """
    random.seed(seed)
    # Collect article indices from each MeSH term
    all_drug_article_indices = set()

    print(
        f"Collecting articles from {len(drug_classes)} drug-related MeSH terms..."
    )
    for mesh_term in drug_classes:
        article_indices = indexer.get_articles_by_mesh_term(mesh_term)
        all_drug_article_indices.update(article_indices)

    print(f"Found {len(all_drug_article_indices)} drug-related articles")

    # Sample from the drug-related articles
    if len(all_drug_article_indices) > n_samples:
        sampled_indices = random.sample(list(all_drug_article_indices),
                                        n_samples)
    else:
        sampled_indices = list(all_drug_article_indices)
        print(
            f"Warning: Only found {len(sampled_indices)} articles, less than requested {n_samples}"
        )

    # Retrieve the articles
    drug_abstracts = [indexer.iloc(idx) for idx in sampled_indices]

    print(f"Sampled {len(drug_abstracts)} drug-related abstracts")
    return drug_abstracts


def create_property_extraction_prompt(abstracts: List[Dict]) -> str:
    """
    Create a prompt for the LLM to extract drug properties from abstracts.
    """
    texts = []
    for a in abstracts:
        texts.append(f"PMID: {a['pmid']}\n{a['title']}\n{a['abstractText']}")

    # Combine abstracts into a single text block
    abstracts_text = "\n\n---ABSTRACT---\n".join(texts)

    prompt = f"""Analyze the following PubMed abstracts and extract quantifiable properties that could be used to compare different drugs.

For each property found, provide:
1. Property name (standardized term)
2. Common keywords/phrases used to search for this property
3. Example of how it appears in the abstracts (with value and units)

Extract properties that are:
- Quantifiable with specific numerical values and units
- Meaningful for comparing drugs in the same therapeutic class
- Consistent enough across studies to enable fair comparison
- Relevant to clinical or scientific decision-making

GOOD properties to extract:

PHARMACOLOGICAL PROPERTIES:
- Pharmacokinetics: half-life, clearance, bioavailability, Cmax, Tmax, AUC
- Binding measures: Ki, Kd, IC50, EC50 (note the target when specified)
- Protein binding percentage
- Drug interactions: CYP inhibition constants

CLINICAL EFFICACY (when quantified):
- Response rates (specify condition and criteria)
- Time to response/remission
- Duration of effect
- Symptom score improvements (specify scale used)
- Disease-specific markers (e.g., HbA1c for diabetes, viral load for antivirals)
- Remission/cure rates

SAFETY METRICS (when quantified):
- Incidence of specific adverse events (as percentages)
- Discontinuation rates due to adverse events
- Specific lab value changes (e.g., QTc prolongation in ms)
- Dose-limiting toxicity thresholds

DO NOT extract:
- Vague outcomes without numbers (e.g., "improved outcomes")
- Study-specific composite endpoints unless clearly defined
- P-values or statistical measures alone
- Properties that vary wildly by study design

IMPORTANT: For clinical outcomes, include enough context to ensure fair comparison:
- Patient population (if specified)
- Dose/regimen (if relevant)
- Measurement timepoint

Abstracts:
{abstracts_text}

Return your analysis as a JSON array where each element has:
{{
  "property": "standardized property name",
  "search_keywords": ["keywords", "that", "would", "find", "this", "property"],
  "example": "example from abstracts showing the value"
}}

Only return the JSON array, no other text."""

    return prompt


def build_property_keyword_list(indexer,
                                drug_classes: List[str],
                                n_abstracts: int = 1000,
                                batch_size: int = 10) -> Dict[str, List[str]]:
    """
    Main function to build the keyword list for property extraction.
    
    Args:
        indexer: The ArrowPMIDIndexer object
        call_llm_batch: Function to call LLM in batch
        n_abstracts: Total number of abstracts to sample
        batch_size: Number of abstracts per LLM call
    
    Returns:
        Dictionary mapping properties to search keywords
    """
    # Step 1: Sample drug-related abstracts
    print("Step 1: Sampling drug-related abstracts...")
    sampled_abstracts = sample_drug_abstracts(indexer,
                                              drug_classes=drug_classes,
                                              n_samples=n_abstracts)

    # Step 2: Create prompts for batches
    print("\nStep 2: Creating prompts for LLM...")
    prompts = []
    for i in range(0, len(sampled_abstracts), batch_size):
        batch = sampled_abstracts[i:i + batch_size]
        prompt = create_property_extraction_prompt(batch)
        prompts.append(prompt)

    print(f"Created {len(prompts)} prompts")

    # Step 3: Call LLM in batch
    print("\nStep 3: Calling LLM to extract properties...")
    responses, _ = call_llm_batch(prompts,
                                  model_name="openai/gpt-4o-mini",
                                  temperature=0.1,
                                  max_tokens=2000)

    # Step 4: Parse and aggregate responses
    print("\nStep 4: Parsing LLM responses...")
    property_keywords = defaultdict(set)
    property_examples = defaultdict(list)

    for response in responses:
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                properties = json.loads(json_match.group())

                for prop in properties:
                    prop_name = prop['property'].lower()
                    # Add keywords
                    keywords = [
                        kw.lower() for kw in prop.get('search_keywords', [])
                    ]
                    property_keywords[prop_name].update(keywords)
                    # Add the property name itself as a keyword
                    property_keywords[prop_name].add(prop_name)
                    # Store examples
                    if 'example' in prop:
                        property_examples[prop_name].append(prop['example'])

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {e}")
            continue

    # Convert sets to lists
    property_keywords_dict = {
        prop: sorted(list(keywords))
        for prop, keywords in property_keywords.items()
    }

    # Print summary
    print(f"\nFound {len(property_keywords_dict)} unique properties")
    print("\nTop 10 properties by keyword count:")
    sorted_props = sorted(property_keywords_dict.items(),
                          key=lambda x: len(x[1]),
                          reverse=True)
    for prop, keywords in sorted_props[:10]:
        print(f"{prop}: {len(keywords)} keywords")
        print(f"  Keywords: {keywords[:5]}...")
        if prop in property_examples and property_examples[prop]:
            print(f"  Example: {property_examples[prop][0][:100]}...")

    return property_keywords_dict


def get_property_keyword_permutations(property_keywords_dict):
    """
    Get permutations of keywords for each property using an LLM to generate
    as many possible synonyms as possible for each property.
    """
    raise ValueError(
        "Decided to skip this step bc it introduced too much noise, and skipping it gives good-enough results"
    )
    print(
        f"Generating keyword permutations for {len(property_keywords_dict)} properties..."
    )

    # Create prompts for each property
    prompts = []
    property_names = list(property_keywords_dict.keys())

    for prop_name in property_names:
        existing_keywords = property_keywords_dict[prop_name]

        prompt = f"""You are a medical/pharmaceutical expert. Generate as many possible synonyms, abbreviations, and alternative phrasings as possible for the following drug property.

Property: "{prop_name}"
Existing synonyms: {existing_keywords}

Please generate additional synonyms that would help find this property in scientific literature. Include:
1. Common abbreviations and acronyms
2. Alternative scientific phrasings
3. Formal and informal terminology
4. Variations in word order
5. Singular and plural forms
6. Different units or measurement contexts
7. Related terms that might be used interchangeably

Return your response as a JSON array of strings, containing ONLY the new synonyms (don't repeat the existing keywords). Focus on terms that would likely appear in research papers when describing this property.

Example format:
["synonym1", "synonym2", "synonym3", ...]

Only return the JSON array, no other text."""

        prompts.append(prompt)

    # Call LLM in batch - one query per property
    print(f"Calling LLM for {len(prompts)} properties...")
    responses, _ = call_llm_batch(prompts,
                                  model_name="openai/gpt-4o-mini",
                                  temperature=0.3,
                                  max_tokens=1000)

    # Parse responses and merge with existing keywords
    expanded_dict = {}

    for i, (prop_name, response) in enumerate(zip(property_names, responses)):
        all_keywords = set(property_keywords_dict[prop_name])

        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                new_synonyms = json.loads(json_match.group())

                # Add new synonyms (convert to lowercase and filter out empty strings)
                new_synonyms = [
                    syn.lower().strip() for syn in new_synonyms if syn.strip()
                ]
                all_keywords.update(new_synonyms)
            else:
                pass

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing response for property '{prop_name}': {e}")

        expanded_dict[prop_name] = sorted(list(all_keywords))

    original_total = sum(
        len(keywords) for keywords in property_keywords_dict.values())
    expanded_total = sum(len(keywords) for keywords in expanded_dict.values())
    print(f"\nKeyword expansion summary:")
    print(f"Original total keywords: {original_total}")
    print(f"Expanded total keywords: {expanded_total}")
    print(f"Added {expanded_total - original_total} new keywords")

    # Show some examples
    print("\nExample expansions:")
    for prop_name in list(expanded_dict.keys())[:3]:
        original_count = len(property_keywords_dict[prop_name])
        expanded_count = len(expanded_dict[prop_name])
        print(f"  {prop_name}: {original_count} -> {expanded_count} keywords")

        # Show a few new keywords
        original_set = set(property_keywords_dict[prop_name])
        new_keywords = [
            k for k in expanded_dict[prop_name] if k not in original_set
        ]
        if new_keywords:
            print(f"    New: {new_keywords[:5]}...")

    return expanded_dict


def deduplicate_properties(
        results: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Use LLM to identify and merge duplicate property keys, keeping all search terms.
    
    Args:
        results: Dictionary mapping property names to lists of search keywords
        call_llm_batch: Function to call LLM in batch
    
    Returns:
        Deduplicated dictionary with merged properties
    """
    property_names = list(results.keys())
    print(f"Starting with {len(property_names)} properties")

    # Create a single prompt with all properties
    prompt = f"""Analyze these drug property names and identify ALL duplicates or near-duplicates.

Group properties that refer to the same concept (e.g., "ic50" and "inhibitory concentration (ic50)" are the same).

Properties to analyze:
{json.dumps(property_names, indent=2)}

Return a JSON object where:
- Keys are the standardized property names (choose the most common/clear version)
- Values are lists of all variations that should be merged into that property

Example format:
{{
  "ic50": ["ic50", "inhibitory concentration (ic50)", "half maximal inhibitory concentration"],
  "protein binding": ["protein binding percentage", "percentage of protein binding", "plasma protein binding"],
  "mic": ["mic", "minimum inhibitory concentration (mic)", "minimal inhibitory concentration (mic)"]
}}

Only include properties that have duplicates. Properties without duplicates don't need to be listed.
Return only the JSON object, no other text."""

    # Call LLM with single prompt
    responses, _ = call_llm_batch([prompt],
                                  model_name="openai/gpt-4o-mini",
                                  temperature=0.1,
                                  max_tokens=4000)

    # Parse response
    merge_mapping = {}  # Maps variant names to canonical names

    try:
        response = responses[0]
        # Extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            groups = json.loads(json_match.group())

            for canonical_name, variants in groups.items():
                canonical_lower = canonical_name.lower()
                for variant in variants:
                    variant_lower = variant.lower()
                    merge_mapping[variant_lower] = canonical_lower

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing response: {e}")
        return results  # Return original if parsing fails

    # Now merge the results
    deduplicated_results = {}
    processed_properties = set()

    for prop_name, keywords in results.items():
        prop_lower = prop_name.lower()

        # Skip if already processed as part of another group
        if prop_lower in processed_properties:
            continue

        # Get canonical name (or use current if not in mapping)
        canonical_name = merge_mapping.get(prop_lower, prop_lower)

        # Initialize with current keywords
        if canonical_name not in deduplicated_results:
            deduplicated_results[canonical_name] = []

        # Add all keywords from this property
        deduplicated_results[canonical_name].extend(keywords)
        processed_properties.add(prop_lower)

        # Find and merge all variants
        for other_prop, other_keywords in results.items():
            other_lower = other_prop.lower()
            if other_lower != prop_lower and merge_mapping.get(
                    other_lower) == canonical_name:
                deduplicated_results[canonical_name].extend(other_keywords)
                processed_properties.add(other_lower)

    # Add properties that weren't identified as duplicates
    for prop_name, keywords in results.items():
        prop_lower = prop_name.lower()
        if prop_lower not in processed_properties:
            deduplicated_results[prop_lower] = keywords

    # Remove duplicate keywords within each property
    for prop in deduplicated_results:
        deduplicated_results[prop] = list(
            dict.fromkeys(deduplicated_results[prop]))

    print(
        f"Reduced from {len(results)} to {len(deduplicated_results)} properties"
    )

    # Show some examples of merges
    print("\nExample merges:")
    merge_examples = []
    for variant, canonical in merge_mapping.items():
        if variant != canonical:
            merge_examples.append(f"  '{variant}' → '{canonical}'")

    # Show up to 10 examples
    for example in merge_examples[:10]:
        print(example)

    if len(merge_examples) > 10:
        print(f"  ... and {len(merge_examples) - 10} more merges")

    return deduplicated_results


def filter_properties_by_blacklist(properties, blacklist=None):
    """
    Filter out properties that are in the blacklist.
    
    Args:
        properties: Dict mapping property names to lists of synonyms
        blacklist: List of property names to filter out
    
    Returns:
        Filtered properties dictionary
    """
    if blacklist is None:
        blacklist = ["idr", "ki", "doses"]

    original_count = len(properties)
    filtered_properties = {
        k: v
        for k, v in properties.items() if k not in blacklist
    }

    filtered_count = original_count - len(filtered_properties)
    print(f"Filtered out {filtered_count} blacklisted properties: {blacklist}")
    print(f"Remaining properties: {len(filtered_properties)}")

    return filtered_properties


def get_mesh_ui_to_trees(path_to_mesh_hierarchy_xml="data/desc2025.xml"):
    import xml.etree.ElementTree as ET

    # Parse the MeSH XML once
    tree = ET.parse('desc2025.xml')
    root = tree.getroot()

    # Build ui_to_trees
    ui_to_trees = {}
    for rec in root.findall('DescriptorRecord'):
        ui = rec.findtext('DescriptorUI')
        trees = [tn.text for tn in rec.findall('TreeNumberList/TreeNumber')]
        ui_to_trees[ui] = trees

    # (Optionally) also build ui_to_name for printing
    ui_to_name = {
        rec.findtext('DescriptorUI'): rec.findtext('DescriptorName/String')
        for rec in root.findall('DescriptorRecord')
    }
    return ui_to_trees, ui_to_name


class MeshDrugUtils:
    """
    Helper for navigating MeSH “Chemicals & Drugs” (D-branch) descriptors and
    identifying *drug-class* nodes whose immediate children are individual
    drug leaves.

    A node is kept if it passes all automatic heuristics **or** is in the
    manual whitelist. It is filtered if it violates any heuristic **and** is
    not whitelisted, or if it appears in the manual blacklist.

    Heuristic defaults
    ------------------
    • Any tree-number starts with  D27.505  (pharmacologic actions)
    • Node has descendants (is not itself a leaf)
    • Depth ≥ 4   (≥ 3 dots in every tree-number)
    • # leaf children ∈ [3, 30]
    • Name is not black-listed, or else contains a strong mechanistic cue
    """

    # ------------------------------------------------------------------ #
    # Parse MeSH XML
    # ------------------------------------------------------------------ #
    def __init__(self, mesh_xml_path: str):
        print(f"Loading MeSH data from {mesh_xml_path} …")
        tree = ET.parse(mesh_xml_path)
        root = tree.getroot()

        self.ui_to_trees: Dict[str, list[str]] = {}
        self.ui_to_name: Dict[str, str] = {}

        for rec in root.findall('DescriptorRecord'):
            ui = rec.findtext('DescriptorUI')
            name = rec.findtext('DescriptorName/String')
            trees = [
                tn.text for tn in rec.findall('TreeNumberList/TreeNumber')
            ]
            self.ui_to_trees[ui] = trees
            self.ui_to_name[ui] = name

        self.all_trees: Set[str] = {
            t
            for ts in self.ui_to_trees.values()
            for t in ts
        }

        print(f"Parsed {len(self.ui_to_trees):,} descriptors and "
              f"{len(self.all_trees):,} tree numbers.")

    # ------------------------------------------------------------------ #
    # Drug-class picker with heuristics + black/white lists
    # ------------------------------------------------------------------ #
    @lru_cache(maxsize=None)
    def get_drug_classes(
        self,
        *,
        depth_min: int = 4,
        leaf_min: int = 3,
        leaf_max: int = 30,
        name_blacklist: List[str] = None,
        name_whitelist: List[str] = None
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Returns (kept, filtered) where each dict maps DescriptorUI → Name.
        """

        # -------- manual black- and white-lists --------
        if name_blacklist is None:
            name_blacklist = {
                # broad umbrellas to exclude outright
                "Anti-Infective Agents",
                "Central Nervous System Agents",
                "Autonomic Agents",
                "Cardiovascular Agents",
                "Micronutrients",
                "Growth Substances",
                "Sensory System Agents",
                "Respiratory System Agents",
                "Anti-Inflammatory Agents",
                "Anti-Bacterial Agents",
                "Adrenergic Agents",
                "Protease Inhibitors",
                # still-too-broad buckets after earlier pruning
                "Analgesics",
                "Anesthetics",
                "Antineoplastic Agents",
                "Central Nervous System Stimulants",
                "Central Nervous System Depressants",
                "Antimetabolites",
                "Gastrointestinal Agents",
                "Dermatologic Agents",
                "Psychotropic Drugs",
                "Neurotransmitter Uptake Inhibitors",
                "Dopamine Agents",
                "Hematologic Agents",
                "Immunologic Factors",
                "Neuromuscular Agents",
                "Tranquilizing Agents",
                "Antiviral Agents",
                "Histamine Antagonists",
                "Serotonin Agents",
                "GABA Agents",
                "Cholinergic Agents",
                "Peripheral Nervous System Agents",
                "Membrane Transport Modulators",
                "Enzyme Activators",
            }

        if name_whitelist is None:
            name_whitelist = {
                # tight mechanistic classes we always want
                "Cyclooxygenase Inhibitors",
                "Angiotensin-Converting Enzyme Inhibitors",
                "Adrenergic alpha-Antagonists",
                "Adrenergic alpha-Agonists",
                "Topoisomerase Inhibitors"
                # Major cardiovascular drug classes
                "Hydroxymethylglutaryl-CoA Reductase Inhibitors",
                "Calcium Channel Blockers",
                "Angiotensin Receptor Antagonists",

                # Specific antidepressant classes
                "Selective Serotonin Reuptake Inhibitors",
                "Serotonin and Noradrenaline Reuptake Inhibitors",
            }

        # -------- regex helpers --------
        GENERIC_BAD = re.compile(
            r'\b(Probes?|Solutions?|Uses|Kits|Mechanisms?)\b', re.I)
        KEEP_IF_HAS = re.compile(
            r'\b(Inhibitors?|Blockers?|Antagonists?|Agonists?|Statins?|'
            r'β[- ]?Blockers?|ACE|NSAID|COX|'
            r'Anti[- ](Inflammatory|HIV|Tubercular|Parkinson|Obesity|'
            r'Bacterial|Viral|Retrov?iral))', re.I)

        kept: Dict[str, str] = {}
        filtered: Dict[str, str] = {}

        for ui, trees in self.ui_to_trees.items():

            # 1) must sit in pharmacologic-action subtree and have descendants
            if not any(t.startswith('D27.505') for t in trees):
                continue
            if not any(
                    other.startswith(t + '.') for t in trees
                    for other in self.all_trees):
                continue

            name = self.ui_to_name[ui]

            # -------- whitelist overrides all remaining checks --------
            if name in name_whitelist:
                kept[ui] = name
                continue

            # 2) depth filter
            if not all(t.count('.') >= depth_min - 1 for t in trees):
                filtered[ui] = name
                continue

            # 3) leaf-count filter
            n_leaves = len(self.get_leaf_drug_uids(ui))
            if not (leaf_min <= n_leaves <= leaf_max):
                filtered[ui] = name
                continue

            # 4) manual blacklist
            if name in name_blacklist:
                filtered[ui] = name
                continue

            # 5) generic bucket check
            if GENERIC_BAD.search(name) and not KEEP_IF_HAS.search(name):
                filtered[ui] = name
                continue

            # passed everything
            kept[ui] = name

        print(
            f"Kept {len(kept)} drug classes "
            f"(depth≥{depth_min}, {leaf_min}–{leaf_max} leaves, name OK/whitelisted)."
        )
        print(f"Filtered out {len(filtered)} others.")
        return kept, filtered

    # ------------------------------------------------------------------ #
    # Leaf-drug helpers
    # ------------------------------------------------------------------ #
    @lru_cache(maxsize=None)
    def get_leaf_drug_uids(self, parent_ui: str) -> list[str]:
        """All descendant DescriptorUIs that are leaves beneath `parent_ui`."""
        parent_prefixes = self.ui_to_trees.get(parent_ui, [])

        subtree = [
            ui for ui, trees in self.ui_to_trees.items() if any(
                tc.startswith(tp + '.') for tp in parent_prefixes
                for tc in trees)
        ]
        leaves = [
            ui for ui in subtree if not any(
                other.startswith(t + '.') for t in self.ui_to_trees[ui]
                for other in self.all_trees)
        ]
        return leaves

    def get_leaf_drug_terms(self, parent_ui: str) -> Dict[str, str]:
        """Return {DescriptorUI: Name} for all leaf drugs under `parent_ui`."""
        return {
            ui: self.ui_to_name.get(ui, '')
            for ui in self.get_leaf_drug_uids(parent_ui)
        }


def get_article_pool(indexer, leaf_map):
    """
    Create a dict mapping each unique drug to a list of pubmed idxs from indexer.
    
    Args:
        indexer: ArrowPMIDIndexer object with get_articles_by_mesh_term method
        leaf_map: Dict mapping drug categories to dicts of {drug_uid: drug_name}
    
    Returns:
        Dict mapping drug_name to list of article indices
    """
    drugs_to_sampleidx = {}

    # Collect all unique drugs from all categories
    all_drugs = set()
    for category_drugs in leaf_map.values():
        for drug_uid, drug_name in category_drugs.items():
            # Assert that drug_uid looks like a MeSH term (starts with D and has numbers)
            assert drug_uid.startswith('D') and drug_uid[1:].isdigit(), \
                f"Drug UID {drug_uid} doesn't look like a MeSH term"
            all_drugs.add(drug_name)

    print(f"Found {len(all_drugs)} unique drugs across all categories")

    # For each unique drug, get the article indices
    for drug_name in tqdm.tqdm(all_drugs, desc="Getting articles for drugs"):
        article_indices = indexer.get_articles_by_mesh_term(drug_name)
        drugs_to_sampleidx[drug_name] = article_indices

    return drugs_to_sampleidx


def build_drug_property_paper_mapping(indexer, drugs_to_sampleidx, properties):
    """
    Create a mapping from drugs to properties to papers that contain those properties.
    
    Args:
        indexer: ArrowPMIDIndexer object
        drugs_to_sampleidx: Dict mapping drug names to lists of paper indices
        properties: Dict mapping property names to lists of synonyms
    
    Returns:
        Dict of {drug_name: {property_name: [paper_ids]}}
    """
    from collections import defaultdict

    drug_property_papers = defaultdict(lambda: defaultdict(list))

    print(
        f"Processing {len(drugs_to_sampleidx)} drugs, {len(properties)} properties..."
    )

    total_operations = sum(
        len(papers) for papers in drugs_to_sampleidx.values())
    print(f"Total papers to process: {total_operations}")

    processed_papers = 0

    for drug_name, paper_indices in tqdm.tqdm(drugs_to_sampleidx.items(),
                                              desc="Processing drugs"):
        for paper_idx in paper_indices:
            # Get the abstract text
            paper_data = indexer.iloc(paper_idx)
            abstract = paper_data.get('abstractText', '')

            if not abstract:
                continue

            abstract_lower = abstract.lower()

            # Check each property
            for property_name, synonyms in properties.items():
                # Check if any synonym appears in the abstract
                property_found = False
                for synonym in synonyms:
                    if synonym.lower() in abstract_lower:
                        drug_property_papers[drug_name][property_name].append(
                            paper_idx)
                        property_found = True
                        break  # Found this property, move to next property

            processed_papers += 1

    # Convert to regular dict for cleaner output
    result = {}
    for drug_name, property_dict in drug_property_papers.items():
        result[drug_name] = dict(property_dict)

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total drugs processed: {len(result)}")

    total_drug_property_pairs = sum(len(props) for props in result.values())
    print(f"Total drug-property pairs: {total_drug_property_pairs}")

    # Show some examples
    print(f"\nExample results:")
    for drug_name in list(result.keys())[:3]:
        print(f"{drug_name}: {len(result[drug_name])} properties found")
        for prop_name in list(result[drug_name].keys())[:3]:
            paper_count = len(result[drug_name][prop_name])
            print(f"  {prop_name}: {paper_count} papers")

    return result


def generate_valid_drug_comparisons(leaf_map,
                                    drug_property_papers,
                                    drug_class_names,
                                    min_papers=1):
    valid_comparisons = []

    for drug_class_uid, drugs_dict in leaf_map.items():
        drug_names = list(drugs_dict.values())

        # Generate all pairs within this drug class
        for i in range(len(drug_names)):
            for j in range(i + 1, len(drug_names)):
                drug_a, drug_b = drug_names[i], drug_names[j]

                # Find shared properties with sufficient evidence
                if drug_a in drug_property_papers and drug_b in drug_property_papers:
                    shared_props = set(
                        drug_property_papers[drug_a].keys()) & set(
                            drug_property_papers[drug_b].keys())

                    for prop in shared_props:
                        a_papers = drug_property_papers[drug_a][prop]
                        b_papers = drug_property_papers[drug_b][prop]

                        if len(a_papers) >= min_papers and len(
                                b_papers) >= min_papers:
                            valid_comparisons.append({
                                "drug_class":
                                drug_class_names[drug_class_uid],
                                "drug_class_uid":
                                drug_class_uid,
                                "drug_a":
                                drug_a,
                                "drug_b":
                                drug_b,
                                "property":
                                prop,
                                "drug_a_papers":
                                a_papers,
                                "drug_b_papers":
                                b_papers,
                                "drug_a_count":
                                len(a_papers),
                                "drug_b_count":
                                len(b_papers)
                            })

    return valid_comparisons


def create_question_generation_prompt(drug_a,
                                      drug_b,
                                      property,
                                      papers_a_indexes,
                                      papers_b_indexes,
                                      drug_class,
                                      indexer,
                                      max_papers=50):
    """
    Create prompt for LLM to generate and validate a comparison question.
    
    Args:
        drug_a, drug_b: Drug names
        property: Property to compare
        papers_a_indexes: List of paper indexes for drug A
        papers_b_indexes: List of paper indexes for drug B
        drug_class: The therapeutic class both drugs belong to
        indexer: ArrowPMIDIndexer object to retrieve paper abstracts
    
    Returns:
        tuple: (prompt_string, metadata_dict)
    """

    # Retrieve paper abstracts from indexes
    papers_a = []
    pmids_a = []
    for idx in papers_a_indexes[:max_papers]:  # Limit to 5 papers
        try:
            paper_data = indexer.iloc(idx)
            papers_a.append({
                'pmid':
                paper_data.get('pmid', 'Unknown'),
                'title':
                paper_data.get('title', 'No title available'),
                'abstract':
                paper_data.get('abstractText', 'No abstract available')
            })
            pmids_a.append(paper_data.get('pmid', 'Unknown'))
        except Exception as e:
            print(f"Error retrieving paper {idx}: {e}")
            continue

    papers_b = []
    pmids_b = []
    for idx in papers_b_indexes[:max_papers]:
        try:
            paper_data = indexer.iloc(idx)
            papers_b.append({
                'pmid':
                paper_data.get('pmid', 'Unknown'),
                'title':
                paper_data.get('title', 'No title available'),
                'abstract':
                paper_data.get('abstractText', 'No abstract available')
            })
            pmids_b.append(paper_data.get('pmid', 'Unknown'))
        except Exception as e:
            print(f"Error retrieving paper {idx}: {e}")
            continue

    # Format abstracts for each drug (now including titles)
    abstracts_a = "\n\n".join([
        f"[Paper A{i+1}] PMID: {paper['pmid']}\nTitle: {paper['title']}\nAbstract: {paper['abstract']}"
        for i, paper in enumerate(papers_a)
    ])

    abstracts_b = "\n\n".join([
        f"[Paper B{i+1}] PMID: {paper['pmid']}\nTitle: {paper['title']}\nAbstract: {paper['abstract']}"
        for i, paper in enumerate(papers_b)
    ])

    prompt = f"""You are creating a drug comparison question for a scientific QA dataset. 

DRUGS TO COMPARE:
- Drug A: {drug_a}
- Drug B: {drug_b}
- Drug Class: {drug_class}
- Property: {property}

ABSTRACTS FOR {drug_a.upper()}:
{abstracts_a}

ABSTRACTS FOR {drug_b.upper()}:
{abstracts_b}

TASK:
1. VERIFY that each abstract actually contains a drug from the specified class
2. Extract specific NUMERICAL values for "{property}" for both drugs
3. Ensure the comparison is scientifically valid

VALIDATION CHECKLIST (ALL must be true for valid comparison):
□ A specific drug from Drug A class is named in Paper A
□ A specific drug from Drug B class is named in Paper B  
□ Both papers provide numerical values (not "increased" or "decreased")
□ Units are identical OR convertible (specify conversion if needed)
□ Similar experimental conditions (species, disease model, route)
□ Values measure the same thing (not mixing absolute vs percentage change)
□ The property measured matches what was requested

OUTPUT FORMAT (JSON):
{{
  "is_valid_comparison": true/false,
  "validation_notes": "Explanation of why comparison is/isn't valid",
  "drug_a_details": {{
    "drug_name": "specific drug name from abstract",
    "value": "numerical value with units",
    "context": "species, condition, measurement method",
    "source_paper": "Paper A1/A2/etc"
  }},
  "drug_b_details": {{
    "drug_name": "specific drug name from abstract",
    "value": "numerical value with units",
    "context": "species, condition, measurement method",
    "source_paper": "Paper B1/B2/etc"
  }},
  "question": "Which [drug class] has [higher/lower/better] [property]: {drug_a} or {drug_b}?",
  "answer": "{drug_a} or {drug_b}",
  "explanation": "Brief explanation including actual drug names and values"
}}

COMMON ERRORS TO AVOID:
- Extracting disease state values instead of drug treatment effects
- Comparing the same drug appearing in both categories
- Missing that no drug from the specified class is mentioned
- Accepting qualitative descriptions as valid comparisons
- Mixing different types of measurements (e.g., peak vs AUC)

If ANY validation check fails, set is_valid_comparison to false and explain which checks failed."""

    # Create metadata dictionary
    metadata = {
        'inputs': {
            'drug_a': drug_a,
            'drug_b': drug_b,
            'property': property,
            'drug_class': drug_class,
            'papers_a_indexes': papers_a_indexes[:5],  # Only the ones we used
            'papers_b_indexes': papers_b_indexes[:5]
        },
        'pmids': {
            'drug_a_pmids': pmids_a,
            'drug_b_pmids': pmids_b
        },
        'paper_counts': {
            'drug_a_papers_used': len(papers_a),
            'drug_b_papers_used': len(papers_b),
            'drug_a_papers_available': len(papers_a_indexes),
            'drug_b_papers_available': len(papers_b_indexes)
        }
    }

    return prompt, metadata


def generate_drug_comparison_questions(valid_comparisons,
                                       indexer,
                                       max_examples=1000):
    """
    Generate drug comparison questions for a set of valid comparisons.
    
    Args:
        valid_comparisons: List of valid comparison dictionaries
        indexer: ArrowPMIDIndexer object
        max_examples: Maximum number of examples to process
    
    Returns:
        Dict containing responses, metadata, and cost information
    """
    assert len(
        valid_comparisons
    ) < max_examples, f"Number of examples ({len(valid_comparisons)}) must be less than {max_examples}"

    # Create prompts for each comparison
    prompts = []
    metadata_list = []

    print(f"Generating prompts for {len(valid_comparisons)} comparisons...")

    for comparison in valid_comparisons:
        prompt, metadata = create_question_generation_prompt(
            drug_a=comparison['drug_a'],
            drug_b=comparison['drug_b'],
            property=comparison['property'],
            papers_a_indexes=comparison['drug_a_papers'],
            papers_b_indexes=comparison['drug_b_papers'],
            drug_class=comparison['drug_class'],
            indexer=indexer)
        prompts.append(prompt)
        metadata_list.append(metadata)

    print(f"Generated {len(prompts)} prompts for LLM processing...")

    # Call LLM in batch
    responses, cost = call_llm_batch(prompts,
                                     model_name="openai/gpt-4o",
                                     temperature=0.1,
                                     json_mode=True,
                                     max_tokens=1000)

    # Parse JSON responses using utility function
    parsed_responses = []
    for response in responses:
        parsed_json = parse_json_from_llm_response(response)
        if parsed_json is None:
            print(f"Failed to parse JSON from response: {response[:200]}...")
        parsed_responses.append(parsed_json)

    # Package results
    results = {
        'responses': responses,  # Keep raw responses for debugging
        'parsed_responses': parsed_responses,  # Add parsed JSON
        'metadata': metadata_list,
        'cost': cost,
        'num_examples': len(valid_comparisons),
        'original_comparisons': valid_comparisons
    }

    print(f"Processed {len(responses)} responses with total cost: {cost}")
    print(
        f"Successfully parsed {len([r for r in parsed_responses if r is not None])} JSON responses"
    )

    ipdb.set_trace()
    pass

    return results


def create_drug_comparison_dataset(indexer,
                                   path_mesh_xml="data/desc2025.xml",
                                   random_state=0):
    """ 
    
    First we get `drug_classes` - about 200 of them. 
    These classes are groupings of drugs that are similar in some way. 
    In terms of Mesh, they are all in the subtree starting with 'D27.505' - "Chemical Actions and Uses – Pharmacologic Actions". 
    AND they are nodes whose children are leaves that are also specific drugs. 
    E.g. {
        'D057847': 'Lipid Regulating Agents',
        'D057911': 'Angiotensin Receptor Antagonists',
        'D057947': 'Parenteral Nutrition Solutions',
        ...
        }
    """
    if not Path(path_mesh_xml).exists():
        raise ValueError(f"Need to download mesh xml file into {path_mesh_xml}" \
                         " from https://www.nlm.nih.gov/databases/download/mesh.html")

    mesh_utils = MeshDrugUtils(path_mesh_xml)

    # get list of all drug drug classes: these are all Mesh terms
    fname_drug_classes = 'data/drug_classes.json'
    if not Path(fname_drug_classes).exists():
        drug_classes, filtered_drug_classes = mesh_utils.get_drug_classes()
        with open(fname_drug_classes, 'w') as f:
            json.dump(
                {
                    'drug_classes': drug_classes,
                    'filtered_drug_classes': filtered_drug_classes
                }, f)
    else:
        assert Path(fname_drug_classes).exists()
        with open(fname_drug_classes, 'r') as f:
            data = json.load(f)
            drug_classes = data['drug_classes']
            filtered_drug_classes = data['filtered_drug_classes']

    # leaves = mesh_utils.get_leaf_drug_terms(key)
    leaf_map = {
        cat: mesh_utils.get_leaf_drug_terms(cat)
        for cat in tqdm.tqdm(drug_classes)
    }

    # get article pool for each drug
    drugs_to_sampleidx = get_article_pool(indexer, leaf_map)

    # get 'keywords' for properties - get synonyms for each property as well
    fname_keywords = 'data/property_keywords_drug_comparison.json'
    if not Path(fname_keywords).exists():
        # if 1:
        properties_all = build_property_keyword_list(
            indexer, list(drug_classes.values()), n_abstracts=1000)
        properties_dedupe = deduplicate_properties(properties_all)
        properties = properties_dedupe

        with open(fname_keywords, 'w') as f:
            json.dump(properties, f)
    else:
        assert Path(fname_keywords).exists()
        with open(fname_keywords, 'r') as f:
            properties = json.load(f)
    properties = filter_properties_by_blacklist(properties)

    # create the drug_property_papers mapping
    output_file = 'data/drug_property_papers.pkl'
    if not Path(output_file).exists():
        # if 1:
        drug_property_papers = build_drug_property_paper_mapping(
            indexer, drugs_to_sampleidx, properties)
        print(f"Saving drug_property_papers to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(drug_property_papers, f)
        print(f"Saved successfully!")
    else:
        with open('data/drug_property_papers.pkl', 'rb') as f:
            drug_property_papers = pickle.load(f)

    # generate valid comparisons
    output_file = 'data/valid_drug_comparisons.pkl'
    if not Path(output_file).exists():
        # if 1:
        valid_comparisons = generate_valid_drug_comparisons(
            leaf_map, drug_property_papers, drug_classes, min_papers=1)
        with open(output_file, 'wb') as f:
            pickle.dump(valid_comparisons, f)
        print(f"Saved valid comparisons to {output_file}")
    else:
        with open(output_file, 'rb') as f:
            valid_comparisons = pickle.load(f)

    # Generate drug comparison questions for the first 100 examples
    print(
        f"Found {len(valid_comparisons)} valid comparisons. Processing random sample of 200..."
    )

    # Set random seed for reproducibility
    random.seed(random_state)

    # Take a random sample of 200 examples (or all if fewer than 200)
    if len(valid_comparisons) > 1000:
        examples_to_process = random.sample(valid_comparisons, 200)
    else:
        examples_to_process = valid_comparisons
        print(
            f"Warning: Only {len(valid_comparisons)} comparisons available, using all of them."
        )

    # Generate questions using the new function
    question_results = generate_drug_comparison_questions(examples_to_process,
                                                          indexer,
                                                          max_examples=1000)

    # Save the results
    output_file = 'data/drug_comparison_questions.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(question_results, f)
    print(f"Saved question generation results to {output_file}")

    ipdb.set_trace()
    pass


if __name__ == "__main__":
    path_mesh_xml = 'data/desc2025.xml'
    indexer = return_indexer('data/allMeSH_2022.parquet')
    # indexer = None
    create_drug_comparison_dataset(indexer, path_mesh_xml, random_state=0)
    ipdb.set_trace()
    pass
