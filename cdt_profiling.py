"""CDT profiling & wikification pipeline.

Parses storyline text into action series, builds CDT trees using canopy,
and generates LLM-powered wiki-style character profiles.

Usage::

    uv run python cdt_profiling.py \\
        --character Yui \\
        --engine claude-haiku-4-5 \\
        --surface_embedder_path ~/models/Qwen3-Embedding-0.6B \\
        --generator_embedder_path ~/models/Qwen3-0.6B \\
        --discriminator_path ~/models/deberta-v3-base-rp-nli \\
        --device_id 0
"""

from __future__ import annotations

import argparse
import logging
import re

import torch
from tqdm import tqdm

from canopy.core import CDTConfig, build_character_cdts
from canopy.embeddings import init_models as init_embedding_models
from canopy.llm import ClaudeCodeAdapter, generate, set_adapter
from canopy.validation import init_models as init_validation_models

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICL examples for storyline parsing (JoJo's Bizarre Adventure)
# ---------------------------------------------------------------------------

JOJO_CHARACTERS = ["Jotaro", "Polnareff", "Joseph", "DIO", "Kakyoin", "Avdol", "Iggy"]

ICL_INPUTS_RAW = [
    "In the year 1987, 17-year-old delinquent Jotaro Kujo, grandson of Joseph Joestar, and son of Holy Kujo, is detained after beating up three armed men and a trained boxer. Despite Holy's insistence, he refuses to leave his cell, claiming that he is possessed by an evil spirit. His prison mates confirm, desperate to get out of the shared cell.",
    "The guards notice Jotaro somehow drinking a can of beer and reading Weekly Shōnen Jump while listening to the radio within his cell, confused as to how he retrieved all those items. Jotaro claims the evil spirit brings things to him.",
    "Near the Canary Islands in 1983, four years earlier, off the coast of Africa, a group of fishermen uncover an extremely heavy 100-year-old coffin from the water, believing it to be a large treasure chest.",
    "In the present, Joseph Joestar arrives in Japan and is greeted by Holy. Joseph brings along a mysterious partner as the three head to the prison. The prison guards stumble upon Jotaro's prison cell filled with a considerable amount of other various objects, from books to jackets on coat hangers to even a guitar and a computer. Joseph confronts him and orders him to come out so that they can go home. Jotaro refuses at first due to his \"evil spirit\" and tears the little finger off of Joseph's prosthetic hand. Joseph is left surprised, as he didn't even notice when Jotaro snatched the finger.",
    "Joseph reveals his partner to be a fortune teller by the name of Muhammad Avdol, who he met three years ago in Egypt. Avdol reveals a spirit of his own called Magician's Red, which is a humanoid with a bird-like head and the ability to produce and control flames. Magician's Red launches a projectile of flames, pinning Jotaro against the back wall of the jail cell. Holy worries about what her father is doing to her son, while the guards are confused as to what's going on as they don't see any flames. Having no other choice, Jotaro's spirit confronts Magician's Red, fully manifesting itself.",
]

ICL_OUTPUTS = [
    """[Jotaro beats up three armed men and a trained boxer] (Jotaro)
[Jotaro is detained in prison] (Jotaro)
[Holy insists Jotaro leave his cell] (Other)
[Jotaro refuses to leave his cell] (Jotaro)
[Jotaro claims he is possessed by an evil spirit] (Jotaro)
[prison mates confirm Jotaro's claim, desperate to get out of the shared cell] (Other)""",
    """[guards notice Jotaro drinking a can of beer, reading Weekly Shōnen Jump, and listening to the radio within his cell] (Other)
[Jotaro drinks a can of beer] (Jotaro)
[Jotaro reads Weekly Shōnen Jump] (Jotaro)
[Jotaro listens to the radio] (Jotaro)
[Jotaro claims the evil spirit brings things to him] (Jotaro)""",
    """[A group of fishermen uncover an extremely heavy 100-year-old coffin from the water near the Canary Islands in 1983] (Other)
[The fishermen believe the coffin to be a large treasure chest] (Other)""",
    """[Joseph arrives in Japan and is greeted by Holy] (Joseph)
[Holy greets Joseph] (Other)
[Joseph brings along a mysterious partner] (Joseph)
[Joseph, Holy, and the partner head to the prison] (Joseph)
[prison guards stumble upon Jotaro's cell filled with various objects such as books, jackets, a guitar, and a computer] (Other)
[Joseph confronts Jotaro and orders him to come out so they can go home] (Joseph)
[Jotaro refuses to leave due to his "evil spirit"] (Jotaro)
[Jotaro tears the little finger off Joseph's prosthetic hand] (Jotaro)
[Joseph is left surprised, realizing he didn't even notice Jotaro snatch the finger] (Joseph)""",
    """[Joseph reveals his partner to be a fortune teller named Muhammad Avdol, whom he met three years ago in Egypt] (Joseph)
[Avdol reveals his spirit, Magician's Red, a humanoid with a bird-like head and the ability to produce and control flames] (Avdol)
[Magician's Red launches a projectile of flames, pinning Jotaro against the back wall of the jail cell] (Avdol)
[Holy worries about what her father is doing to her son] (Other)
[guards are confused as they don't see any flames] (Other)
[Jotaro's spirit confronts Magician's Red, fully manifesting itself] (Jotaro)""",
]


# ---------------------------------------------------------------------------
# Unique functions (not in canopy)
# ---------------------------------------------------------------------------


def fill_in_instruction(paragraph: str, main_characters: list[str]) -> str:
    """Build a storyline-parsing prompt for the given paragraph."""
    return f"""{paragraph}

Parse the paragraph above into a series of actions taken by characters, the format should be

[action 1] (character 1)
[action 2] (character 2)
[action 3] (character 3)
...

character should either be a main character {main_characters} or 'Other' or 'Environment'"""


def parse_scene_to_actions(text: str) -> list[dict[str, str]]:
    """Parse [action] (character) format into a list of dicts."""
    pattern = r"\[([^\[\]]+?)\] \((.*?)\)"
    results: list[dict[str, str]] = []
    for match in re.finditer(pattern, text):
        action = match.group(1).strip()
        character = match.group(2).strip()
        results.append({"action": action, "character": character})
    return results


def build_icl_prompt(main_characters: list[str]) -> str:
    """Build flattened ICL examples as a single prompt prefix.

    The legacy code used alternating assistant/user turns (reversed order),
    which violates Claude's strict alternation requirement. Instead, we
    flatten everything into a single prompt string.
    """
    icl_inputs = [fill_in_instruction(inp, JOJO_CHARACTERS) for inp in ICL_INPUTS_RAW]

    parts: list[str] = []
    for i, (inp, out) in enumerate(zip(icl_inputs, ICL_OUTPUTS), 1):
        parts.append(f"Example {i}:")
        parts.append(f"Input: {inp}")
        parts.append(f"Output:\n{out}")
        parts.append("")

    parts.append("Now parse this:")
    return "\n".join(parts)


def wikify_llm(
    character: str,
    content: str,
    cdt_tree_verbalized: str,
    attribute: str,
    note: str,
    lang: str = "English",
) -> str:
    """Generate a wiki-style section from CDT tree via LLM.

    This is NOT the same as canopy.wikify (which is deterministic markdown).
    This function uses the LLM to produce narrative wiki text.
    """
    prompt = f"""# Wiki:
{content}

# Pseudo code

{cdt_tree_verbalized}

# Task
The pseudo code above describes the character behavior logic of {character}. Summarize the information from the given code for "{attribute}" section in a wiki style to entail the given character Wiki.
{note}
Output in the following format (using {lang} and the name: {character}):
- {attribute} (The title should also be translated to target language: {lang}) -

{{Content}}"""

    return generate(prompt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="CDT / RP profiling & wikification pipeline",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="claude-haiku-4-5",
        help="The LLM engine for profiling",
    )

    # Wiki / language settings
    parser.add_argument(
        "--character_wiki_name",
        type=str,
        default="平泽唯",
        help="Target character name in wiki language",
    )
    parser.add_argument("--lang", type=str, default="Chinese", help="Output language")
    parser.add_argument(
        "--note",
        type=str,
        default="Yui -> 平泽唯; Ritsu -> 田井中律; Mio -> 秋山澪; Mugi -> 琴吹䌷; Azusa -> 中野梓",
        help="Name mapping or annotation note",
    )

    # Profiling targets
    parser.add_argument(
        "--profiling_character",
        type=str,
        default="Yui",
        help="Main character to profile",
    )
    parser.add_argument(
        "--main_characters",
        type=str,
        nargs="+",
        default=["Yui", "Ritsu", "Mio", "Mugi", "Azusa"],
        help="List of main characters",
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        default=["identity", "personality", "ability", "relationship"],
        help="List of main chapter topics",
    )

    # IO paths
    parser.add_argument(
        "--storyline_path",
        type=str,
        default="kon.storyline.part.txt",
        help="Input storyline file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="profiles/Yui.wikified.profile.txt",
        help="Output wikified profile path",
    )

    # Model paths
    parser.add_argument(
        "--surface_embedder_path",
        type=str,
        default="~/models/Qwen3-Embedding-0.6B",
        help="Path to surface embedding model",
    )
    parser.add_argument(
        "--generator_embedder_path",
        type=str,
        default="~/models/Qwen3-0.6B",
        help="Path to generative embedding model",
    )
    parser.add_argument(
        "--discriminator_path",
        type=str,
        default="~/models/deberta-v3-base-rp-nli",
        help="Path to NLI discriminator model",
    )

    # Device
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Device id, e.g. 0, 1",
    )

    return parser


def main() -> None:
    """Run the CDT profiling pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = build_arg_parser().parse_args()

    # 1. Configure LLM adapter
    set_adapter(ClaudeCodeAdapter(default_model=args.engine))

    # 2. Initialize embedding and validation models
    device = torch.device(f"cuda:{args.device_id}")
    log.info("Initializing embedding models on %s...", device)
    init_embedding_models(args.surface_embedder_path, args.generator_embedder_path, device)
    log.info("Initializing validation models on %s...", device)
    init_validation_models(args.discriminator_path, device)

    # 3. Parse storyline into action series
    with open(args.storyline_path, encoding="utf-8") as f:
        raw_storyline = f.read()

    paragraphs = raw_storyline.split("\n\n")
    paragraph_per_block = 3
    blocks = [
        "\n\n".join(paragraphs[idx : idx + paragraph_per_block])
        for idx in range(0, len(paragraphs), paragraph_per_block)
    ]

    icl_prefix = build_icl_prompt(args.main_characters)
    action_series: list[dict[str, str]] = []

    for block in tqdm(blocks[:10], desc="Parsing into action series...", leave=True):
        instruction = fill_in_instruction(block, args.main_characters)
        prompt = f"{icl_prefix}\n{instruction}"
        scene_text = generate(prompt)
        actions = parse_scene_to_actions(scene_text)
        action_series.extend(actions)

    # 4. Extract target character pairs
    profiling_character = args.profiling_character
    all_actions: list[str] = []
    pairs: list[dict[str, object]] = []
    last_character: list[str] = []

    for item in tqdm(action_series, desc="Extracting target actions...", leave=True):
        all_actions.append(item["action"])
        characters = [item["character"]]
        if profiling_character in characters:
            scene = "\n".join(all_actions[-11:-1])
            pairs.append(
                {
                    **item,
                    "characters": characters,
                    "scene": scene,
                    "last_character": last_character,
                }
            )
        last_character = characters

    # 5. Build CDTs using canopy
    other_characters = [c for c in args.main_characters if c != profiling_character]
    config = CDTConfig()
    topic2cdt, rel_topic2cdt = build_character_cdts(
        profiling_character, pairs, other_characters, config=config,
    )

    # 6. Wikify via LLM
    character_wiki_name = args.character_wiki_name
    content = character_wiki_name

    for topic, cdt in tqdm(topic2cdt.items(), desc="Wikifying attribute information..."):
        increment = wikify_llm(
            character_wiki_name,
            content,
            cdt.verbalize(),
            topic,
            args.note,
            args.lang,
        )
        content = "\n\n".join([content, increment])

    for rel_topic, cdt in tqdm(rel_topic2cdt.items(), desc="Wikifying interactive information..."):
        increment = wikify_llm(
            character_wiki_name,
            content,
            cdt.verbalize(),
            rel_topic,
            args.note,
            args.lang,
        )
        content = "\n\n".join([content, increment])

    # 7. Save output
    with open(args.output_path, "w", encoding="utf-8") as writer:
        writer.write(content)

    log.info("Profile written to %s", args.output_path)


if __name__ == "__main__":
    main()
