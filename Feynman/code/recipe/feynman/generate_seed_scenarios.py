"""Generate the first 100 source-grounded Feynman26 scenarios.

This pilot deliberately stores paraphrased evidence rather than copyrighted source
passages.  Four derivative tasks are made from each of 25 source cards.  The source
ID remains in every task ID so downstream splits can keep each source group intact.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.feynman.schema import Scenario


@dataclass(frozen=True)
class SourceCard:
    source_id: str
    title: str
    year: int
    mode: str
    topic: str
    principle: str
    fact: str
    questions: tuple[str, str, str, str]


CARDS = (
    SourceCard("F001", "The Feynman Lectures on Physics, Volume I", 1963, "historical_simulation", "mechanics", "Explain physical laws through concrete predictions and limiting cases, not definitions alone.", "A useful physical model connects mathematical quantities to observable motion and states its approximations.", ("A student says a formula is an explanation by itself. Show what is missing.", "Why do physicists test a mechanics formula in extreme or simple cases?", "A calculation gives a precise answer from rough assumptions. How should the result be presented?", "Explain why naming a force is not yet a mechanism.")),
    SourceCard("F002", "The Feynman Lectures on Physics, Volume II", 1964, "historical_simulation", "electromagnetism", "Use fields as predictive physical bookkeeping and connect them to forces and energy flow.", "Electric and magnetic fields are operationally tied to forces on charges and can carry energy.", ("A student asks whether an electric field is merely a drawing of arrows. Explain its physical content.", "Where does energy travel in a simple electrical circuit?", "Why can a conductor rearrange charge even when its net charge is zero?", "Explain what a field description buys us compared with action at a distance.")),
    SourceCard("F003", "The Feynman Lectures on Physics, Volume III", 1965, "historical_simulation", "quantum mechanics", "Start from measurable alternatives and amplitudes; resist forcing classical pictures where experiments do not support them.", "Quantum alternatives combine as amplitudes, and measurement arrangements determine which distinctions are physically available.", ("Which slit did an unmeasured electron really pass through?", "Why can probabilities not simply be added in every quantum experiment?", "A student wants a tiny classical orbit for an atomic electron. Diagnose the assumption.", "Explain why changing the detector can change the pattern without invoking consciousness.")),
    SourceCard("F004", "Space-Time Approach to Non-Relativistic Quantum Mechanics", 1948, "historical_simulation", "path integrals", "Represent quantum propagation by combining amplitudes associated with alternative paths.", "The path-integral formulation sums complex amplitudes over alternatives rather than selecting one unobserved classical trajectory.", ("Does the path integral say a particle secretly chooses one path?", "Why do wildly nonclassical paths not dominate ordinary large-scale motion?", "Explain the difference between summing probabilities and summing amplitudes.", "What experimental question would make a path distinction meaningful?")),
    SourceCard("F005", "The Theory of Positrons", 1949, "historical_simulation", "antimatter", "Prefer a compact space-time account when it unifies apparently different processes.", "The formalism relates electron and positron processes within a unified relativistic description, but a diagram is a calculational representation rather than a literal movie.", ("Is a positron literally an electron traveling backward through everyday time?", "What does the backward-time line in a diagram mean operationally?", "Why is a vivid interpretation not enough without quantitative predictions?", "Explain how one formalism can organize particle and antiparticle processes.")),
    SourceCard("F008", "Atomic Theory of the Two-Fluid Model of Liquid Helium", 1954, "historical_simulation", "superfluidity", "Connect macroscopic behavior to constrained microscopic motion and testable excitations.", "Superfluid behavior requires a quantum many-body account; the two-fluid description separates effective components without implying two chemical liquids.", ("Are there literally two different liquids inside superfluid helium?", "How can microscopic quantum rules affect a visible flow experiment?", "What evidence would distinguish a useful two-fluid model from a verbal picture?", "Why does low temperature change which excitations matter?")),
    SourceCard("F009", "Slow Electrons in a Polar Crystal", 1955, "historical_simulation", "polarons", "Treat a particle together with the response it induces in its environment.", "An electron in a polar crystal interacts with lattice polarization, so its effective motion differs from that of an isolated electron.", ("Why can an electron in a crystal behave as if it has a different mass?", "Is a polaron a new elementary particle?", "Explain why ignoring the lattice response can give the wrong motion.", "How would you test a model of an electron dressed by polarization?")),
    SourceCard("F010", "Theory of the Fermi Interaction", 1958, "historical_simulation", "weak interactions", "Let symmetry and experimental regularities constrain the mathematical form of an interaction.", "The V-A structure organizes weak-interaction phenomena and encodes parity violation; agreement must be checked across distinct decays.", ("Why did parity violation force physicists to revise the weak-interaction model?", "What would make a compact weak-interaction formula more than elegant notation?", "Explain why one successful decay calculation is not enough validation.", "How can symmetry restrict possible interaction terms without proving which one nature uses?")),
    SourceCard("F011", "There's Plenty of Room at the Bottom", 1959, "historical_simulation", "nanotechnology", "Analyze small-scale engineering through physical limits and concrete mechanisms.", "Miniaturization is constrained by heat, fabrication accuracy, noise, and quantum effects rather than by scale alone.", ("Is making a machine smaller just a matter of using smaller tools?", "Estimate what becomes difficult when information is stored in very small structures.", "Which physical limits should be checked before claiming unlimited miniaturization?", "How could one manipulate matter at small scales without relying on a vague vision?")),
    SourceCard("F013", "The Character of Physical Law: Gravitation", 1964, "historical_simulation", "gravitation", "Separate the descriptive reach of a law from an unsupported story about why it exists.", "A law can predict diverse observations precisely while deeper explanations of its form remain open.", ("Does writing Newton's law explain why gravity exists?", "Why is the inverse-square form connected with spreading through three-dimensional space?", "What is gained by finding one law behind falling objects and planetary motion?", "How should we talk about a law that predicts well but lacks a deeper mechanism?")),
    SourceCard("F014", "The Character of Physical Law: Mathematics and Physics", 1964, "historical_simulation", "mathematics", "Use mathematics as a precise language for consequences while keeping contact with experiment.", "Mathematical consistency is necessary for a physical theory but empirical agreement decides whether it describes nature.", ("If a theory is mathematically beautiful, why is that not sufficient evidence?", "Why do the same equations appear in apparently different physical systems?", "A student can manipulate symbols but cannot state a prediction. What should they do next?", "Explain the partnership and the boundary between mathematics and physics.")),
    SourceCard("F015", "The Character of Physical Law: Conservation", 1964, "historical_simulation", "conservation laws", "Describe conservation operationally, including hidden transfers and the accounting boundary.", "A conserved quantity may change form or move across a boundary while the total for a closed system remains fixed.", ("Energy seems to disappear as a sliding block stops. Where should we look?", "Why is conservation more than saying that the total stays the same?", "How does choosing the system boundary affect an energy balance?", "Explain how an unseen form of energy can be inferred without inventing it freely.")),
    SourceCard("F016", "The Character of Physical Law: Symmetry", 1964, "historical_simulation", "symmetry", "Ask which transformations leave experimental predictions unchanged and which do not.", "A physical symmetry is an invariance of observable laws under a specified transformation, not merely visual balance.", ("What does it mean for a law, rather than an object, to be symmetric?", "How would you test whether mirror reflection is a symmetry of an interaction?", "Why is a nearly symmetric pattern not the same as an exact physical symmetry?", "Explain how symmetry can reduce the number of independent possibilities in a theory.")),
    SourceCard("F017", "The Character of Physical Law: Past and Future", 1964, "historical_simulation", "time asymmetry", "Distinguish time-symmetric microscopic laws from statistical boundary conditions and irreversible appearances.", "Macroscopic irreversibility can arise statistically even when many microscopic equations are reversible.", ("If microscopic laws can run backward, why does a broken cup not reassemble?", "What role does an initially ordered state play in the arrow of time?", "Is friction evidence that fundamental laws know the direction of time?", "Explain irreversibility without saying that reverse motion is logically impossible.")),
    SourceCard("F018", "The Character of Physical Law: Probability and Uncertainty", 1964, "historical_simulation", "probability", "Treat uncertainty as quantitatively testable structure, not permission for arbitrary claims.", "Quantum theory predicts probability distributions that can be tested over repeated preparations even when individual outcomes are not fixed.", ("If an event is probabilistic, does that mean the theory predicts nothing?", "How can a random individual result support a precise physical law?", "What experiment distinguishes a claimed probability from guesswork?", "Explain why uncertainty does not make every outcome equally plausible.")),
    SourceCard("F019", "The Character of Physical Law: Seeking New Laws", 1964, "historical_simulation", "scientific method", "Guess consequences, compare with experiment, and reject a proposal when its consequences disagree.", "A proposed law must yield discriminating consequences; authority and elegance cannot rescue empirical failure.", ("How do you test two theories that explain all observations measured so far?", "A famous physicist proposes an elegant law that misses the data. What follows?", "Why must a scientific guess risk being wrong?", "What makes a new measurement capable of choosing between competing explanations?")),
    SourceCard("F020", "Simulating Physics with Computers", 1982, "historical_simulation", "quantum simulation", "Match the simulator's information-processing resources to the structure of the physical system.", "Generic quantum evolution can require resources that grow prohibitively on a classical description, motivating quantum simulators.", ("Why might a classical computer struggle to simulate an arbitrary quantum system?", "Does a quantum simulator reproduce every microscopic detail of its target?", "What evidence would show that a simulator captures the intended physics?", "Explain why computational cost, not just logical possibility, matters in simulation.")),
    SourceCard("F023", "Personal Observations on the Reliability of the Shuttle", 1986, "historical_simulation", "engineering risk", "Compare management claims with component evidence and keep numerical uncertainty visible.", "Reliability estimates must be tied to failure mechanisms and data; large gaps between official probabilities and engineering evidence are warning signs.", ("Management quotes a one-in-one-hundred-thousand failure risk without a derivation. How should an engineer respond?", "Why can repeated past success fail to prove that a stressed component is safe?", "How should uncertainty be reported when component data are sparse?", "Explain why a system-level reliability number must be traceable to physical failure modes.")),
    SourceCard("F024", "Cargo Cult Science", 1974, "historical_simulation", "scientific integrity", "Report evidence that could prove you wrong and avoid fooling yourself.", "Scientific integrity includes exposing contrary evidence, alternative explanations, and methodological weaknesses.", ("A study reports only the trials that worked. What is scientifically missing?", "Why is copying the outward form of an experiment not enough?", "A researcher finds a calibration error that weakens the headline result. What should be reported?", "How can a method look rigorous yet fail to test its central claim?")),
    SourceCard("F025", "The Pleasure of Finding Things Out", 1981, "historical_simulation", "learning", "Prefer curiosity, observation, and usable understanding over prestige or memorized labels.", "Understanding is demonstrated by explaining behavior and making predictions, not merely recalling terminology.", ("A student memorizes every term but cannot predict what will happen. Diagnose the gap.", "Why can admitting 'I don't know' be productive in science?", "How would you turn curiosity about an everyday effect into an investigation?", "Explain the difference between recognizing an expert phrase and understanding a mechanism.")),
    SourceCard("M005", "Programmable Dual-RNA-Guided DNA Endonuclease", 2012, "modern_feynman_inspired", "CRISPR", "Separate demonstrated molecular mechanism from broader claims and ask what controls distinguish them.", "Cas9 can be programmed by guide RNA to cut complementary DNA near an appropriate motif; delivery, off-target effects, and repair outcomes are separate problems.", ("Explain CRISPR-Cas9 without calling it magic molecular scissors.", "What experiment shows that guide sequence, rather than chance damage, directs cutting?", "Why does successful cutting not guarantee a desired genetic edit?", "Which claims about CRISPR follow from the biochemical mechanism, and which require additional evidence?")),
    SourceCard("M006", "Algorithms for Quantum Computation", 1994, "modern_feynman_inspired", "quantum algorithms", "Explain the resource advantage for a specified task without turning it into a claim of universal speedup.", "Shor's algorithm gives polynomial-time quantum algorithms for factoring and discrete logarithms, conditional on a sufficiently capable fault-tolerant quantum computer.", ("Does Shor's algorithm make every computation exponentially faster?", "What exactly is the demonstrated algorithmic advantage in quantum factoring?", "Why must logical error correction be separated from the abstract circuit complexity?", "Explain the difference between a quantum algorithm and a present-day machine that can run it at useful scale.")),
    SourceCard("M008", "Loophole-Free Bell Test", 2015, "modern_feynman_inspired", "Bell tests", "State what an experiment rules out under explicit assumptions and avoid faster-than-light storytelling.", "Loophole-free Bell tests violate a Bell inequality while closing major detection and locality loopholes, constraining local hidden-variable accounts.", ("What did a loophole-free Bell test rule out, and what did it not prove?", "Why is entanglement correlation not a usable faster-than-light signal?", "What are the detection and locality loopholes in a Bell experiment?", "How can a statistical inequality test a broad class of hidden-variable models?")),
    SourceCard("M009", "Quantum Supremacy Using a Programmable Processor", 2019, "modern_feynman_inspired", "quantum computing", "Tie advantage claims to the exact benchmark, verification method, and classical baseline.", "A programmable superconducting processor sampled from a designed random-circuit distribution faster than then-used classical methods; this was a narrow benchmark, not general practical superiority.", ("Did the 2019 random-circuit experiment prove quantum computers are better at everything?", "How can researchers check a sampling task that becomes hard to reproduce classically at full scale?", "Why can an improved classical algorithm change the size of a claimed quantum advantage?", "What would turn a benchmark advantage into evidence of practical computational value?")),
    SourceCard("M010", "Suppressing Quantum Errors by Scaling a Surface Code Logical Qubit", 2023, "modern_feynman_inspired", "error correction", "Focus on the scaling relation between physical resources and logical error rather than a single best run.", "Surface-code experiments test whether increasing code distance suppresses logical error under the achieved physical error rates; this is progress toward, not completion of, fault tolerance.", ("What does it mean for a larger surface code to reduce logical error?", "Why is one long-lived logical qubit not yet a fault-tolerant computer?", "Which scaling evidence matters more than quoting the lowest observed error rate?", "Explain the threshold idea without implying that errors vanish.")),
)


AUDIENCES = ("curious high-school student", "first-year university student", "working engineer", "general scientific reader")
CONTEXTS = (
    "The learner is relying on a memorable slogan instead of a mechanism.",
    "The learner wants an operational explanation tied to a possible measurement.",
    "The learner has a numerical or mathematical result but has not checked its assumptions.",
    "The learner asks how strongly the available evidence supports the conclusion.",
)


def build(card: SourceCard, variant: int) -> Scenario:
    modern = card.mode == "modern_feynman_inspired"
    scenario_date = "2026-01-01" if modern else f"{card.year}-12-31"
    source_date = f"{card.year}-01-01"
    forbidden = ["invented quotation", "unsupported authority claim", "long verbatim source reproduction"]
    if modern:
        forbidden.append("attribute post-1988 knowledge to historical Feynman")
    else:
        forbidden.append(f"claims knowledge later than {card.year}")
    return Scenario.model_validate(
        {
            "task_id": f"f26_{card.source_id.lower()}_{variant + 1:02d}",
            "mode": card.mode,
            "date": scenario_date,
            "audience": AUDIENCES[variant],
            "scenario": f"{CONTEXTS[variant]} Topic: {card.topic}.",
            "user_message": card.questions[variant],
            "persona_card": {
                "identity": "anonymous_persona_x",
                "documented_principles": [
                    card.principle,
                    "Expose assumptions and connect claims to observations or discriminating tests.",
                    "Use direct, concrete language and mark uncertainty honestly.",
                ],
                "knowledge_cutoff": scenario_date,
            },
            "evidence": [
                {
                    "source_id": card.source_id,
                    "type": "paraphrased_source_card",
                    "content": card.fact,
                    "allowed_for_answer": True,
                    "source_date": source_date,
                    "title": card.title,
                    "location": "catalog-level pilot card; verify against primary text before production",
                }
            ],
            "forbidden_shortcuts": forbidden,
            "evaluation_tags": [
                "scientific_correctness",
                "source_grounding",
                "reasoning_fidelity",
                "epistemic_honesty",
                "non_recitation",
                "temporal_honesty",
            ],
            "reference_facts": [card.fact],
            "source_passages": [],
        }
    )


def generate(destination: Path) -> int:
    scenarios = [build(card, variant) for card in CARDS for variant in range(4)]
    assert len(CARDS) == 25 and len(scenarios) == 100
    assert len({row.task_id for row in scenarios}) == 100
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for scenario in scenarios:
            handle.write(scenario.model_dump_json() + "\n")
    return len(scenarios)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path, nargs="?", default=Path("recipe/feynman/scenarios.seed100.jsonl"))
    args = parser.parse_args()
    count = generate(args.destination)
    print(f"Wrote {count} validated scenarios to {args.destination}")


if __name__ == "__main__":
    main()
