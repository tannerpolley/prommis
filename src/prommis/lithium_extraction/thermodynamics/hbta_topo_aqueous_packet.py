"""Aqueous chloride parameter sources for extraction calculations."""

from prommis.lithium_extraction.paths import REPO_ROOT

AQUEOUS_PACKET = "figiel-2025-chloride-mixture-refit/4"
SHARED_AQUEOUS_PACKET = "lithium-extraction-aqueous-chlorides-option-1-primary/2"


def shared_packet_directory():
    return REPO_ROOT / "data/reference/aqueous_chloride_packets" / SHARED_AQUEOUS_PACKET


def shared_parameter_path():
    return shared_packet_directory() / "parameters/parameters.json"
