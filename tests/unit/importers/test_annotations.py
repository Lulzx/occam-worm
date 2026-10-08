"""OW-013 annotation build on tiny synthetic tarballs: aliases, conflicts with provenance, sign, label audit."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from occamworm.importers import __main__ as cli
from occamworm.importers import annotations as A
from occamworm.importers import openworm as ow
from occamworm.importers import wormneuroatlas as wa
from occamworm.importers._archive import ArchiveError, read_members
from occamworm.importers._xlsx import read_workbook
from occamworm.importers.aliases import (
    MAPPING_VERSION,
    AliasRow,
    AliasTable,
    derive_aliases,
    non_neuron_reason,
    suggestions,
)
from occamworm.importers.annotation_edges import Recon, sign_for

CANONICAL = ["AVAL", "AVAR", "AIBL", "AIBR", "RIML", "RIMR", "DB1", "DB2", "DB3", "VA1", "AWCL", "AWCR"]

CELLS_PY = b"""
PREFERRED_HERM_NEURON_NAMES = [
    "AVAL", "AVAR", "AIBL", "AIBR", "RIML", "RIMR", "DB1", "DB2", "DB3", "VA1", "AWCL", "AWCR",
]
SENSORY_NEURONS_COOK = ["AWCL", "AWCR"]
INTERNEURONS_COOK = ["AVAL", "AVAR", "AIBL", "AIBR", "RIML", "RIMR"]
MOTORNEURONS_COOK = ["DB1", "DB2", "DB3", "VA1"]
PHARYNGEAL_POLYMODAL_NEURONS = []
UNKNOWN_FUNCTION_NEURONS = []
PHARYNGEAL_NEURONS = []
UNSPECIFIED = "BWM"
PREFERRED_MUSCLE_NAMES = ["MDL01"] + [UNSPECIFIED]
raise RuntimeError("upstream code must never be executed")
"""

CELL_INFO = (
    b"Cell name,Type,Name details,Lineage,Classification\n"
    b"AVAL,Layer 3 interneuron,x,y,z\nAVAR,Layer 3 interneuron,x,y,z\n"
    b"AIBL,Layer 2 interneuron,x,y,z\nAIBR,Layer 2 interneuron,x,y,z\n"
    b"RIML,Layer 2 interneuron,x,y,z\nRIMR,Layer 2 interneuron,x,y,z\n"
    b"DB1,Ventral cord motor neuron,x,y,z\nDB2,Ventral cord motor neuron,x,y,z\n"
    b"DB3,Ventral cord motor neuron,x,y,z\nVA1,Ventral cord motor neuron,x,y,z\n"
    b"AWCL,Amphid,x,y,z\nAWCR,Amphid,x,y,z\n"
)

BENTLEY_MA = b"RIML,AVAL,tyramine,lgc-55\nRIMR,AVAL,tyramine,lgc-55\n"
BENTLEY_NP = b"AVAL,AVAR,flp-1,npr-11\nAVAL,AVAR,flp-2,npr-4\nAVAR,AVAL,flp-1,npr-11\n"
RIPOLL_CSV = b"Row,AVAL,AVAR,VA01\nAVAL,0,2,0\nAVAR,1,0,3\nVA01,0,0,0\n"

LINEAGE = (
    b"#Cell\tLineage Name\tDescription\nAVAL\tAB a\tinterneuron\n"
    b"AMsoL\tH2\tAmphid socket\nAMsoR\tH2\tAmphid socket\nGLRL\tM\tGLR cell\n"
)
GANGLIA = json.dumps(
    {
        "head": ["anterior ganglion"],
        "pharynx": ["pharyngeal bulb"],
        "anterior ganglion": ["AVAL", "AVAR", "AWCL", "AWCR"],
        "pharyngeal bulb": [],
        "ventral nerve cord": ["DB1", "DB2", "DB3", "VA1"],
    }
).encode()
POSITIONS = b"#AVAL AVAR AVAL RIML\n1.0 2.0 3.0\n4.0 5.0 6.0\n1.5 2.5 3.5\n7.0 8.0 9.0\n"
NEURON_IDS = b"000\tAVAL\n001\tAVAR\n002\tAWCON\n"
WNA_TOY_CSV = (
    b"pre\tpost\ttype\tsynapses\nAVAL\tAVAR\tchemical\t3\nAVAR\tAVAL\tchemical\t5\nAVAL\tRIML\tchemical\t1\n"
    b"DB2\tVA1\tchemical\t4\nAVAL\tRIMR\tchemical\t1\nAVAL\tAVAR\telectrical\t1\n"
)


def _col(i: int) -> str:
    return chr(ord("A") + i)


def make_xlsx(sheets: dict[str, list[list[str | None]]]) -> bytes:
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        sheet_tags = "".join(
            f'<sheet name="{escape(n)}" sheetId="{i}" r:id="rId{i}"/>' for i, n in enumerate(sheets, start=1)
        )
        z.writestr(
            "xl/workbook.xml", f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets>{sheet_tags}</sheets></workbook>'
        )
        rel_tags = "".join(
            f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1)
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rel_tags}</Relationships>',
        )
        for i, rows in enumerate(sheets.values(), start=1):
            body = ""
            for r, row in enumerate(rows, start=1):
                cells = "".join(
                    f'<c r="{_col(c)}{r}" t="inlineStr"><is><t>{escape(v)}</t></is></c>'
                    for c, v in enumerate(row)
                    if v is not None
                )
                body += f'<row r="{r}">{cells}</row>'
            z.writestr(
                f"xl/worksheets/sheet{i}.xml", f'<worksheet xmlns="{main}"><sheetData>{body}</sheetData></worksheet>'
            )
    return buf.getvalue()


WANG = make_xlsx(
    {
        "Supp File 2": [
            [],
            [None, "title"],
            [],
            [None, "Class", "Neuron", "Lineage", "Neurotransmitter(s)", "Comments"],
            [None, "AVA", "AVAL", "l", "ACh"],
            [None, None, "AVAR", "l", "GABA"],
            [None, "AIB", "AIBL", "l", "Glu"],
            [None, None, "AIBR", "l", "Glu"],
            [None, "RIM", "RIML", "l", "Glu"],
            [None, None, "RIMR", "l", "Glu"],
            [None, "DB", "DB1/3", "l", "ACh", "pair label"],
            [None, None, "DB2", "l", "ACh"],
            [None, None, "DB3/1", "l", "*ACh - NEW"],
            [None, "VA", "VA1", "l", "ACh"],
            [None, "AWC", "AWCL", "l", "Glu"],
            [None, None, "AWCR", "l", "Glu"],
            [None, "note row"],
        ]
    }
)
RIPOLL_NEURONS = make_xlsx(
    {
        "Neuron information": [
            ["Neuron", "ID", "Type", "Segment"],
            [None, None, "sub", None],
            ["AVAL", "1", "interneuron", "Head"],
            ["AVAR", "2", "interneuron", "Head"],
            ["AIBL", "3", "interneuron", "Head"],
            ["DB1", "4", "motor neuron", "Midbody"],
            ["AWCL", "5", "sensory neuron", "Head"],
        ]
    }
)


def _fenyves_edge(pre: str, post: str, w: int, polarity: str) -> list[str | None]:
    return [pre, "1mary", "2ndary", post, str(w), "chemical", *["0"] * 10, polarity, None, "x", "FALSE", "0"]


FENYVES = make_xlsx(
    {
        "1. NT expr": [
            ["Class member", "Dominant NT", "Alternative NT", "Neurotransmitters"],
            ["AVAL", "ACh", None, "ACh"],
            ["AVAR", "ACh", "GABA", "ACh"],
            ["DB2", "GABA", None, "GABA"],
            ["VA01", "ACh", None, "ACh"],
        ],
        "2. Receptor gene table": [["Glu Pos"]],
        "3. Receptor Gene expression ": [[], ["AVAL", "glr-1", "acr-2"], ["AVAR"]],
        "4. Receptor type expression ": [
            ["h"],
            ["h2"],
            ["AVAL", "0", "0", "0", "0", "0", "0", "Interneuron"],
            ["RIML", "0", "0", "0", "0", "0", "0", "Interneuron, motor neuron"],
            ["AIBL", "0", "0", "0", "0", "0", "0", "Motor neuron"],
            ["VA01", "0", "0", "0", "0", "0", "0", "Motor neuron"],
        ],
        "5. Sign prediction": [
            ["Source"],
            ["Neuron"],
            _fenyves_edge("AVAL", "AVAR", 3, "+"),
            _fenyves_edge("AVAR", "AVAL", 2, "-"),
            _fenyves_edge("AVAL", "RIML", 1, "complex"),
            _fenyves_edge("DB2", "VA01", 4, "no pred"),
        ],
    }
)

NODES = ["AVAL", "AVAR", "RIML", "DB2", "VA1", "MDL01", "GLRL"]


def _toy_cache() -> bytes:
    idx = {n: i for i, n in enumerate(NODES)}
    cs = np.zeros((len(NODES), len(NODES)))
    gj = np.zeros_like(cs)
    for a, b, w in [
        ("AVAL", "AVAR", 3),
        ("AVAR", "AVAL", 2),
        ("AVAL", "RIML", 1),
        ("DB2", "VA1", 4),
        ("DB2", "MDL01", 5),
    ]:
        cs[idx[a], idx[b]] = w
    cs[idx["AVAL"], idx["GLRL"]] = 2
    for a, b, w in [
        ("AVAL", "AVAR", 1),
        ("AVAR", "AVAL", 1),
        ("RIML", "DB2", 2),
        ("AVAL", "MDL01", 1),
        ("MDL01", "AVAL", 1),
    ]:
        gj[idx[a], idx[b]] = w
    return json.dumps(
        {"nodes": NODES, "connections": {"Generic_CS": cs.tolist(), "Generic_GJ": gj.tolist()}, "summary": ""}
    ).encode()


RECONS = (
    Recon("toy-a", "toolbox_cache", ow.SOURCE, ow.cache_member("ToyA"), "Toy A", "adult", "count", "ToyA", "toy-a:wna"),
    Recon(
        "toy-a:wna",
        "wna_csv",
        wa.SOURCE,
        "wormneuroatlas/data/toy_a.csv",
        "Toy A (wna)",
        "adult",
        "count",
        None,
        "toy-a",
    ),
    Recon("toy-fenyves", "fenyves", wa.SOURCE, wa.FENYVES, "Fenyves toy", None, "count"),
    Recon("toy-ma", "bentley_ma", ow.SOURCE, ow.BENTLEY_MA, "Bentley MA toy", None, "pairs"),
    Recon("toy-np", "bentley_np", ow.SOURCE, ow.BENTLEY_NP, "Bentley NP toy", None, "pairs"),
    Recon("toy-ripoll", "ripoll", ow.SOURCE, "cect/data/toy_ripoll.csv", "Ripoll toy", "adult", "pathways"),
)


def _tar(path: Path, top: str, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for rel, data in files.items():
            info = tarfile.TarInfo(f"{top}/{rel}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


ATLAS_ROWS = [
    ("r1", "AVAL", "name", False),
    ("r2", "AVAL", "name", False),
    ("r1", "AVAR", "name", True),
    ("r1", "AVAR", "name", True),
    ("r1", "rimr", "name", False),
    ("r1", "AWC", "name", False),
    ("r2", "AWCON", "name", False),
    ("r1", "AMsoL", "name", False),
    ("r1", "AIBLL", "name", False),
    ("r2", "ZZZ9", "name", False),
    ("r1", "", "blank", False),
    ("r1", "41", "numeric", False),
    ("r1", "merge", "other", False),
]


def make_inputs(tmp_path: Path) -> tuple[Path, Path]:
    raw, atlas = tmp_path / "raw", tmp_path / "atlas"
    _tar(
        raw / ow.SOURCE / "toolbox.tar.gz",
        "ConnectomeToolbox-abc",
        {
            ow.CELLS_PY: CELLS_PY,
            ow.CELL_INFO: CELL_INFO,
            ow.WANG_NT: WANG,
            ow.RIPOLL_NEURONS: RIPOLL_NEURONS,
            ow.BENTLEY_MA: BENTLEY_MA,
            ow.BENTLEY_NP: BENTLEY_NP,
            ow.cache_member("ToyA"): _toy_cache(),
            "cect/data/toy_ripoll.csv": RIPOLL_CSV,
        },
    )
    _tar(
        raw / wa.SOURCE / "wna.tar.gz",
        "wormneuroatlas-abc",
        {
            wa.NEURON_IDS: NEURON_IDS,
            wa.GANGLIA: GANGLIA,
            wa.POSITIONS: POSITIONS,
            wa.LINEAGE: LINEAGE,
            wa.FENYVES: FENYVES,
            wa.BENTLEY_MA: b"#source neuron, target neuron, transmitter, receptor,\n"
            + b"RIML,AVAL,tyramine,lgc-55,seq\nRIMR,AVAL,tyramine,lgc-55,seq\n",
            wa.BENTLEY_NP: b"#source neuron, target neuron, transmitter, receptor,\n"
            + b"AVAL,AVAR,flp-1,npr-11,seq\nAVAL,AVAR,flp-2,npr-4,seq\nAVAR,AVAL,flp-1,npr-11,seq\n",
            "wormneuroatlas/data/toy_a.csv": WNA_TOY_CSV,
        },
    )
    atlas.mkdir()
    cols = list(zip(*ATLAS_ROWS, strict=True))
    pq.write_table(
        pa.table(
            {
                "recording_id": pa.array(cols[0], pa.string()),
                "label": pa.array(cols[1], pa.string()),
                "label_status": pa.array(cols[2], pa.string()),
                "label_duplicated": pa.array(cols[3], pa.bool_()),
            }
        ),
        atlas / "neurons.parquet",
    )
    return raw, atlas


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    tmp = tmp_path_factory.mktemp("annotations")
    raw, atlas = make_inputs(tmp)
    out = tmp / "out"
    manifest = A.run_build(raw, out, atlas_dir=atlas, inputs=[{"source": "synthetic"}], recons=RECONS)
    return out, manifest


def rows(out: Path, name: str) -> list[dict[str, Any]]:
    return pq.read_table(out / f"{name}.parquet").to_pylist()


# --------------------------------------------------------------------------- alias mapping


def test_alias_rules_are_explicit_and_typos_are_not_aliases() -> None:
    observed = {
        "VA01": {"fenyves"},
        "rimr": {"atlas"},
        "AVA": {"atlas"},
        "AWCON": {"wna"},
        "DB1/3": {"wang"},
        "AIBLL": {"atlas"},
        "AVAL": {"atlas"},
    }
    table = AliasTable(CANONICAL, derive_aliases(CANONICAL, observed))
    by = {r.alias: r for r in table.rows}
    assert by["VA01"].kind == "zero_padded" and by["VA01"].canonical_neuron_id == "VA1" and not by["VA01"].ambiguous
    assert by["rimr"].kind == "case_variant" and by["rimr"].canonical_neuron_id == "RIMR"
    assert "AIBLL" not in by and "AVAL" not in by
    assert table.resolve("VA01").status == "alias" and table.resolve("VA01").neuron_id == "VA1"
    assert table.resolve("AVAL").status == "exact"
    assert table.resolve("AIBLL").status == "unresolved"
    ava = table.resolve("AVA")
    assert (ava.status, ava.candidates, ava.kind) == ("ambiguous", ("AVAL", "AVAR"), "class_label")
    assert ava.neuron_id is None and "left/right" in ava.reason
    # AWC ON/OFF and the Wang pair label are never resolved to one cell
    assert table.resolve("AWCON").status == "ambiguous" and table.resolve("AWCON").candidates == ("AWCL", "AWCR")
    assert table.resolve("DB1/3").candidates == ("DB1", "DB3")
    assert all(r.alias not in CANONICAL for r in table.rows)


def test_alias_table_rejects_inconsistent_rows() -> None:
    with pytest.raises(ValueError, match="itself a canonical"):
        AliasTable(CANONICAL, [AliasRow("AVAL", "AVAR", "x", False, 1, "", ())])
    with pytest.raises(ValueError, match="unknown neuron"):
        AliasTable(CANONICAL, [AliasRow("foo", "NOPE", "x", False, 1, "", ())])


def test_suggestions_and_non_neuron_reasons() -> None:
    assert suggestions("AIBLL", CANONICAL) == ["AIBL"]
    assert suggestions("AIB", CANONICAL) == ["AIBL", "AIBR"]
    assert suggestions("AV", CANONICAL) == []
    cells = {"AMsoL": "Amphid socket", "AMsoR": "Amphid socket"}
    assert non_neuron_reason("AMsoL", cells) == "known non-neuron cell: Amphid socket"
    assert "case differs" in (non_neuron_reason("AMSOL", cells) or "")
    assert "class-level" in (non_neuron_reason("AMso", cells) or "")
    assert non_neuron_reason("ZZZ9", cells) is None


def test_aliases_table_has_version_and_reason(built: tuple[Path, dict[str, Any]]) -> None:
    out, manifest = built
    aliases = rows(out, "aliases")
    assert {r["mapping_version"] for r in aliases} == {MAPPING_VERSION} == {manifest["mapping_version"]}
    va = next(r for r in aliases if r["alias"] == "VA01")
    assert va["canonical_neuron_id"] == "VA1" and va["reason"] and "fenyves2020-s3" in va["observed_in"]
    db = [r for r in aliases if r["alias"] == "DB1/3"]
    assert {r["canonical_neuron_id"] for r in db} == {"DB1", "DB3"} and all(r["ambiguous"] for r in db)
    cls = [r for r in aliases if r["alias"] == "AVA"]
    assert {r["canonical_neuron_id"] for r in cls} == {"AVAL", "AVAR"} and "class:wang2024" in cls[0]["observed_in"]


# --------------------------------------------------------------------------- conflicts keep provenance


def test_conflicting_transmitters_are_kept_with_sources(built: tuple[Path, dict[str, Any]]) -> None:
    out, _ = built
    conflicts = {(r["entity_id"], r["attribute"]): r for r in rows(out, "conflicts") if r["entity_type"] == "neuron"}
    hard = conflicts[("DB2", "transmitter")]
    assert hard["conflict_kind"] == "classical_transmitter_differs" and hard["severity"] == "hard"
    assert hard["values"] == ["ACh", "GABA"]
    assert hard["sources"] == ["ACh@wang2024-nt-atlas", "GABA@fenyves2020-s3"]
    soft = conflicts[("AVAR", "transmitter")]
    assert soft["conflict_kind"] == "dominant_matches_alternative_only" and soft["severity"] == "soft"
    mono = conflicts[("RIML", "transmitter")]
    assert mono["conflict_kind"] == "monoamine_reported_by_one_source_only"
    assert "tyramine@bentley2016-monoamine" in mono["sources"]

    neurons = {r["neuron_id"]: r for r in rows(out, "neurons")}
    db2 = neurons["DB2"]
    assert db2["transmitter_values"] == ["ACh", "GABA"] and db2["transmitter_conflict"] is True
    assert db2["transmitter_sources"] == ["ACh@wang2024-nt-atlas", "GABA@fenyves2020-s3"]
    assert neurons["AVAL"]["transmitter_conflict"] is False and neurons["AVAL"]["transmitter_values"] == ["ACh"]

    evidence = [
        r for r in rows(out, "neuron_attributes") if r["neuron_id"] == "DB2" and r["attribute"] == "transmitter"
    ]
    assert {(r["value"], r["source"], r["role"]) for r in evidence} == {
        ("ACh", "wang2024-nt-atlas", "reported"),
        ("GABA", "fenyves2020-s3", "dominant"),
    }
    assert all(r["source_member"] and r["raw_label"] for r in evidence)


def test_type_conflicts_distinguish_disjoint_from_overlap(built: tuple[Path, dict[str, Any]]) -> None:
    out, _ = built
    conflicts = {(r["entity_id"], r["attribute"]): r for r in rows(out, "conflicts") if r["entity_type"] == "neuron"}
    assert conflicts[("AIBL", "neuron_type")]["conflict_kind"] == "type_sets_disjoint"  # interneuron vs motor
    assert conflicts[("AIBL", "neuron_type")]["severity"] == "hard"
    assert conflicts[("RIML", "neuron_type")]["conflict_kind"] == "type_sets_overlap_but_differ"
    assert ("AVAL", "neuron_type") not in conflicts
    neurons = {r["neuron_id"]: r for r in rows(out, "neurons")}
    assert neurons["AIBL"]["neuron_type_values"] == ["interneuron", "motor"]
    assert neurons["AIBL"]["neuron_type_conflict"] is True


def test_position_with_two_values_in_one_source_is_flagged_not_chosen(built: tuple[Path, dict[str, Any]]) -> None:
    out, _ = built
    conflict = next(r for r in rows(out, "conflicts") if r["entity_id"] == "AVAL" and r["attribute"] == "position_xyz")
    assert conflict["conflict_kind"] == "several_values_in_one_source" and conflict["severity"] == "soft"
    neurons = {r["neuron_id"]: r for r in rows(out, "neurons")}
    assert neurons["AVAL"]["position_x"] is None and neurons["RIML"]["position_x"] == 7.0


def test_edge_weight_disagreement_between_bundles(built: tuple[Path, dict[str, Any]]) -> None:
    out, _ = built
    edge = {r["entity_id"]: r for r in rows(out, "conflicts") if r["entity_type"] == "edge"}
    differs = edge["AVAR>AVAL|chem"]
    assert differs["conflict_kind"] == "synapse_count_differs"
    assert differs["sources"] == ["2@toy-a", "5@toy-a:wna"]
    only = edge["AVAL>RIMR|chem"]
    assert only["conflict_kind"] == "edge_in_one_bundle_only"
    assert only["sources"] == ["1@toy-a:wna", "absent@toy-a"]
    assert "AVAL>AVAR|chem" not in edge  # equal weights are not conflicts


# --------------------------------------------------------------------------- edges and sign


def test_edges_schema_and_kinds(built: tuple[Path, dict[str, Any]]) -> None:
    out, _ = built
    table = pq.read_table(out / "edges.parquet")
    assert table.column_names == [
        "source_neuron_id",
        "target_neuron_id",
        "edge_kind",
        "synapse_count",
        "sign",
        "confidence",
        "source_reconstruction",
        "life_stage",
        "provenance_reference",
    ]
    assert table.schema.field("synapse_count").type == pa.float32()
    assert table.schema.field("confidence").type == pa.float32()
    edges = table.to_pylist()
    assert {e["edge_kind"] for e in edges} == {"chem", "gap", "putative_mod", "nmj"}
    assert {e["sign"] for e in edges} <= {"excitatory", "inhibitory", "unknown", None}
    assert all(e["provenance_reference"] and e["source_reconstruction"] for e in edges)
    details = rows(out, "edge_details")
    assert [d["edge_row"] for d in details] == list(range(len(edges)))


def test_sign_is_unknown_unless_the_dataset_states_it(built: tuple[Path, dict[str, Any]]) -> None:
    out, _ = built
    edges = rows(out, "edges")
    for e in edges:
        if e["source_reconstruction"] == "toy-fenyves":
            continue
        # DB2 is GABAergic in one source, AVAL cholinergic in another: no edge inherits a sign from that
        assert e["sign"] == (None if e["edge_kind"] == "gap" else "unknown"), e
    fenyves = {
        (e["source_neuron_id"], e["target_neuron_id"]): e["sign"]
        for e in edges
        if e["source_reconstruction"] == "toy-fenyves"
    }
    assert fenyves == {
        ("AVAL", "AVAR"): "excitatory",
        ("AVAR", "AVAL"): "inhibitory",
        ("AVAL", "RIML"): "unknown",  # 'complex'
        ("DB2", "VA1"): "unknown",  # 'no pred'; VA01 mapped through the alias table
    }
    details = {(d["edge_row"]): d for d in rows(out, "edge_details")}
    signed = [i for i, e in enumerate(edges) if e["source_reconstruction"] == "toy-fenyves"]
    assert {details[i]["sign_basis"] for i in signed} == {"fenyves2020_nt_r_prediction"}
    assert {details[i]["sign_raw"] for i in signed} == {"+", "-", "complex", "no pred"}


def test_sign_for_unit() -> None:
    fen = Recon("f", "fenyves", wa.SOURCE, "m", "t", None, "w")
    cache = Recon("c", "toolbox_cache", ow.SOURCE, "m", "t", None, "w")
    assert sign_for(fen, "chem", {"+"}) == ("excitatory", "fenyves2020_nt_r_prediction")
    assert sign_for(fen, "chem", {"+", "-"})[0] == "unknown"
    assert sign_for(fen, "chem", {"complex"})[0] == "unknown"
    assert sign_for(cache, "chem", set())[0] == "unknown"
    assert sign_for(cache, "putative_mod", set())[0] == "unknown"
    assert sign_for(cache, "nmj", set())[0] == "unknown"
    assert sign_for(cache, "gap", set()) == (None, "not_applicable")


def test_reconstructions_stay_separate_and_gaps_are_symmetric(built: tuple[Path, dict[str, Any]]) -> None:
    out, manifest = built
    edges = rows(out, "edges")
    chem_ab = {
        e["source_reconstruction"]: e["synapse_count"]
        for e in edges
        if (e["source_neuron_id"], e["target_neuron_id"], e["edge_kind"]) == ("AVAL", "AVAR", "chem")
    }
    assert chem_ab == {"toy-a": 3.0, "toy-a:wna": 3.0, "toy-fenyves": 3.0}
    gap = {
        (e["source_neuron_id"], e["target_neuron_id"])
        for e in edges
        if e["edge_kind"] == "gap" and e["source_reconstruction"] == "toy-a"
    }
    assert gap == {("AVAL", "AVAR"), ("AVAR", "AVAL"), ("RIML", "DB2"), ("DB2", "RIML")}
    details = rows(out, "edge_details")
    flagged = [
        (e["source_neuron_id"], e["target_neuron_id"])
        for e, d in zip(edges, details, strict=True)
        if d["symmetrised"] and e["source_reconstruction"] == "toy-a"
    ]
    assert flagged == [("DB2", "RIML")]
    info = {r["id"]: r for r in manifest["reconstructions"]}
    assert info["toy-a"]["life_stage"] == "adult" and info["toy-a"]["edges_by_kind"] == {"chem": 4, "gap": 4, "nmj": 2}
    assert info["toy-a"]["dropped"] == {"chem:non_neuron_cell": 1, "gap:mirror_of_neuron_to_muscle": 1}
    assert info["toy-a"]["dropped_top_labels"] == {"GLRL": 1}  # the mirrored muscle gap row is not a label problem


def test_nmj_and_modulatory_edges(built: tuple[Path, dict[str, Any]]) -> None:
    out, _ = built
    edges = rows(out, "edges")
    nmj = {
        (e["source_neuron_id"], e["target_neuron_id"])
        for e in edges
        if e["edge_kind"] == "nmj" and e["source_reconstruction"] == "toy-a"
    }
    assert nmj == {("DB2", "MDL01"), ("AVAL", "MDL01")}
    mod = {
        (e["source_reconstruction"], e["source_neuron_id"], e["target_neuron_id"]): e
        for e in edges
        if e["edge_kind"] == "putative_mod"
    }
    assert mod[("toy-np", "AVAL", "AVAR")]["synapse_count"] is None
    details = rows(out, "edge_details")
    idx = edges.index(mod[("toy-np", "AVAL", "AVAR")])
    assert details[idx]["native_weight"] == 2.0 and details[idx]["transmitters"] == ["flp-1", "flp-2"]
    assert details[idx]["receptors"] == ["npr-11", "npr-4"]
    # Ripoll labels were zero-padded in the file; the alias table resolved VA01
    assert ("toy-ripoll", "AVAR", "VA1") in mod


# --------------------------------------------------------------------------- unresolved-ID audit


def test_unresolved_audit_covers_every_distinct_name_label(built: tuple[Path, dict[str, Any]]) -> None:
    out, manifest = built
    audit = {r["label"]: r for r in rows(out, "unresolved")}
    assert set(audit) == {"AVAL", "AVAR", "rimr", "AWC", "AWCON", "AMsoL", "AIBLL", "ZZZ9"}
    assert (
        audit["AVAL"]["resolution"] == "exact" and audit["AVAL"]["n_rois"] == 2 and audit["AVAL"]["n_recordings"] == 2
    )
    assert audit["AVAR"]["n_rois_label_duplicated"] == 2
    assert audit["rimr"]["resolution"] == "alias" and audit["rimr"]["neuron_id"] == "RIMR" and audit["rimr"]["resolved"]
    awc = audit["AWC"]
    assert (awc["resolution"], awc["candidates"], awc["resolved"], awc["neuron_id"]) == (
        "ambiguous",
        ["AWCL", "AWCR"],
        False,
        None,
    )
    assert audit["AWCON"]["mapping_kind"] == "convention" and not audit["AWCON"]["resolved"]
    assert audit["AMsoL"]["reason"].startswith("known non-neuron cell")
    typo = audit["AIBLL"]
    assert not typo["resolved"] and typo["neuron_id"] is None and typo["suggested_neuron_id"] == "AIBL"
    assert "suggestion only" in typo["reason"]
    assert audit["ZZZ9"]["reason"].startswith("unknown name") and audit["ZZZ9"]["suggested_neuron_id"] is None
    assert {r["mapping_version"] for r in audit.values()} == {MAPPING_VERSION}
    labels = manifest["summary"]["atlas_labels"]
    assert labels["by_resolution"] == {"alias": 1, "ambiguous": 2, "exact": 2, "unresolved": 3}
    assert labels["other_label_status_rois_not_audited"] == {"blank": 1, "numeric": 1, "other": 1}


def test_intersection_counts(built: tuple[Path, dict[str, Any]]) -> None:
    out, manifest = built
    inter = manifest["intersection"]
    # functional: AVAL (exact) and RIMR (alias 'rimr'); AVAR only has duplicated labels and does not count
    assert inter["functional_atlas"] == 2 and inter["canonical_neurons"] == 12
    assert inter["anatomical_any"] == 6  # AVAL AVAR RIML DB2 VA1 RIMR
    assert inter["molecular_annotated"] == 12
    assert inter["three_way_any"] == 2 and inter["three_way_neuron_ids"] == ["AVAL", "RIMR"]
    assert inter["by_reconstruction"]["toy-a"]["with_functional"] == 1
    assert "toy-ma" not in inter["by_reconstruction"]  # modulatory-only reconstructions carry no anatomy
    neurons = {r["neuron_id"]: r for r in rows(out, "neurons")}
    assert neurons["AVAL"]["atlas_n_recordings"] == 2 and neurons["RIMR"]["atlas_n_rois"] == 1
    assert neurons["AVAR"]["atlas_n_rois"] == 0


# --------------------------------------------------------------------------- manifest and guards


def test_manifest_digests_and_counts(built: tuple[Path, dict[str, Any]]) -> None:
    out, manifest = built
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest
    for name, meta in manifest["outputs"].items():
        data = (out / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == meta["sha256"]
        assert pq.ParquetFile(out / name).metadata.num_rows == meta["rows"]
    assert {"neurons.parquet", "aliases.parquet", "edges.parquet", "conflicts.parquet", "unresolved.parquet"} <= set(
        manifest["outputs"]
    )
    assert manifest["summary"]["neurons"] == 12 == manifest["outputs"]["neurons.parquet"]["rows"]
    assert manifest["inputs"] == [{"source": "synthetic"}]
    assert manifest["atlas_input"]["sha256"]
    cross = manifest["cross_checks"]
    assert cross["bundle_pairs"]["toy-a vs toy-a:wna"]["only_b"] >= 1
    assert cross["bentley2016_monoamine_toolbox_vs_wna"] == {"both": 2, "only_toolbox": 0, "only_wna": 0}
    assert cross["wna_neuron_ids"]["ambiguous"] == ["AWCON"]


def test_refuses_to_overwrite_outputs(tmp_path: Path) -> None:
    raw, atlas = make_inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "something").write_text("x")
    with pytest.raises(A.AnnotationError, match="not empty"):
        A.run_build(raw, out, atlas_dir=atlas, inputs=[], recons=RECONS)


def test_missing_atlas_table_is_an_error(tmp_path: Path) -> None:
    raw, _ = make_inputs(tmp_path)
    with pytest.raises(A.AnnotationError, match="atlas ROI table not found"):
        A.run_build(raw, tmp_path / "out", atlas_dir=tmp_path / "nowhere", inputs=[], recons=RECONS)


def test_missing_archive_member_is_an_error(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    _tar(archive, "top", {"a.txt": b"x"})
    assert read_members(archive, ["a.txt"]) == {"a.txt": b"x"}
    with pytest.raises(ArchiveError, match="missing members"):
        read_members(archive, ["a.txt", "b.txt"])


def test_cli_verifies_sources_before_building(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw, atlas = make_inputs(tmp_path)  # synthetic archives do not match the registry's checksums
    code = cli.main(
        ["annotations", "build", "--data-dir", str(raw), "--atlas-dir", str(atlas), "--out", str(tmp_path / "out")]
    )
    assert code == 1
    assert "not building" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit):
        cli.main(["annotations", "import"])


# --------------------------------------------------------------------------- readers


def test_cells_py_is_parsed_statically() -> None:
    lists = ow.static_lists(CELLS_PY)  # the module ends with a raise; it must not run
    assert lists["PREFERRED_HERM_NEURON_NAMES"] == CANONICAL
    assert lists["PREFERRED_MUSCLE_NAMES"] == ["MDL01", "BWM"]
    assert lists["PHARYNGEAL_NEURONS"] == []
    with pytest.raises(ow.ToolboxError, match="lists not found"):
        ow.static_lists(b"X = ['a']\n")


def test_xlsx_reader_handles_shared_and_inline_strings() -> None:
    sheets = read_workbook(make_xlsx({"S": [["a", None, "c"], [], ["x"]]}))
    assert sheets == {"S": [["a", None, "c"], [], ["x"]]}
    wang = ow.read_wang_nt(WANG)
    assert [w.neuron for w in wang][:3] == ["AVAL", "AVAR", "AIBL"]
    avar = wang[1]
    assert avar.cell_class == "AVA" and avar.class_filled_down  # merged cell: value only on the first row
    assert wang[-1].neuron == "AWCR"  # trailing note row ignored
    fen = wa.read_fenyves(FENYVES)
    assert fen.transmitters[1] == wa.FenyvesNT("AVAR", "ACh", "GABA")
    assert fen.edges[2].polarity == "complex" and fen.receptors["AVAL"] == ["glr-1", "acr-2"]


def test_normalize_transmitter_and_function_types() -> None:
    from occamworm.importers.annotation_neurons import function_types, normalize_transmitter

    assert normalize_transmitter("*Glu - NEW ") == ["Glu"]
    assert normalize_transmitter("GABA (uptake)") == ["GABA_uptake"]
    assert normalize_transmitter("5-HT") == ["serotonin"]
    assert normalize_transmitter("DA") == ["dopamine"]
    assert normalize_transmitter("unknown (orphan, unc-47 expression)") == ["unknown_orphan"]
    assert normalize_transmitter("betaine (uptake) ") == ["betaine_uptake"]
    assert function_types("Interneuron, motor neuron") == ["interneuron", "motor"]
    assert function_types("MOtor neuron") == ["motor"]
