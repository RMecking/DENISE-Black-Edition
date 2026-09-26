"""Frozen one-step FP64 state oracle for exact viscoelastic P/SV adjoint."""

from __future__ import annotations

import hashlib
import struct

import pytest

from tests.utilities.visco_psv_one_step_adjoint_reference import (
    ALL_FIELDS,
    CPML_FIELDS,
    MAIN_FIELDS,
    build_fixture,
    inject_receiver_residuals,
    one_reverse_step,
    transpose_errors,
)


# Each digest covers every FP64 value in the named field, including FD4 halos.
# Field-specific hashes localize a failed comparison to one state component.
GOLDEN_SHA256 = {
    ("standard", 1): {
        "avx": "45390b36f5430e5fe3f6feb9534b3de3d8b87bdf4119998851c62b329531ea61",
        "avy": "0ad40cd35845d041a108f403aac686c941fcbfd519a3ef77d4745488d70dfc83",
        "asxx": "f7328ca47f572a463f320a131522ac3a4c141c2610e245d093bcab8d88b1b997",
        "asyy": "f33c05e7aae72a4de2332a1715c8e9a1bd1c6af35d5e3d10d3e3856c97aa085c",
        "asxy": "5626c2841d2f5f0fe09104a805bef7d435259530fac82e3834582677d846b003",
        "ar": "89ef9b0623da19d6dd24e5494804d1c6465b850ee7888d094f1afffbe9f690c9",
        "ap": "d46b83b96489e5c6bdffc14016230596dc35e2a7089bdd5c2013b4a260a6ddb9",
        "aq": "a4eec47a03f21c95c739947ff2f6cc15214049faebb8a535a81a35693cb78286",
        "psxx": "56d06b08467a47aa91e537fb6b3885725914ad9ca4eaa17ecb5a87520c6f21cd",
        "psxyx": "51fb316c6e00000ebe136d8b71b4684aa8123e81c63d148901e65c4229382015",
        "psxyy": "62c0489865ddef8e89ec7982d261c4b021ff5c98f3c8a96df8f7dbda7789df4b",
        "psyy": "28f0002a22fdf1708c3962557c62ce56270ce771c0fcb2fd7156b097c0b1c384",
        "pvxx": "554a17b5b40770ef1c64b5afcfcc1cb0ba3a759a0a4b59418a3a0518afa5e673",
        "pvyx": "84127ce479c6e3de139d1e89639bdc58ba8ce9147e1e2fef5e157e8f2e01d8f1",
        "pvxy": "d16d800645563fa744be2c386038bc73ab4fac7ff7e6b8de0927bedc9dd875e8",
        "pvyy": "6eeb20542ddf711696cbaa4ccf495f8ae92ed4964a6b15500c3c4031f55c2f8f",
    },
    ("standard", 2): {
        "avx": "17201a7d84001e5587df5cac827641849e351b7d76946d77a92eb8fdd930a629",
        "avy": "ecdd5bceb1739d866cb1eae0a8f6a6e1067bdc7e78bd8d8ff6a364c8e927b96d",
        "asxx": "f168d8a10b04b1380087224bb6edcd99d5d6c2603da9b6519da33e203b6ee7cf",
        "asyy": "3902f443e178b30c9e4888c6e598a55b0235731ae6eec9bd0c7a92a311784bcb",
        "asxy": "a1cbab18b71c274f81224994ebcb4daa97b146f75a54540be41bd0e069fa3ce2",
        "ar": "89ef9b0623da19d6dd24e5494804d1c6465b850ee7888d094f1afffbe9f690c9",
        "ap": "d46b83b96489e5c6bdffc14016230596dc35e2a7089bdd5c2013b4a260a6ddb9",
        "aq": "a4eec47a03f21c95c739947ff2f6cc15214049faebb8a535a81a35693cb78286",
        "psxx": "c54673d9690fcc050c0fecd2bcbaf8b66f1b80c34d97065aa42a608d63fa32b5",
        "psxyx": "4ad91031fcb62984cb961bbce2a498d57211a508dc119bf14abd2554f003bf3a",
        "psxyy": "aa10e5eccaaa60101e7ebcb02057d8c273b23a30e2abc442cc8e0cc7f8c0368e",
        "psyy": "a5f76d8f3ddb99000b3c815eb6165734992871abe33379798b6574c487eaf373",
        "pvxx": "554a17b5b40770ef1c64b5afcfcc1cb0ba3a759a0a4b59418a3a0518afa5e673",
        "pvyx": "84127ce479c6e3de139d1e89639bdc58ba8ce9147e1e2fef5e157e8f2e01d8f1",
        "pvxy": "d16d800645563fa744be2c386038bc73ab4fac7ff7e6b8de0927bedc9dd875e8",
        "pvyy": "6eeb20542ddf711696cbaa4ccf495f8ae92ed4964a6b15500c3c4031f55c2f8f",
    },
    ("edge_non_square", 1): {
        "avx": "2cd90a08b11680b9de304621bce46d1a458a338ab9b0ac554980a8a51ba655f5",
        "avy": "ab63441c584dc0ac2badbd6e1451acdb9a3c93fa7e67877907204cb4cc504988",
        "asxx": "a381c9eb65ab1de69f99d539bf6c0034ca08937951a71b7bab9c6361b567965b",
        "asyy": "e467e5896735854a4685ce1d8d266ab6c382ccc3e33d9704c7510ed2c754fbdd",
        "asxy": "9aa48b5055471bf13da661da1e7efe593385034cddb30f3f2cfd7c5cfc9f4721",
        "ar": "dde84d8bb97ebf6c4fcd7006f1cd129f3b1cfcbeb3cdc742332b5a52ed3ac1cb",
        "ap": "80c031228d08f28d48d30f9221f7aac7406a1f2aa01739d7263c133342476046",
        "aq": "8fdf086a69d65421d33c7838ed163cb0008cb724f93b285a0058c9fb621c540d",
        "psxx": "5947d05e17754f9d5df37f4e2303b4111b02592d32f5a9e8fed024ed5c9d8a62",
        "psxyx": "18f8c64f484aee5aa849f71626a04c2d7bce61309d00388cf07eaf60ef570b99",
        "psxyy": "27058dd3b25662ac2c123ecb1b003caec7867af1b02661471f87abe17da176db",
        "psyy": "983966fa07a617165ea39b5d7e8cc379691bc658a89c50fa18e6b0dc6503dea0",
        "pvxx": "d019580c64a390ff48215890ee0c5d7c9165c9429ab99316fb1ca1de5cc38788",
        "pvyx": "2d0f09b92b0b9155613fd00949bf53b5a46bd79b1d008b19126067ee316e9169",
        "pvxy": "5defa783683e5dacc62d0b0d2abf5257166aed45980c3936dcdcceb077e6c55c",
        "pvyy": "29be0ae4bd5dc24c7a685db028e13826665a076d852a695ccedff3337bd7ebaf",
    },
    ("edge_non_square", 2): {
        "avx": "859e38b9d3e8ada5cbf2601abdd5eaafc5d0ed2ca735d88bb9b5cd395250e7e4",
        "avy": "c9e9460af6493211c725b0640db900e7ece120a6ceee28963180d17eb10c6afa",
        "asxx": "6c68041128ab71f24671f6eb0bb91c3e1e627800b5e6bcf7de754acd6bb3359c",
        "asyy": "56128b91fc671cef7ebd80d85c31ef2dd05f6b4e1e5996a79dbe20c19491aa4b",
        "asxy": "fc87e6bd572aa3e2ccbab8c8a6fb236e684a92fffbdb496c38a1eae07bc013bd",
        "ar": "dde84d8bb97ebf6c4fcd7006f1cd129f3b1cfcbeb3cdc742332b5a52ed3ac1cb",
        "ap": "80c031228d08f28d48d30f9221f7aac7406a1f2aa01739d7263c133342476046",
        "aq": "8fdf086a69d65421d33c7838ed163cb0008cb724f93b285a0058c9fb621c540d",
        "psxx": "9d970e96a5cf61cb8300530a35e86f3e950784b9d9edb8af537d4ea82f0594cb",
        "psxyx": "b529c699e5e989bc63f8646df42952046c00b258bc3a98a83a545d087be565db",
        "psxyy": "5a6f992d5546fbaa49faf0330411673f8f826f9db3dbec619623c58a11be6848",
        "psyy": "02aeb791e5c0f8fb5392b02699704408a200bb06d522351ae3cdb2b51e70dc9b",
        "pvxx": "d019580c64a390ff48215890ee0c5d7c9165c9429ab99316fb1ca1de5cc38788",
        "pvyx": "2d0f09b92b0b9155613fd00949bf53b5a46bd79b1d008b19126067ee316e9169",
        "pvxy": "5defa783683e5dacc62d0b0d2abf5257166aed45980c3936dcdcceb077e6c55c",
        "pvyy": "29be0ae4bd5dc24c7a685db028e13826665a076d852a695ccedff3337bd7ebaf",
    },
}

FIXTURES = (("standard", 24, 20, 4), ("edge_non_square", 19, 27, 4))


def _sha256_fp64(values: list[float]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


@pytest.mark.parametrize(
    ("name", "nx", "ny", "fw", "timestep"),
    [(name, nx, ny, fw, t) for name, nx, ny, fw in FIXTURES for t in (1, 2)],
)
def test_one_reverse_step_matches_field_level_fp64_goldens(
    name: str, nx: int, ny: int, fw: int, timestep: int
) -> None:
    fixture = build_fixture(name, nx, ny, fw)
    output = one_reverse_step(fixture, timestep)
    assert tuple(output) == ALL_FIELDS
    assert set(MAIN_FIELDS).isdisjoint(CPML_FIELDS)

    expected_count = (nx + 6) * (ny + 6)
    for field in ALL_FIELDS:
        assert len(output[field]) == expected_count
        actual = _sha256_fp64(output[field])
        assert actual == GOLDEN_SHA256[(name, timestep)][field], (
            f"{name} t={timestep} field={field}: expected "
            f"{GOLDEN_SHA256[(name, timestep)][field]}, got {actual}"
        )


@pytest.mark.parametrize(("name", "nx", "ny", "fw"), FIXTURES)
def test_stress_gsls_and_velocity_transpose_identities(
    name: str, nx: int, ny: int, fw: int
) -> None:
    errors = transpose_errors(build_fixture(name, nx, ny, fw))
    # Dot products use math.fsum in FP64.  This bound admits summation
    # roundoff only, not stencil, sign, staggering, or CPML errors.
    assert errors["stress_gsls_cpml"] <= 5.0e-14, errors
    assert errors["velocity_cpml"] <= 5.0e-14, errors


@pytest.mark.parametrize(("name", "nx", "ny", "fw"), FIXTURES)
def test_fixture_covers_interior_and_all_cpml_sides(
    name: str, nx: int, ny: int, fw: int
) -> None:
    fixture = build_fixture(name, nx, ny, fw)
    assert nx - 2 * fw > 0 and ny - 2 * fw > 0
    for field in ("pvxx", "psxx"):
        values = fixture.fields[field]
        assert any(values[fixture.index(j, i)] != 0.0
                   for j in range(1, ny + 1) for i in range(1, fw + 1))
        assert any(values[fixture.index(j, i)] != 0.0
                   for j in range(1, ny + 1) for i in range(nx - fw + 1, nx + 1))
    for field in ("pvyy", "psyy"):
        values = fixture.fields[field]
        assert any(values[fixture.index(j, i)] != 0.0
                   for j in range(1, fw + 1) for i in range(1, nx + 1))
        assert any(values[fixture.index(j, i)] != 0.0
                   for j in range(ny - fw + 1, ny + 1) for i in range(1, nx + 1))
    assert fixture.index(1, fw + 1) != fixture.index(ny, nx - fw)


@pytest.mark.parametrize(("name", "nx", "ny", "fw"), FIXTURES)
def test_receiver_residual_injection_excludes_production_sample_one(
    name: str, nx: int, ny: int, fw: int
) -> None:
    fixture = build_fixture(name, nx, ny, fw)
    no_injection = {field: values.copy() for field, values in fixture.fields.items()}
    at_sample_one = {field: values.copy() for field, values in fixture.fields.items()}
    at_sample_two = {field: values.copy() for field, values in fixture.fields.items()}
    inject_receiver_residuals(fixture, at_sample_one, 1)
    inject_receiver_residuals(fixture, at_sample_two, 2)
    assert at_sample_one == no_injection
    for j, i, modeled_vx, observed_vx, modeled_vy, observed_vy in fixture.receivers:
        p = fixture.index(j, i)
        assert at_sample_two["avx"][p] == no_injection["avx"][p] + (modeled_vx - observed_vx)
        assert at_sample_two["avy"][p] == no_injection["avy"][p] + (modeled_vy - observed_vy)
    changed = [p for p, pair in enumerate(zip(no_injection["avx"], at_sample_two["avx"]))
               if pair[0] != pair[1]]
    assert changed == [fixture.index(j, i) for j, i, *_ in fixture.receivers]
